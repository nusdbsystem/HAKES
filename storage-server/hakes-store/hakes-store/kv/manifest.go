/*
 * Copyright 2024 The HAKES Authors
 * Copyright 2017 Dgraph Labs, Inc. and Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package kv

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"log"
	"sync"
	"time"

	pb "hakes-store/hakes-store/kv/proto"

	"github.com/dgraph-io/badger/v3/options"
	"github.com/dgraph-io/badger/v3/y"
	"github.com/golang/protobuf/proto"
	"github.com/pkg/errors"
)

// Manifest represents the contents of the MANIFEST file in a Badger store.
//
// The MANIFEST file describes the startup state of the db -- all LSM files and what level they're
// at.
//
// It consists of a ManifestChangeSet object.
// It contains a sequence of ManifestChange's (file creations/deletions)
// which we use to reconstruct the manifest at startup.
type Manifest struct {
	minWal uint64 // keep track of the WAL discard; wal number < minWal no replay during recovery
	epoch  uint64

	Levels []levelManifest
	Tables map[string]TableManifest
}

func createManifest() Manifest {
	levels := make([]levelManifest, 0)
	return Manifest{
		Levels: levels,
		Tables: make(map[string]TableManifest),
	}
}

func (m *Manifest) setMinWal(mw uint64) {
	m.minWal = mw
}

func (m *Manifest) setEpoch(e uint64) {
	m.epoch = e
}

func (m *Manifest) GetEpoch() uint64 {
	return m.epoch
}

// levelManifest contains information about LSM tree levels
// in the MANIFEST file.
type levelManifest struct {
	Tables map[string]struct{} // Set of table id's
}

// TableManifest contains information about a specific table
// in the LSM tree.
type TableManifest struct {
	Level       uint8
	Compression options.CompressionType
}

// manifestFile holds the file pointer (and other info) about the manifest file, which is a log
// file we append to.
type manifestFile struct {
	// Guards appends, which includes access to the manifest field.
	appendLock sync.Mutex

	// Used to track the current state of the manifest, used when rewriting.
	manifest Manifest
	handler  ManifestHandler
}

const (
	manifestHeaderSize = 4 + 4 + 8
)

// asChanges returns a sequence of changes that could be used to recreate the Manifest in its
// present state.
func (m *Manifest) asChanges() []*pb.ManifestChange {
	changes := make([]*pb.ManifestChange, 0, len(m.Tables))
	for id, tm := range m.Tables {
		changes = append(changes, newCreateChange(id, int(tm.Level), tm.Compression))
	}
	return changes
}

func (m *Manifest) clone() Manifest {
	changeSet := pb.ManifestChangeSet{Changes: m.asChanges()}
	ret := createManifest()
	ret.setMinWal(m.minWal)
	ret.setEpoch(m.epoch)
	y.Check(applyChangeSet(&ret, &changeSet))
	return ret
}

func openOrCreateManifestFile(opt Options) (
	ret *manifestFile, result Manifest, err error) {
	return helpOpenOrCreateManifestFile(opt)
}

func helpOpenOrCreateManifestFile(opt Options) (
	*manifestFile, Manifest, error) {

	lastManifest, err := opt.manifestHandler.Reopen()

	var m Manifest
	if err != nil {
		if err != ErrEmptyManifest {
			return nil, Manifest{}, err
		}
		// no existing manifest file.
		// no existing manifest file, create a new one.
		m = createManifest()
		m.setMinWal(0)
		m.setEpoch(0)
	} else if m, err = ReplayManifestFile(lastManifest); err != nil {
		log.Printf("replay latest manifest error, cannot restore the data store to valid state")
		return nil, Manifest{}, err
	}

	m.epoch++ // increment the epoch after reopen
	err = helpRewrite(&m, opt.manifestHandler)
	if err != nil {
		return nil, Manifest{}, err
	}

	mf := &manifestFile{
		manifest: m.clone(),
		handler:  opt.manifestHandler,
	}
	return mf, m, nil
}

func (mf *manifestFile) close() error {
	return mf.handler.Close()
}

// addChanges writes a batch of changes, atomically, to the file.  By "atomically" that means when
// we replay the MANIFEST file, we'll either replay all the changes or none of them.  (The truth of
// this depends on the filesystem -- some might append garbage data if a system crash happens at
// the wrong time.)
func (mf *manifestFile) addChanges(changesParam []*pb.ManifestChange) error {
	changes := pb.ManifestChangeSet{Changes: changesParam}
	// Maybe we could use O_APPEND instead (on certain file systems)
	mf.appendLock.Lock()
	defer mf.appendLock.Unlock()
	if err := applyChangeSet(&mf.manifest, &changes); err != nil {
		return err
	}
	// always rewrite
	// Rewrite manifest if it'd shrink by 1/10 and it's big enough to care
	if err := mf.rewrite(); err != nil {
		return err
	}
	return nil
}

// Has to be 4 bytes.  The value can never change, ever, anyway.
var magicText = [4]byte{'B', 'd', 'g', 'r'}

// The magic version number.
const magicVersion = 8

func helpRewrite(m *Manifest, handler ManifestHandler) error {
	changes := m.asChanges()
	set := pb.ManifestChangeSet{Changes: changes}

	changeBuf, err := proto.Marshal(&set)
	if err != nil {
		return err
	}

	buf := make([]byte, 8*5+len(changeBuf))
	copy(buf[0:4], magicText[:])
	binary.BigEndian.PutUint32(buf[4:8], magicVersion)
	startoff := 8

	// add the timestamp entry
	ts := time.Now().UnixNano()
	binary.BigEndian.PutUint64(buf[startoff:startoff+8], uint64(ts))
	startoff += 8

	// add the min wal entry
	binary.BigEndian.PutUint64(buf[startoff:startoff+8], uint64(m.minWal))
	startoff += 8

	// add the epoch entry
	binary.BigEndian.PutUint64(buf[startoff:startoff+8], uint64(m.epoch))
	startoff += 8

	// change append to copy for crc
	binary.BigEndian.PutUint32(buf[startoff:startoff+4], uint32(len(changeBuf)))
	binary.BigEndian.PutUint32(buf[startoff+4:startoff+8], crc32.Checksum(changeBuf, y.CastagnoliCrcTable))
	startoff += 8
	// change append to copy for change buf
	copy(buf[startoff:], changeBuf)

	return handler.Update(buf)
}

// Must be called while appendLock is held.
func (mf *manifestFile) rewrite() error {
	// In Windows the files should be closed before doing a Rename.
	err := helpRewrite(&mf.manifest, mf.handler)
	if err != nil {
		return err
	}
	return nil
}

var (
	errBadMagic    = errors.New("manifest has bad magic")
	errBadChecksum = errors.New("manifest has checksum mismatch")
	// no valid entry but with valid header
	ErrEmptyManifest = errors.New("empty manifest")
)

// check validity of header and return the creation time
func readManifestHeader(r io.Reader) (uint64, uint64, uint64, error) {
	var headerBuf [8 + 8 + 8 + 8]byte
	if _, err := io.ReadFull(r, headerBuf[:]); err != nil {
		return 0, 0, 0, errBadMagic
	}
	if !bytes.Equal(headerBuf[0:4], magicText[:]) {
		return 0, 0, 0, errBadMagic
	}
	version := y.BytesToU32(headerBuf[4:8])
	if version != magicVersion {
		return 0, 0, 0,
			//nolint:lll
			fmt.Errorf("manifest has unsupported version: %d (we support %d).\n"+
				"Please see https://github.com/dgraph-io/badger/blob/master/README.md#i-see-manifest-has-unsupported-version-x-we-support-y-error"+
				" on how to fix this",
				version, magicVersion)
	}
	ts := y.BytesToU64(headerBuf[8:16])
	minWal := y.BytesToU64(headerBuf[16:24])
	epoch := y.BytesToU64(headerBuf[24:32])
	return ts, minWal, epoch, nil
}

func readManifestEntry(r io.Reader, maxSize uint32) ([]byte, error) {
	var lenCrcBuf [8]byte
	if _, err := io.ReadFull(r, lenCrcBuf[:]); err != nil {
		return nil, err
	}
	length := y.BytesToU32(lenCrcBuf[0:4])
	// Sanity check to ensure we don't over-allocate memory.
	if length > maxSize {
		return nil, errors.Errorf(
			"Buffer length: %d greater than max size: %d. Manifest might be corrupted",
			length, maxSize)
	}
	var buf = make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	if crc32.Checksum(buf, y.CastagnoliCrcTable) != y.BytesToU32(lenCrcBuf[4:8]) {
		return nil, errBadChecksum
	}
	return buf, nil
}

func ReplayManifestFile(data []byte) (Manifest, error) {
	logSize := len(data)
	r := bytes.NewReader(data)

	_, minWal, epoch, err := readManifestHeader(r)
	if err != nil {
		return Manifest{}, err
	}

	build := createManifest()
	build.setMinWal(minWal)
	build.setEpoch(epoch)
	for {
		buf, err := readManifestEntry(r, uint32(logSize))
		if err != nil {
			// read the last entry
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				break
			}
			// corruption
			return Manifest{}, err
		}

		var changeSet pb.ManifestChangeSet
		if err := proto.Unmarshal(buf, &changeSet); err != nil {
			return Manifest{}, err
		}

		if err := applyChangeSet(&build, &changeSet); err != nil {
			return Manifest{}, err
		}
	}

	return build, nil
}

func applyManifestChange(build *Manifest, tc *pb.ManifestChange) error {
	// creating level handler to reach base level at the beginning
	ensureLevelExists := func(level int) {
		for len(build.Levels) <= int(level) {
			build.Levels = append(build.Levels, levelManifest{make(map[string]struct{})})
		}
	}
	switch tc.Op {
	case pb.ManifestChange_CREATE:
		if _, ok := build.Tables[tc.Id]; ok {
			return fmt.Errorf("MANIFEST invalid, table %v exists", tc.Id)
		}
		build.Tables[tc.Id] = TableManifest{
			Level:       uint8(tc.Level),
			Compression: options.CompressionType(tc.Compression),
		}
		ensureLevelExists(int(tc.Level))
		build.Levels[tc.Level].Tables[tc.Id] = struct{}{}
	case pb.ManifestChange_DELETE:
		tm, ok := build.Tables[tc.Id]
		if !ok {
			return fmt.Errorf("MANIFEST removes non-existing table %v", tc.Id)
		}
		delete(build.Levels[tm.Level].Tables, tc.Id)
		delete(build.Tables, tc.Id)
	case pb.ManifestChange_MOVE:
		// checking
		tm, ok := build.Tables[tc.Id]
		if !ok {
			return fmt.Errorf("Manifest invalid, table %v to move non-exist", tc.Id)
		}
		fromLevel := tm.Level
		build.Tables[tc.Id] = TableManifest{
			Level:       uint8(tc.Level),
			Compression: options.CompressionType(tc.Compression),
		}
		ensureLevelExists(int(tc.Level))
		build.Levels[tc.Level].Tables[tc.Id] = struct{}{}
		delete(build.Levels[fromLevel].Tables, tc.Id)
	default:
		return fmt.Errorf("MANIFEST file has invalid manifestChange op")
	}
	return nil
}

// This is not a "recoverable" error -- opening the KV store fails because the MANIFEST file is
// just plain broken.
func applyChangeSet(build *Manifest, changeSet *pb.ManifestChangeSet) error {
	for _, change := range changeSet.Changes {
		if err := applyManifestChange(build, change); err != nil {
			return err
		}
	}
	return nil
}

func newCreateChange(
	id string, level int, c options.CompressionType) *pb.ManifestChange {
	return &pb.ManifestChange{
		Id:          id,
		Op:          pb.ManifestChange_CREATE,
		Level:       uint32(level),
		Compression: uint32(c),
	}
}

func newDeleteChange(id string) *pb.ManifestChange {
	return &pb.ManifestChange{
		Id: id,
		Op: pb.ManifestChange_DELETE,
	}
}

func newMoveChange(id string, targetLevel int, c options.CompressionType) *pb.ManifestChange {
	return &pb.ManifestChange{
		Id:          id,
		Op:          pb.ManifestChange_MOVE,
		Level:       uint32(targetLevel),
		Compression: uint32(c),
	}
}
