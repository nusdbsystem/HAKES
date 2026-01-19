/*
 * Copyright 2024 The HAKES Authors
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
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"

	kvio "hakes-store/hakes-store/io"

	"github.com/dgraph-io/badger/v3/y"
)

type ManifestHandler interface {
	// returns the latest manifest content
	// otherwise return nil and ErrEmptyManifest (for empty manifest)/other errors (db will stop)
	Reopen() ([]byte, error)
	// input ownership is transferred to manifest handler, the caller no longer modify it afterwards
	Update([]byte) error
	GetStats() string
	// not used
	Sync()
	// not used
	Close() error
}

const (
	ManifestFilename0 = "MANIFEST0"
	ManifestFilename1 = "MANIFEST1"
)

func pickNextManifestToWrite(cur string) string {
	if cur == "" {
		return ManifestFilename0
	}
	y.AssertTrue((cur == ManifestFilename0) || (cur == ManifestFilename1))
	if cur == ManifestFilename0 {
		return ManifestFilename1
	} else {
		return ManifestFilename0
	}
}

// openManifest open an existing Manifest and check if the first entry is valid
// readManifestEntry do checksum here, if failed it means failure happended while we are writing the new file
func openManifest(name string, dir string) (*os.File, uint64, error) {
	path := filepath.Join(dir, name)
	fp, err := y.OpenExistingFile(path, 0)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, 0, ErrEmptyManifest
		}
		return nil, 0, y.Wrap(err, "fail to open manifest")
	}
	// get the timestamp
	ts, _, _, err := readManifestHeader(fp)
	if err != nil {
		return nil, 0, y.Wrap(err, "fail to validate header during manifest open")
	}
	// check the first entry validity
	fileInfo, err := fp.Stat()
	if err != nil || fileInfo.Size() <= manifestHeaderSize {
		return nil, 0, y.Wrap(err, "fail to get the manifest size")
	}
	if _, err = readManifestEntry(fp, uint32(fileInfo.Size())); err != nil {
		return nil, 0, ErrEmptyManifest
	}
	fp.Seek(0, io.SeekStart)
	return fp, ts, nil
}

func openLatestManifest(dir string) (*os.File, string, error) {
	fp0, ts0, err0 := openManifest(ManifestFilename0, dir)
	fp1, ts1, err1 := openManifest(ManifestFilename1, dir)
	if (err0 != nil) && (err1 != nil) {
		// no valid manifest file have ever been created. crash during open or there is no manifest file
		if err0 == ErrEmptyManifest {
			return nil, "", ErrEmptyManifest
		}
		return nil, "", fmt.Errorf("both manifest are invalid: 0: %v, 1: %v", err0, err1)
	}
	// not possible to have two manifest with exactly same timestamp
	y.AssertTrue(ts0 != ts1)
	if ts0 > ts1 {
		fp1.Close()
		return fp0, ManifestFilename0, nil
	} else {
		fp0.Close()
		return fp1, ManifestFilename1, nil
	}
}

var _ ManifestHandler = (*LFManifestHandler)(nil)

type LFManifestHandler struct {
	dir             string
	curManifestName string
	stats           kvio.FSStats
}

func NewLFManifestHandler(dir string) *LFManifestHandler {
	return &LFManifestHandler{
		dir: dir,
	}
}

func (h *LFManifestHandler) GetStats() string {
	return h.stats.Encode()
}

// return the latest manifest content, follows the usage pattern of single instance deployment that opens the latest manifest when reopened.
func (h *LFManifestHandler) Reopen() ([]byte, error) {
	fp, manifestName, err := openLatestManifest(h.dir)
	h.curManifestName = manifestName
	if err != nil {
		return nil, err
	}
	fileInfo, err := fp.Stat()
	if err != nil {
		return nil, err
	}
	logSize := fileInfo.Size()
	ret := make([]byte, logSize)
	if readLen, err := fp.Read(ret); readLen != int(logSize) || err != nil {
		return nil, err
	}
	fp.Close()
	h.stats.AddReadSize(len(ret))
	return ret, nil
}

func (h *LFManifestHandler) Update(data []byte) error {
	targetName := pickNextManifestToWrite(h.curManifestName)
	path := filepath.Join(h.dir, targetName)
	fp, err := y.OpenTruncFile(path, false)
	if err != nil {
		return err
	}
	if _, err := fp.Write(data); err != nil {
		fp.Close()
		return err
	}
	if err := fp.Sync(); err != nil {
		fp.Close()
		return err
	}
	h.stats.AddWriteSize(len(data))
	h.curManifestName = targetName
	fp.Close()
	return nil
}

func (h *LFManifestHandler) Sync() {
	// sync upon update, so no-op here
}

func (h *LFManifestHandler) Close() error {
	return nil
}

// a new type of manifest handler that uses sequential increasing file
const (
	ManifestPrefix = "MANIFEST-"
)

func CreateSeqManifestName(id int) string {
	return fmt.Sprintf("%v%d", ManifestPrefix, id)
}

var (
	errInvalidManifestName = errors.New("invalid manifest name")
)

func GetSeqManifestID(in string) (int, error) {
	if len(in) <= len(ManifestPrefix) {
		return -1, errInvalidManifestName
	}
	return strconv.Atoi(in[len(ManifestPrefix):])
}

func openLatestSeqManifest(dir string) (*os.File, int, error) {
	fileInfos, err := os.ReadDir(dir)
	if err != nil {
		return nil, -1, err
	}
	maxID := -1
	toOpen := ""
	for _, n := range fileInfos {
		if n.IsDir() {
			continue
		}
		fileName := n.Name()
		if cur, err := GetSeqManifestID(fileName); err == nil && cur > maxID {
			maxID = cur
			toOpen = fileName
		}
	}
	if maxID == -1 {
		// no manifest
		return nil, -1, ErrEmptyManifest
	}
	if fp, _, err := openManifest(toOpen, dir); err == nil {
		return fp, maxID, nil
	} else if err != ErrEmptyManifest {
		return nil, -1, err
	} else {
		fp.Close()
		path := filepath.Join(dir, toOpen)
		os.Remove(path)
		// the new manifest has no valid entry, we should try to open the last one
		// should not have been deleted
		if maxID == 0 {
			return nil, -1, err
		}
		fp, _, err := openManifest(CreateSeqManifestName(maxID-1), dir)
		return fp, maxID - 1, err
	}
}

var _ ManifestHandler = (*SeqLFManifestHandler)(nil)

type SeqLFManifestHandler struct {
	dir           string
	curManifestId int
	stats         kvio.FSStats
}

func NewSeqLFManifestHandler(dir string) *SeqLFManifestHandler {
	return &SeqLFManifestHandler{
		dir: dir,
	}
}

func (h *SeqLFManifestHandler) GetStats() string {
	return h.stats.Encode()
}

// return the latest manifest content, follows the usage pattern of single instance deployment that opens the latest manifest when reopened.
func (h *SeqLFManifestHandler) Reopen() ([]byte, error) {
	// todo
	fp, manifestID, err := openLatestSeqManifest(h.dir)
	h.curManifestId = manifestID
	if err != nil {
		return nil, err
	}
	fileInfo, err := fp.Stat()
	if err != nil {
		return nil, err
	}
	logSize := fileInfo.Size()
	ret := make([]byte, logSize)
	if readLen, err := fp.Read(ret); readLen != int(logSize) || err != nil {
		return nil, err
	}
	fp.Close()
	h.stats.AddReadSize(len(ret))
	return ret, nil
}

func (h *SeqLFManifestHandler) Update(data []byte) error {
	targetName := CreateSeqManifestName(h.curManifestId + 1)
	path := filepath.Join(h.dir, targetName)
	fp, err := y.OpenTruncFile(path, false)
	if err != nil {
		return err
	}
	if _, err := fp.Write(data); err != nil {
		fp.Close()
		return err
	}
	if err := fp.Sync(); err != nil {
		fp.Close()
		return err
	}
	h.stats.AddWriteSize(len(data))
	fp.Close()
	// drop the last file once new one is durable
	dropPath := filepath.Join(h.dir, CreateSeqManifestName(h.curManifestId))
	os.Remove(dropPath)
	h.curManifestId++
	return nil
}

func (h *SeqLFManifestHandler) Sync() {
	// sync upon update, so no-op here
}

func (h *SeqLFManifestHandler) Close() error {
	return nil
}
