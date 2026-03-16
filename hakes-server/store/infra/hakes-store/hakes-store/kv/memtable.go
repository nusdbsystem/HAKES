/*
 * Copyright 2024 The HAKES Authors
 * Copyright 2020 Dgraph Labs, Inc. and Contributors
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
	"sort"
	"strconv"
	"strings"

	"github.com/dgraph-io/badger/v3/skl"
	"github.com/dgraph-io/badger/v3/y"
)

// memTable structure stores a skiplist and a corresponding WAL. Writes to memTable are written
// both to the WAL and the skiplist. On a crash, the WAL is replayed to bring the skiplist back to
// its pre-crash form.
type memTable struct {
	sl         *skl.Skiplist
	wal        *logFile
	maxVersion uint64
	disableWAL bool
	buf        *bytes.Buffer
}

func (db *DB) openMemTables(opt Options, minWal uint64) error {
	ids, err := db.opt.dlogHandler.List()
	if err != nil {
		return y.Wrapf(err, "Unable to list mem backing wal.")
	}

	var fids []int
	for _, id := range ids {
		if !strings.HasSuffix(id, memFileExt) {
			continue
		}
		fsz := len(id)
		fid, err := strconv.ParseInt(id[:fsz-len(memFileExt)], 10, 64)
		if err != nil {
			return y.Wrapf(err, "Unable to parse log id %v.", fid)
		}
		if uint64(fid) < minWal {
			// this wal already persisted
			db.opt.dlogHandler.Drop(id)
			continue
		}
		fids = append(fids, int(fid))
	}

	// Sort in ascending order.
	sort.Slice(fids, func(i, j int) bool {
		return fids[i] < fids[j]
	})
	for _, fid := range fids {
		mt, err := db.reopenMemTable(fid)
		if err != nil {
			return y.Wrapf(err, "while opening fid: %d", fid)
		}
		// If this memtable is empty we don't need to add it. This is a
		// memtable that was completely truncated.
		if mt.sl.Empty() {
			mt.DecrRef()
			continue
		}
		// These should no longer be written to. So, make them part of the imm.
		db.imm = append(db.imm, mt)
	}
	if len(fids) != 0 {
		db.nextMemFid = fids[len(fids)-1]
	} else {
		db.nextMemFid = int(minWal)
	}
	db.nextMemFid++
	return nil
}

func (db *DB) openNewMemTable(fid int) (*memTable, error) {
	path := getWalName(fid)
	s := skl.NewSkiplist(arenaSize(db.opt))
	mt := &memTable{
		sl:         s,
		buf:        &bytes.Buffer{},
		disableWAL: db.opt.DisableWAL,
	}

	mt.wal = &logFile{
		path:    path,
		writeAt: walHeaderSize,
	}
	if mt.disableWAL {
		return mt, nil
	}
	// prepare wal
	if lerr := mt.wal.openNew(path, 2*db.opt.MemTableSize, db.opt.dlogHandler); lerr != nil {
		return nil, y.Wrapf(lerr, "While opening new memtable: %s", path)
	}

	// Have a callback set to delete WAL when skiplist reference count goes down to zero. That is,
	// when it gets flushed to L0.
	s.OnClose = func() {
		if err := mt.wal.Close(); err != nil {
			db.opt.Errorf("while deleting file: %s, err: %v", path, err)
		}
		if err := db.opt.dlogHandler.Drop(path); err != nil {
			db.opt.Errorf("while deleting file: %s, err: %v", path, err)
		}
	}

	return mt, nil
}

// disable wal does not affect reopenMemtable which can recover from the previous openning that had wal enabled.
func (db *DB) reopenMemTable(fid int) (*memTable, error) {
	path := getWalName(fid)
	s := skl.NewSkiplist(arenaSize(db.opt))
	mt := &memTable{
		sl:  s,
		buf: &bytes.Buffer{},
	}

	mt.wal = &logFile{
		path:    path,
		writeAt: walHeaderSize,
	}
	if lerr := mt.wal.reopen(path, 2*db.opt.MemTableSize, db.opt.dlogHandler); lerr != nil {
		return nil, y.Wrapf(lerr, "While reopening memtable: %s", path)
	}

	// Have a callback set to delete WAL when skiplist reference count goes down to zero. That is,
	// when it gets flushed to L0.
	s.OnClose = func() {
		if err := mt.wal.Close(); err != nil {
			db.opt.Errorf("while deleting file: %s, err: %v", path, err)
		}
		if err := db.opt.dlogHandler.Drop(path); err != nil {
			db.opt.Errorf("while deleting file: %s, err: %v", path, err)
		}
	}

	err := mt.UpdateSkipList()
	return mt, y.Wrapf(err, "while updating skiplist")
}

func (db *DB) newMemTable() (*memTable, error) {
	if mt, err := db.openNewMemTable(db.nextMemFid); err != nil {
		db.opt.Errorf("Got error: %v for id: %d\n", err, db.nextMemFid)
		return nil, y.Wrapf(err, "newMemTable")
	} else {
		db.nextMemFid++
		return mt, nil
	}
}

func (mt *memTable) SyncWAL() error {
	if mt.disableWAL {
		return nil
	}
	return mt.wal.Sync()
}

func (mt *memTable) exceedSize(threshold int64) bool {
	if mt.sl.MemSize() >= threshold {
		return true
	}
	return !mt.disableWAL && int64(mt.wal.writeAt) >= threshold
}

func (mt *memTable) Put(key []byte, value y.ValueStruct) error {
	entry := &Entry{
		Key:       key,
		Value:     value.Value,
		UserMeta:  value.UserMeta,
		meta:      value.Meta,
		ExpiresAt: value.ExpiresAt,
	}

	if !mt.disableWAL {
		if err := mt.wal.writeEntry(mt.buf, entry); err != nil {
			return y.Wrapf(err, "cannot write entry to WAL file")
		}
	}
	// We insert the finish marker in the WAL but not in the memtable.
	if entry.meta&bitFinTxn > 0 {
		return nil
	}

	// Write to skiplist and update maxVersion encountered.
	mt.sl.Put(key, value)
	if ts := y.ParseTs(entry.Key); ts > mt.maxVersion {
		mt.maxVersion = ts
	}
	return nil
}

func (mt *memTable) UpdateSkipList() error {
	if mt.wal == nil || mt.sl == nil {
		return nil
	}
	_, err := mt.wal.iterate(true, 0, mt.replayFunction())
	if err != nil {
		return y.Wrapf(err, "while iterating wal: %s", mt.wal.path)
	}
	return nil
}

// IncrRef increases the refcount
func (mt *memTable) IncrRef() {
	mt.sl.IncrRef()
}

// DecrRef decrements the refcount, deallocating the Skiplist when done using it
func (mt *memTable) DecrRef() {
	mt.sl.DecrRef()
}

func (mt *memTable) replayFunction() func(Entry) error {
	return func(e Entry) error { // Function for replaying.
		if ts := y.ParseTs(e.Key); ts > mt.maxVersion {
			mt.maxVersion = ts
		}
		v := y.ValueStruct{
			Value:     e.Value,
			Meta:      e.meta,
			UserMeta:  e.UserMeta,
			ExpiresAt: e.ExpiresAt,
		}
		// This is already encoded correctly. Value would be either a vptr, or a full value
		// depending upon how big the original value was. Skiplist makes a copy of the key and
		// value.
		mt.sl.Put(e.Key, v)
		return nil
	}
}
