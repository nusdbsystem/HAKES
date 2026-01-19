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

	"hakes-store/hakes-store/io"
	"hakes-store/hakes-store/table"

	"github.com/dgraph-io/badger/v3/y"
)

type HakesKV struct {
	db   *DB
	opts Options
}

func NewHakesKV(path string) *HakesKV {
	// default setting all using local fs
	cssCli := &io.FSCli{}
	cssCli.Connect(path)
	dlogHandler := &io.FDLogHandler{}
	dlogHandler.Connect(path)
	mh := NewSeqLFManifestHandler(path)
	opt := DefaultOptions().WithCSSCli(cssCli).WithDLogHandler(dlogHandler).WithManifestHandler(mh)
	return NewHakeSKVWithOpts(opt)
}

func NewHakeSKVWithOpts(opts Options) *HakesKV {
	return &HakesKV{db: nil, opts: opts}
}

func (n *HakesKV) Open() error {
	if db, err := Open(n.opts); err != nil {
		return err
	} else {
		n.db = db
		return nil
	}
}

func (n *HakesKV) Put(key, val []byte) error {
	// we need to increase ment this value which is used as a discardTs upperbound below which records can be removed
	defer n.db.orc.readMark.SetDoneUntil(rtts.GetTs())
	e := NewEntry(key, val)
	return n.db.batchSet([]*Entry{e})
}

func (n *HakesKV) PutAsync(key, val []byte) error {
	// with no txn support, there is no view maintenance
	// we need to increase ment this value which is used as a discardTs upperbound below which records can be removed
	defer n.db.orc.readMark.SetDoneUntil(rtts.GetTs())
	e := NewEntry(key, val)
	// log on error
	return n.db.batchSetAsync(
		[]*Entry{e},
		func(e error) { n.opts.Logger.Errorf(e.Error()) },
	)
}

func (n *HakesKV) Get(key []byte) ([]byte, error) {
	seekKey := y.KeyWithTs(key, rtts.GetTs())
	vs, err := n.db.get(seekKey)
	if err != nil {
		return nil, errors.New("internal get error")
	}
	if vs.Value == nil && vs.Meta == 0 {
		return nil, ErrKeyNotFound
	}
	if isDeletedOrExpired(vs.Meta, vs.ExpiresAt) {
		return nil, ErrKeyNotFound
	}
	// there is much to do here to check if it is value pointer and load from value log.
	if (vs.Meta & bitValuePointer) != 0 {
		var vp valueBlobPtr
		vp.Decode(vs.Value)
		vBlob, vbcb, err := n.db.getValueBlob(key, vp)
		if err != nil {
			return nil, err
		}
		defer runCallback(vbcb)
		dst := make([]byte, 0)
		return y.SafeCopy(dst, vBlob), err

	} else {
		value := make([]byte, 0)
		return y.SafeCopy(value, vs.Value), nil
	}
}

func (n *HakesKV) Delete(key []byte) error {
	defer n.db.orc.readMark.SetDoneUntil(rtts.GetTs())
	d := &Entry{
		Key:  key,
		meta: bitDelete,
	}
	return n.db.batchSet([]*Entry{d})
}

func (n *HakesKV) NewIterator(opts IteratorOptions) *Iterator {
	tables, decr := n.db.getMemTables()
	defer decr()

	var iters []y.Iterator
	for i := 0; i < len(tables); i++ {
		iters = append(iters, tables[i].sl.NewUniIterator(opts.Reverse))
	}
	iters = n.db.lc.appendIterators(iters, &opts)
	return &Iterator{
		iitr:   table.NewMergeIterator(iters, opts.Reverse),
		opt:    opts,
		db:     n.db,
		readTs: rtts.GetTs(),
	}
}

func (n *HakesKV) Close() error {
	return n.db.Close()
}
