/*
 * Copyright 2024 The HAKES Authors
 * Copyright 2019 Dgraph Labs, Inc. and Contributors
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
	"encoding/binary"
	"fmt"
	"time"
)

// Values have their first byte being byteData or byteDelete. This helps us distinguish between
// a key that has never been seen and a key that has been explicitly deleted.
const (
	bitDelete                 byte = 1 << 0 // Set if the key has been deleted.
	bitValuePointer           byte = 1 << 1 // Set if the value is NOT stored directly next to key.
	bitDiscardEarlierVersions byte = 1 << 2 // Set if earlier versions can be discarded.
	// Set if item shouldn't be discarded via compactions (used by merge operator)
	bitMergeEntry byte = 1 << 3
	// The MSB 2 bits are for transactions.
	bitTxn    byte = 1 << 6 // Set if the entry is part of a txn.
	bitFinTxn byte = 1 << 7 // Set if the entry is to indicate end of txn in value log.
)

func isDeletedOrExpired(meta byte, expiresAt uint64) bool {
	if meta&bitDelete > 0 {
		return true
	}
	if expiresAt == 0 {
		return false
	}
	return expiresAt <= uint64(time.Now().Unix())
}

// Entry provides Key, Value, UserMeta and ExpiresAt. This struct can be used by
// the user to set data.
type Entry struct {
	Key       []byte
	Value     []byte
	ExpiresAt uint64 // time.Unix
	UserMeta  byte

	// Fields maintained internally.
	offset uint32 // offset is an internal field.
	meta   byte
	hlen   int // Length of the header.
}

func (e *Entry) isZero() bool {
	return len(e.Key) == 0
}

func (db *DB) valueThreshold() int64 {
	// only static threshold
	return db.opt.ValueThreshold
}

type valueBlob struct {
	id  string
	val []byte
}

func createVBlobId(key []byte, ts uint64) string {
	return fmt.Sprintf("%v-%v", string(key), ts)
}

type valueBlobPtr struct {
	ts     uint64
	valLen uint64
}

// Decode decodes the value pointer into the provided byte buffer.
func (p *valueBlobPtr) Decode(b []byte) {
	p.ts = binary.BigEndian.Uint64(b[:8])
	p.valLen = binary.BigEndian.Uint64(b[8:16])
}

// it is asumed here the entry key is not including the timestamp
// actually the size should be 8 bytes larger as later key will be suffixed with ts
func (e *Entry) maybeSeparateValue(threshold int64) (int64, *valueBlob) {
	k := int64(len(e.Key))
	v := int64(len(e.Value))
	if v < threshold {
		return k + v + 2, nil
	}
	vBlobPtr := make([]byte, 8+8)
	ts := rtts.GetTs()
	// encode the timestamp id and the size.
	binary.BigEndian.PutUint64(vBlobPtr, ts)
	binary.BigEndian.PutUint64(vBlobPtr[8:], uint64(v))
	vBlob := e.Value
	e.Value = vBlobPtr
	copy(e.Value, vBlobPtr) // for overlaying on top of vlog
	// set the value separation bit
	e.meta = e.meta | bitValuePointer
	return k + 16 + 2, &valueBlob{createVBlobId(e.Key, ts), vBlob}
}

// NewEntry creates a new entry with key and value passed in args. This newly created entry can be
// set in a transaction by calling txn.SetEntry(). All other properties of Entry can be set by
// calling WithMeta, WithDiscard, WithTTL methods on it.
// This function uses key and value reference, hence users must
// not modify key and value until the end of transaction.
func NewEntry(key, value []byte) *Entry {
	return &Entry{
		Key:   key,
		Value: value,
	}
}

// WithMeta adds meta data to Entry e. This byte is stored alongside the key
// and can be used as an aid to interpret the value or store other contextual
// bits corresponding to the key-value pair of entry.
func (e *Entry) WithMeta(meta byte) *Entry {
	e.UserMeta = meta
	return e
}

// WithDiscard adds a marker to Entry e. This means all the previous versions of the key (of the
// Entry) will be eligible for garbage collection.
// This method is only useful if you have set a higher limit for options.NumVersionsToKeep. The
// default setting is 1, in which case, this function doesn't add any more benefit. If however, you
// have a higher setting for NumVersionsToKeep (in Dgraph, we set it to infinity), you can use this
// method to indicate that all the older versions can be discarded and removed during compactions.
func (e *Entry) WithDiscard() *Entry {
	e.meta = bitDiscardEarlierVersions
	return e
}

// WithTTL adds time to live duration to Entry e. Entry stored with a TTL would automatically expire
// after the time has elapsed, and will be eligible for garbage collection.
func (e *Entry) WithTTL(dur time.Duration) *Entry {
	e.ExpiresAt = uint64(time.Now().Add(dur).Unix())
	return e
}
