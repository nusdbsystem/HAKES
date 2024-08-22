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
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"hash"
	"hash/crc32"
	"io"
	"strconv"

	kvio "hakes-store/hakes-store/io"

	"github.com/dgraph-io/badger/v3/y"
	"github.com/pkg/errors"
)

const (
	memFileExt string = ".mem"
	// used to be called vlogHeaderSize.
	// +----------------+------------------+
	// | keyID(8 bytes) |  baseIV(12 bytes)|
	// +----------------+------------------+
	walHeaderSize = 20
)

var errStop = errors.New("Stop iteration")
var errTruncate = errors.New("Do truncate")

func errFile(err error, path string, msg string) error {
	return fmt.Errorf("%s. Path=%s. Error=%v", msg, path, err)
}

func getWalName(fid int) string {
	return fmt.Sprintf("%05d%s", fid, memFileExt)
}

func IsKVBootstrap(logName string, curOffset uint32) bool {
	return logName == getWalName(1) && curOffset == walHeaderSize
}

// header is used in value log as a header before Entry.
type header struct {
	klen      uint32
	vlen      uint32
	expiresAt uint64
	meta      byte
	userMeta  byte
}

const (
	// Maximum possible size of the header. The maximum size of header struct will be 18 but the
	// maximum size of varint encoded header will be 21.
	maxHeaderSize = 21
)

// Encode encodes the header into []byte. The provided []byte should be atleast 5 bytes. The
// function will panic if out []byte isn't large enough to hold all the values.
// The encoded header looks like
// +------+----------+------------+--------------+-----------+
// | Meta | UserMeta | Key Length | Value Length | ExpiresAt |
// +------+----------+------------+--------------+-----------+
func (h header) Encode(out []byte) int {
	out[0], out[1] = h.meta, h.userMeta
	index := 2
	index += binary.PutUvarint(out[index:], uint64(h.klen))
	index += binary.PutUvarint(out[index:], uint64(h.vlen))
	index += binary.PutUvarint(out[index:], h.expiresAt)
	return index
}

// Decode decodes the given header from the provided byte slice.
// Returns the number of bytes read.
func (h *header) Decode(buf []byte) int {
	h.meta, h.userMeta = buf[0], buf[1]
	index := 2
	klen, count := binary.Uvarint(buf[index:])
	h.klen = uint32(klen)
	index += count
	vlen, count := binary.Uvarint(buf[index:])
	h.vlen = uint32(vlen)
	index += count
	h.expiresAt, count = binary.Uvarint(buf[index:])
	return index + count
}

// DecodeFrom reads the header from the hashReader.
// Returns the number of bytes read.
func (h *header) DecodeFrom(reader *hashReader) (int, error) {
	var err error
	h.meta, err = reader.ReadByte()
	if err != nil {
		return 0, err
	}
	h.userMeta, err = reader.ReadByte()
	if err != nil {
		return 0, err
	}
	klen, err := binary.ReadUvarint(reader)
	if err != nil {
		return 0, err
	}
	h.klen = uint32(klen)
	vlen, err := binary.ReadUvarint(reader)
	if err != nil {
		return 0, err
	}
	h.vlen = uint32(vlen)
	h.expiresAt, err = binary.ReadUvarint(reader)
	if err != nil {
		return 0, err
	}
	return reader.bytesRead, nil
}

type safeRead struct {
	k []byte
	v []byte

	recordOffset uint32
	lf           *logFile
}

// hashReader implements io.Reader, io.ByteReader interfaces. It also keeps track of the number
// bytes read. The hashReader writes to h (hash) what it reads from r.
type hashReader struct {
	r         io.Reader
	h         hash.Hash32
	bytesRead int // Number of bytes read.
}

func newHashReader(r io.Reader) *hashReader {
	hash := crc32.New(y.CastagnoliCrcTable)
	return &hashReader{
		r: r,
		h: hash,
	}
}

// Read reads len(p) bytes from the reader. Returns the number of bytes read, error on failure.
func (t *hashReader) Read(p []byte) (int, error) {
	n, err := t.r.Read(p)
	if err != nil {
		return n, err
	}
	t.bytesRead += n
	return t.h.Write(p[:n])
}

// ReadByte reads exactly one byte from the reader. Returns error on failure.
func (t *hashReader) ReadByte() (byte, error) {
	b := make([]byte, 1)
	_, err := t.Read(b)
	return b[0], err
}

// Sum32 returns the sum32 of the underlying hash.
func (t *hashReader) Sum32() uint32 {
	return t.h.Sum32()
}

// Entry reads an entry from the provided reader. It also validates the checksum for every entry
// read. Returns error on failure.
func (r *safeRead) Entry(reader io.Reader) (*Entry, error) {
	tee := newHashReader(reader)
	var h header
	hlen, err := h.DecodeFrom(tee)
	if err != nil {
		return nil, err
	}
	if h.klen > uint32(1<<16) { // Key length must be below uint16.
		return nil, errTruncate
	}
	kl := int(h.klen)
	if cap(r.k) < kl {
		r.k = make([]byte, 2*kl)
	}
	vl := int(h.vlen)
	if cap(r.v) < vl {
		r.v = make([]byte, 2*vl)
	}

	e := &Entry{}
	e.offset = r.recordOffset
	e.hlen = hlen
	buf := make([]byte, h.klen+h.vlen)
	if _, err := io.ReadFull(tee, buf[:]); err != nil {
		if err == io.EOF {
			err = errTruncate
		}
		return nil, err
	}
	e.Key = buf[:h.klen]
	e.Value = buf[h.klen:]
	var crcBuf [crc32.Size]byte
	if _, err := io.ReadFull(reader, crcBuf[:]); err != nil {
		if err == io.EOF {
			err = errTruncate
		}
		return nil, err
	}
	crc := y.BytesToU32(crcBuf[:])
	if crc != tee.Sum32() {
		return nil, errTruncate
	}
	e.meta = h.meta
	e.UserMeta = h.userMeta
	e.ExpiresAt = h.expiresAt
	return e, nil
}

type logFile struct {
	kvio.DLog
	path    string
	writeAt uint32
}

// encodeEntry will encode entry to the buf
// layout of entry
// +--------+-----+-------+-------+
// | header | key | value | crc32 |
// +--------+-----+-------+-------+
func (lf *logFile) encodeEntry(buf *bytes.Buffer, e *Entry, offset uint32) (int, error) {
	h := header{
		klen:      uint32(len(e.Key)),
		vlen:      uint32(len(e.Value)),
		expiresAt: e.ExpiresAt,
		meta:      e.meta,
		userMeta:  e.UserMeta,
	}

	hash := crc32.New(y.CastagnoliCrcTable)
	writer := io.MultiWriter(buf, hash)

	// encode header.
	var headerEnc [maxHeaderSize]byte
	sz := h.Encode(headerEnc[:])
	y.Check2(writer.Write(headerEnc[:sz]))
	y.Check2(writer.Write(e.Key))
	y.Check2(writer.Write(e.Value))
	// }
	// write crc32 hash.
	var crcBuf [crc32.Size]byte
	binary.BigEndian.PutUint32(crcBuf[:], hash.Sum32())
	y.Check2(buf.Write(crcBuf[:]))
	// return encoded length.
	return len(headerEnc[:sz]) + len(e.Key) + len(e.Value) + len(crcBuf), nil
}

func (lf *logFile) writeEntry(buf *bytes.Buffer, e *Entry) error {
	buf.Reset()
	plen, err := lf.encodeEntry(buf, e, lf.writeAt)
	if err != nil {
		return err
	}
	written, err := lf.Append(buf.Bytes())
	if err != nil {
		return y.Wrap(err, "write entry to logFile")
	}
	y.AssertTrue(plen == written)
	lf.writeAt += uint32(plen)
	return nil
}

type logEntry func(e Entry) error

// iterate iterates over log file. It doesn't not allocate new memory for every kv pair.
// Therefore, the kv pair is only valid for the duration of fn call.
func (lf *logFile) iterate(readOnly bool, offset uint32, fn logEntry) (uint32, error) {
	if offset == 0 {
		// If offset is set to zero, let's advance past the encryption key header.
		offset = walHeaderSize
	}

	// For now, read directly from file, because it allows
	reader := bufio.NewReader(lf.NewReader(int(offset)))
	read := &safeRead{
		k:            make([]byte, 10),
		v:            make([]byte, 10),
		recordOffset: offset,
		lf:           lf,
	}

	var lastCommit uint64
	var validEndOffset uint32 = offset

	var entries []*Entry

loop:
	for {
		e, err := read.Entry(reader)
		switch {
		// We have not reached the end of the file but the entry we read is
		// zero. This happens because we have truncated the file and
		// zero'ed it out.
		case err == io.EOF:
			break loop
		case err == io.ErrUnexpectedEOF || err == errTruncate:
			break loop
		case err != nil:
			return 0, err
		case e == nil:
			continue
		case e.isZero():
			break loop
		}

		read.recordOffset += uint32(int(e.hlen) + len(e.Key) + len(e.Value) + crc32.Size)

		switch {
		case e.meta&bitTxn > 0:
			txnTs := y.ParseTs(e.Key)
			if lastCommit == 0 {
				lastCommit = txnTs
			}
			if lastCommit != txnTs {
				break loop
			}
			entries = append(entries, e)

		case e.meta&bitFinTxn > 0:
			txnTs, err := strconv.ParseUint(string(e.Value), 10, 64)
			if err != nil || lastCommit != txnTs {
				break loop
			}
			// Got the end of txn. Now we can store them.
			lastCommit = 0
			validEndOffset = read.recordOffset

			for _, e := range entries {
				if err := fn(*e); err != nil {
					if err == errStop {
						break
					}
					return 0, errFile(err, lf.path, "Iteration function")
				}
			}
			entries = entries[:0]

		default:
			if lastCommit != 0 {
				// This is most likely an entry which was moved as part of GC.
				// We shouldn't get this entry in the middle of a transaction.
				break loop
			}
			validEndOffset = read.recordOffset

			if err := fn(*e); err != nil {
				if err == errStop {
					break
				}
				return 0, errFile(err, lf.path, "Iteration function")
			}
		}
	}
	return validEndOffset, nil
}

func (lf *logFile) openNew(path string, fsize int64, dlogHandler kvio.DLogHandler) error {

	// there is need to have both open supported
	var mf kvio.DLog
	var ferr error
	mf, ferr = dlogHandler.OpenNewLog(path, int(fsize))
	if ferr != nil {
		return y.Wrapf(ferr, "while opening new file: %s", path)
	}
	lf.DLog = mf
	if err := lf.bootstrap(); err != nil {
		dlogHandler.Drop(path)
		return err
	}
	// Preserved ferr so we can return if this was a new file.
	return nil
}

func (lf *logFile) reopen(path string, fsize int64, dlogHandler kvio.DLogHandler) error {
	var mf kvio.DLog
	var ferr error
	if mf, ferr = dlogHandler.OpenLog(path, int(fsize)); ferr != nil {
		return y.Wrapf(ferr, "while ropening file: %s", path)
	} else {
		lf.DLog = mf
		return nil
	}
}

// bootstrap will initialize the log file with key id and baseIV.
// The below figure shows the layout of log file.
// +----------------+------------------+------------------+
// | keyID(8 bytes) |  baseIV(12 bytes)|	 entry...     |
// +----------------+------------------+------------------+
func (lf *logFile) bootstrap() error {
	// We'll always preserve vlogHeaderSize for key id and baseIV.
	buf := make([]byte, walHeaderSize)

	if len, err := lf.Append(buf); err != nil {
		return err
	} else {
		y.AssertTrue(walHeaderSize == len)
	}
	return nil
}
