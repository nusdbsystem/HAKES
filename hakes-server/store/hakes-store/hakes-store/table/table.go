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

package table

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/golang/protobuf/proto"
	"github.com/golang/snappy"
	"github.com/pkg/errors"

	"github.com/dgraph-io/badger/v3/options"
	"github.com/dgraph-io/badger/v3/pb"
	"github.com/dgraph-io/badger/v3/y"
	"github.com/dgraph-io/ristretto"
	"github.com/dgraph-io/ristretto/z"

	io "hakes-store/hakes-store/io"
	fb "hakes-store/hakes-store/table/fb"
)

const fileSuffix = ".sst"
const intSize = int(unsafe.Sizeof(int(0)))

// Options contains configurable options for Table/Builder.
type Options struct {
	// Options for Opening/Building Table.

	CssCli io.CSSCli

	// Open tables in read only mode.
	MetricsEnabled bool

	// Maximum size of the table.
	TableSize     uint64
	tableCapacity uint64 // 0.9x TableSize.

	// ChkMode is the checksum verification mode for Table.
	ChkMode options.ChecksumVerificationMode

	// Options for Table builder.

	// BloomFalsePositive is the false positive probabiltiy of bloom filter.
	BloomFalsePositive float64

	// BlockSize is the size of each block inside SSTable in bytes.
	BlockSize int

	// Compression indicates the compression algorithm used for block compression.
	Compression options.CompressionType

	// Block cache is used to cache decompressed and decrypted blocks.
	BlockCache *ristretto.Cache
	IndexCache *ristretto.Cache

	AllocPool *z.AllocatorPool

	// ZSTDCompressionLevel is the ZSTD compression level used for compressing blocks.
	ZSTDCompressionLevel int
}

// TableInterface is useful for testing.
type TableInterface interface {
	Smallest() []byte
	Biggest() []byte
	DoesNotHave(hash uint32) bool
	MaxVersion() uint64
}

// Table represents a loaded table file with the info we have about it.
type Table struct {
	sync.Mutex
	io.CSSF

	tableSize int // Initialized in OpenTable, using fd.Stat().

	_index *fb.TableIndex // Nil if encryption is enabled. Use fetchIndex to access.
	_cheap *cheapIndex
	ref    int32 // For file garbage collection. Atomic.

	// The following are initialized once and const.
	smallest, biggest []byte // Smallest and largest keys (with timestamps).
	id                string // file id, part of filename

	CreatedAt      time.Time
	indexStart     int
	indexLen       int
	hasBloomFilter bool

	opt *Options

	// filters (used ot be decoded on demand)
	filters []y.Filter

	// whether this table is cached in SSTCache
	inSstCache     bool
	deleteCallBack func() error // a general callback to be invoked by DecrRef.
}

func (t *Table) SetUseSSTCache() {
	t.inSstCache = true
}

func (t *Table) ReplaceDeleteCallBack(cb func() error) {
	t.deleteCallBack = cb
}

// expose it
func (t *Table) InSstCache() bool {
	return t.inSstCache
}

// serialization method for debugging
func (t *Table) String() string {
	return fmt.Sprintf("Table ID: %v, index size: %d, bloom %v, minKey: %v, maxKey: %v, cached: %v", t.id, t.indexLen, t.hasBloomFilter, string(y.ParseKey(t.smallest)), string(y.ParseKey(t.biggest)), t.inSstCache)
}

type cheapIndex struct {
	MaxVersion        uint64
	UncompressedSize  uint32
	OnDiskSize        uint32
	BloomFilterLength int
	OffsetsLength     int
}

func (t *Table) cheapIndex() *cheapIndex {
	return t._cheap
}
func (t *Table) offsetsLength() int { return t.cheapIndex().OffsetsLength }

// MaxVersion returns the maximum version across all keys stored in this table.
func (t *Table) MaxVersion() uint64 { return t.cheapIndex().MaxVersion }

// BloomFilterSize returns the size of the bloom filter in bytes stored in memory.
func (t *Table) BloomFilterSize() int { return t.cheapIndex().BloomFilterLength }

// UncompressedSize is the size uncompressed data stored in this file.
func (t *Table) UncompressedSize() uint32 { return t.cheapIndex().UncompressedSize }

// OnDiskSize returns the total size of key-values stored in this table (including the
// disk space occupied on the value log).
func (t *Table) OnDiskSize() uint32 { return t.cheapIndex().OnDiskSize }

// CompressionType returns the compression algorithm used for block compression.
func (t *Table) CompressionType() options.CompressionType {
	return t.opt.Compression
}

// IncrRef increments the refcount (having to do with whether the file should be deleted)
func (t *Table) IncrRef() {
	atomic.AddInt32(&t.ref, 1)
}

// DecrRef decrements the refcount and possibly deletes the table
func (t *Table) DecrRef() error {
	newRef := atomic.AddInt32(&t.ref, -1)
	if newRef == 0 {
		// We can safely delete this file, because for all the current files, we always have
		// at least one reference pointing to them.

		// Delete all blocks from the cache.
		for i := 0; i < t.offsetsLength(); i++ {
			t.opt.BlockCache.Del(t.blockCacheKey(i))
		}

		if t.deleteCallBack != nil {
			if err := t.deleteCallBack(); err != nil {
				return err
			}
		}
	}
	return nil
}

// BlockEvictHandler is used to reuse the byte slice stored in the block on cache eviction.
func BlockEvictHandler(value interface{}) {
	if b, ok := value.(*block); ok {
		b.decrRef()
	}
}

type block struct {
	offset            int
	data              []byte
	checksum          []byte
	entriesIndexStart int      // start index of entryOffsets list
	entryOffsets      []uint32 // used to binary search an entry in the block.
	chkLen            int      // checksum length.
	freeMe            bool     // used to determine if the blocked should be reused.
	ref               int32
}

var NumBlocks int32

// incrRef increments the ref of a block and return a bool indicating if the
// increment was successful. A true value indicates that the block can be used.
func (b *block) incrRef() bool {
	for {
		// We can't blindly add 1 to ref. We need to check whether it has
		// reached zero first, because if it did, then we should absolutely not
		// use this block.
		ref := atomic.LoadInt32(&b.ref)
		// The ref would not be equal to 0 unless the existing
		// block get evicted before this line. If the ref is zero, it means that
		// the block is already added the the blockPool and cannot be used
		// anymore. The ref of a new block is 1 so the following condition will
		// be true only if the block got reused before we could increment its
		// ref.
		if ref == 0 {
			return false
		}
		// Increment the ref only if it is not zero and has not changed between
		// the time we read it and we're updating it.
		//
		if atomic.CompareAndSwapInt32(&b.ref, ref, ref+1) {
			return true
		}
	}
}
func (b *block) decrRef() {
	if b == nil {
		return
	}

	// Insert the []byte into pool only if the block is resuable. When a block
	// is reusable a new []byte is used for decompression and this []byte can
	// be reused.
	// In case of an uncompressed block, the []byte is a reference to the
	// table.mmap []byte slice. Any attempt to write data to the mmap []byte
	// will lead to SEGFAULT.
	if atomic.AddInt32(&b.ref, -1) == 0 {
		if b.freeMe {
			z.Free(b.data)
		}
		atomic.AddInt32(&NumBlocks, -1)
	}
	y.AssertTrue(atomic.LoadInt32(&b.ref) >= 0)
}
func (b *block) size() int64 {
	return int64(3*intSize /* Size of the offset, entriesIndexStart and chkLen */ +
		cap(b.data) + cap(b.checksum) + cap(b.entryOffsets)*4)
}

func (b block) verifyCheckSum() error {
	cs := &pb.Checksum{}
	if err := proto.Unmarshal(b.checksum, cs); err != nil {
		return y.Wrapf(err, "unable to unmarshal checksum for block")
	}
	return y.VerifyChecksum(b.data, cs)
}

func maybeWriteToSstCache(fname string, bd *buildData, sc io.SstCache) io.CSSF {
	if f := sc.Add(fname, bd.Size); f == nil {
		// fail to add to cache
		return nil
	} else {
		// write to cache
		bd.WriteToCSSF(f)
		if err := f.Sync(); err != nil {
			log.Printf("failed to sync %s to SSTCache, dropping it: %v", fname, err)
			sc.Drop(fname, bd.Size)
			return nil
		}
	}

	// reopen from the cache
	if f, err := sc.Get(fname); err != nil {
		log.Printf("failed to reopen %s from SSTCache, dropping it: %v", fname, err)
		sc.Drop(fname, bd.Size)
		return nil
	} else {
		return f
	}
}

// pass nil as sc, if the table is not intended to place in SstCache
func CreateTable(fname string, builder *Builder, cssCli io.CSSCli, sc io.SstCache) (*Table, error) {
	bd := builder.Done()
	// write to durable storage
	if mf, err := bd.WriteTo(fname, cssCli); err != nil {
		return nil, err
	} else {
		mf.Close(-1)
	}

	// write to cache if intended
	if sc != nil {
		if mf := maybeWriteToSstCache(fname, &bd, sc); mf != nil {
			if t, err := OpenTableFromBuilder(bd, mf, *builder.opts, true); err == nil {
				// register deletion callback to delete the file from slow storage
				t.ReplaceDeleteCallBack(func() error {
					log.Printf("calling deletion call back on %v (release: %d)", fname, bd.Size)
					sc.Drop(fname, bd.Size)
					return cssCli.Delete(fname)
				})
				return t, nil
			} else {
				return nil, err
			}
		}
	}

	// reopen the file from durable storage
	if mf, err := cssCli.OpenFile(fname, bd.Size); err != nil {
		return nil, err
	} else {
		return OpenTableFromBuilder(bd, mf, *builder.opts, false)
	}
}

// avoid disk io entirely by getting the info from buildData
func OpenTableFromBuilder(bd buildData, mf io.CSSF, opts Options, inSstCache bool) (*Table, error) {
	if opts.BlockSize == 0 && opts.Compression != options.None {
		return nil, errors.New("Block size cannot be zero")
	}
	filename := mf.Name()
	id, ok := ParseFileID(filename)
	if !ok {
		mf.Close(-1)
		return nil, errors.Errorf("Invalid filename: %s", filename)
	}
	t := &Table{
		CSSF:           mf,
		ref:            1, // Caller is given one reference.
		id:             id,
		opt:            &opts,
		tableSize:      bd.Size,
		CreatedAt:      time.Now(), // just created
		inSstCache:     inSstCache,
		deleteCallBack: mf.Delete,
	}
	t.indexLen = len(bd.index)
	t.indexStart = bd.Size - 4 - 4 - len(bd.checksum) - t.indexLen

	// get index
	indexData := y.Copy(bd.index)
	index := fb.GetRootAsTableIndex(indexData, 0)

	t._index = index

	filterLen := 0
	filterCount := index.BloomFiltersLength()
	for i := 0; i < filterCount; i++ {
		var f fb.Filter
		y.AssertTrue(index.BloomFilters(&f, i))
		fBytes := f.DataBytes()
		filterLen += len(fBytes)
		t.filters = append(t.filters, y.Filter(fBytes))
	}
	t.hasBloomFilter = filterLen > 0

	t._cheap = &cheapIndex{
		MaxVersion:        index.MaxVersion(),
		UncompressedSize:  index.UncompressedSize(),
		OnDiskSize:        index.OnDiskSize(),
		OffsetsLength:     index.OffsetsLength(),
		BloomFilterLength: filterLen,
	}
	t.smallest = y.Copy(bd.blockList[0].baseKey)
	t.biggest = y.Copy(bd.blockList[len(bd.blockList)-1].maxKey)
	y.AssertTrue(y.CompareKeys(t.biggest, t.smallest) >= 0)
	return t, nil
}

// OpenTable assumes file has only one table and opens it. Takes ownership of fd upon function
// entry. Returns a table with one reference count on it (decrementing which may delete the file!
// -- consider t.Close() instead). The fd has to writeable because we call Truncate on it before
// deleting. Checksum for all blocks of table is verified based on value of chkMode.
// func OpenTable(mf *z.MmapFile, opts Options) (*Table, error) {
func OpenTable(mf io.CSSF, opts Options) (*Table, error) {
	// BlockSize is used to compute the approximate size of the decompressed
	// block. It should not be zero if the table is compressed.
	if opts.BlockSize == 0 && opts.Compression != options.None {
		return nil, errors.New("Block size cannot be zero")
	}
	filename := mf.Name()
	id, ok := ParseFileID(filename)
	if !ok {
		mf.Close(-1)
		return nil, errors.Errorf("Invalid filename: %s", filename)
	}
	t := &Table{
		CSSF:      mf,
		ref:       1, // Caller is given one reference.
		id:        id,
		opt:       &opts,
		tableSize: int(mf.Size()),
		CreatedAt: mf.ModTime(),
	}
	// default deletion callback
	t.deleteCallBack = func() error {
		return t.Delete()
	}

	if err := t.initBiggestAndSmallest(); err != nil {
		return nil, y.Wrapf(err, "failed to initialize table")
	}

	if opts.ChkMode == options.OnTableRead || opts.ChkMode == options.OnTableAndBlockRead {
		if err := t.VerifyChecksum(); err != nil {
			mf.Close(-1)
			return nil, y.Wrapf(err, "failed to verify checksum")
		}
	}

	return t, nil
}

func (t *Table) CloseTable() error {
	return t.Close(-1)
}

func (t *Table) initBiggestAndSmallest() error {
	if err := t.initIndex(); err != nil {
		return y.Wrapf(err, "failed to read index.")
	}

	var ko fb.BlockOffset
	y.AssertTrue(t.offsets(&ko, 0))

	t.smallest = y.Copy(ko.KeyBytes())

	y.AssertTrue(t.offsets(&ko, t._cheap.OffsetsLength-1))
	t.biggest = y.Copy(ko.MaxKeyBytes())
	y.AssertTrue(y.CompareKeys(t.biggest, t.smallest) >= 0)
	return nil
}

// this is where we will differentiate the block loading logic through a table option param.
func (t *Table) loadBlock(bo *fb.BlockOffset, pb PrefetchBuffer) ([]byte, error) {
	// prefetch buffer first
	if pb != nil {
		if data, err := pb.Bytes(int(bo.Offset()), int(bo.Len())); err != nil {
			return nil, err
		} else if len(data) > 0 {
			// data read from prefetch buffer
			return data, nil
		}
		// fall through: prefetch not used.
	}

	return t.read(int(bo.Offset()), int(bo.Len()))
}

func (t *Table) read(off, sz int) ([]byte, error) {
	return t.Bytes(off, sz)
}

func (t *Table) readNoFail(off, sz int) []byte {
	res, err := t.read(off, sz)
	y.Check(err)
	return res
}

// initIndex reads the index and populate the necessary table fields and returns
// first block offset
func (t *Table) initIndex() error {
	readPos := t.tableSize

	// prefetch the tail (0.04 should be sufficient for NoCompression case)
	// In RocksDB, prefetch is adjusted based on previous read value. We may have that strategy in the future
	var prefetchSz int

	if t.opt.Compression == options.None {
		prefetchSz = int(float32(t.tableSize) * 0.04)
	} else {
		prefetchSz = int(float32(t.tableSize) * 0.4)
	}
	prefetchOff := t.tableSize - prefetchSz
	// at least prefetch one page
	if prefetchSz < (64 << 10) {
		if t.tableSize < (64 << 10) {
			prefetchOff = 0
			prefetchSz = t.tableSize
		} else {
			prefetchSz = (64 << 10)
			prefetchOff = t.tableSize - prefetchSz
		}
	}
	prefetchTail := t.readNoFail(prefetchOff, prefetchSz)

	// Read checksum len from the last 4 bytes.
	readPos -= 4
	var buf []byte
	// buf := t.readNoFail(readPos, 4)
	if readPos >= prefetchOff {
		off := readPos - prefetchOff
		buf = prefetchTail[off : off+4]
	} else {
		buf = t.readNoFail(readPos, 4)
	}
	checksumLen := int(y.BytesToU32(buf))
	if checksumLen < 0 {
		return errors.New("checksum length less than zero. Data corrupted")
	}

	// Read checksum.
	expectedChk := &pb.Checksum{}
	readPos -= checksumLen
	if readPos >= prefetchOff {
		off := readPos - prefetchOff
		buf = prefetchTail[off : off+checksumLen]
	} else {
		buf = t.readNoFail(readPos, checksumLen)
	}
	if err := proto.Unmarshal(buf, expectedChk); err != nil {
		return err
	}

	// Read index size from the footer.
	readPos -= 4
	if readPos >= prefetchOff {
		off := readPos - prefetchOff
		buf = prefetchTail[off : off+4]
	} else {
		buf = t.readNoFail(readPos, 4)
	}
	t.indexLen = int(y.BytesToU32(buf))

	// Read index.
	readPos -= t.indexLen
	t.indexStart = readPos
	var data []byte
	if readPos >= prefetchOff {
		off := readPos - prefetchOff
		data = prefetchTail[off : off+t.indexLen]
	} else {
		log.Printf("prefetched tail failed to cover index (prefetchOff: %d, indexStart: %d)", prefetchOff, readPos)
		data = t.readNoFail(readPos, t.indexLen)
	}

	if err := y.VerifyChecksum(data, expectedChk); err != nil {
		return y.Wrapf(err, "failed to verify checksum for table: %s", t.Filename())
	}

	// create a copy of data to let table hold it.
	index := fb.GetRootAsTableIndex(y.Copy(data), 0)

	t._index = index

	filterLen := 0
	// initialize filter struct
	filterCount := index.BloomFiltersLength()
	for i := 0; i < filterCount; i++ {
		var f fb.Filter
		y.AssertTrue(index.BloomFilters(&f, i))
		fBytes := f.DataBytes()
		filterLen += len(fBytes)
		t.filters = append(t.filters, y.Filter(fBytes))
	}

	t.hasBloomFilter = filterLen > 0

	t._cheap = &cheapIndex{
		MaxVersion:        index.MaxVersion(),
		UncompressedSize:  index.UncompressedSize(),
		OnDiskSize:        index.OnDiskSize(),
		OffsetsLength:     index.OffsetsLength(),
		BloomFilterLength: filterLen,
	}
	return nil
}

func (t *Table) fetchIndex() *fb.TableIndex {
	return t._index
}

func (t *Table) offsets(ko *fb.BlockOffset, i int) bool {
	return t.fetchIndex().Offsets(ko, i)
}

// block function return a new block. Each block holds a ref and the byte
// slice stored in the block will be reused when the ref becomes zero. The
// caller should release the block by calling block.decrRef() on it.
//
// we add an argument to indicate whether we only try loading from cache.
func (t *Table) block(idx int, useCache bool, pb PrefetchBuffer) (*block, error) {
	y.AssertTruef(idx >= 0, "idx=%d", idx)
	if idx >= t.offsetsLength() {
		return nil, errors.New("block out of index")
	}
	if t.opt.BlockCache != nil && !t.inSstCache {
		key := t.blockCacheKey(idx)
		blk, ok := t.opt.BlockCache.Get(key)
		if ok && blk != nil {
			// Use the block only if the increment was successful. The block
			// could get evicted from the cache between the Get() call and the
			// incrRef() call.
			if b := blk.(*block); b.incrRef() {
				if pb != nil {
					// update the pb usage pattern even though served by block cache
					pb.UpdateAccess(b.offset, int(b.size()))
				}
				return b, nil
			}
		}
	}

	var ko fb.BlockOffset
	y.AssertTrue(t.offsets(&ko, idx))
	blk := &block{
		offset: int(ko.Offset()),
		ref:    1,
	}
	defer blk.decrRef() // Deal with any errors, where blk would not be returned.
	atomic.AddInt32(&NumBlocks, 1)

	var err error
	if data, err := t.loadBlock(&ko, pb); err != nil {
		return nil, y.Wrapf(err,
			"failed to read from file: %s at offset: %d, len: %d",
			t.Name(), blk.offset, ko.Len())
	} else {
		// create a local copy. It may be cached.
		blk.data = y.Copy(data)
	}

	if err = t.decompress(blk); err != nil {
		return nil, y.Wrapf(err,
			"failed to decode compressed data in file: %s at offset: %d, len: %d",
			t.Name(), blk.offset, ko.Len())
	}

	// Read meta data related to block.
	readPos := len(blk.data) - 4 // First read checksum length.
	blk.chkLen = int(y.BytesToU32(blk.data[readPos : readPos+4]))

	// Checksum length greater than block size could happen if the table was compressed and
	// it was opened with an incorrect compression algorithm (or the data was corrupted).
	if blk.chkLen > len(blk.data) {
		return nil, errors.New("invalid checksum length. Either the data is " +
			"corrupted or the table options are incorrectly set")
	}

	// Read checksum and store it
	readPos -= blk.chkLen
	blk.checksum = blk.data[readPos : readPos+blk.chkLen]
	// Move back and read numEntries in the block.
	readPos -= 4
	numEntries := int(y.BytesToU32(blk.data[readPos : readPos+4]))
	entriesIndexStart := readPos - (numEntries * 4)
	entriesIndexEnd := entriesIndexStart + numEntries*4

	blk.entryOffsets = y.BytesToU32Slice(blk.data[entriesIndexStart:entriesIndexEnd])

	blk.entriesIndexStart = entriesIndexStart

	// Drop checksum and checksum length.
	// The checksum is calculated for actual data + entry index + index length
	blk.data = blk.data[:readPos+4]

	// Verify checksum on if checksum verification mode is OnRead on OnStartAndRead.
	if t.opt.ChkMode == options.OnBlockRead || t.opt.ChkMode == options.OnTableAndBlockRead {
		if err = blk.verifyCheckSum(); err != nil {
			return nil, err
		}
	}

	blk.incrRef()
	if useCache && t.opt.BlockCache != nil && !t.inSstCache {
		key := t.blockCacheKey(idx)
		// incrRef should never return false here because we're calling it on a
		// new block with ref=1.
		y.AssertTrue(blk.incrRef())

		// Decrement the block ref if we could not insert it in the cache.
		if !t.opt.BlockCache.Set(key, blk, blk.size()) {
			blk.decrRef()
		}
		// We have added an OnReject func in our cache, which gets called in case the block is not
		// admitted to the cache. So, every block would be accounted for.
	}
	return blk, nil
}

func (t *Table) blockCacheKey(idx int) []byte {
	y.AssertTrue(uint32(idx) < math.MaxUint32)

	buf := make([]byte, len(t.id)+4)
	copy(buf, []byte(t.id))
	binary.BigEndian.PutUint32(buf[len(buf)-4:], uint32(idx))
	return buf
}

// IndexSize is the size of table index in bytes.
func (t *Table) IndexSize() int {
	return t.indexLen
}

// Size is its file size in bytes
func (t *Table) Size() int64 { return int64(t.tableSize) }

// StaleDataSize is the amount of stale data (that can be dropped by a compaction )in this SST.
func (t *Table) StaleDataSize() uint32 { return t.fetchIndex().StaleDataSize() }

// Smallest is its smallest key, or nil if there are none
func (t *Table) Smallest() []byte { return t.smallest }

// Biggest is its biggest key, or nil if there are none
func (t *Table) Biggest() []byte { return t.biggest }

// Filename is NOT the file name.  Just kidding, it is.
func (t *Table) Filename() string { return t.Name() }

// ID is the table's ID number (used to make the file name).
func (t *Table) ID() string { return t.id }

// DoesNotHave returns true if and only if the table does not have the key hash.
// It does a bloom filter lookup.
func (t *Table) DoesNotHave(hash uint32) bool {
	if !t.hasBloomFilter {
		return false
	}

	y.NumLSMBloomHitsAdd(t.opt.MetricsEnabled, "DoesNotHave_ALL", 1)
	mayContain := t.filters[0].MayContain(hash)
	if !mayContain {
		y.NumLSMBloomHitsAdd(t.opt.MetricsEnabled, "DoesNotHave_HIT", 1)
	}
	return !mayContain
}

// VerifyChecksum verifies checksum for all blocks of table. This function is called by
// OpenTable() function. This function is also called inside levelsController.VerifyChecksum().
func (t *Table) VerifyChecksum() error {
	ti := t.fetchIndex()
	for i := 0; i < ti.OffsetsLength(); i++ {
		b, err := t.block(i, true, nil)
		if err != nil {
			return y.Wrapf(err, "checksum validation failed for table: %s, block: %d, offset:%d",
				t.Filename(), i, b.offset)
		}
		// We should not call incrRef here, because the block already has one ref when created.
		defer b.decrRef()
		// OnBlockRead or OnTableAndBlockRead, we don't need to call verify checksum
		// on block, verification would be done while reading block itself.
		if !(t.opt.ChkMode == options.OnBlockRead || t.opt.ChkMode == options.OnTableAndBlockRead) {
			if err = b.verifyCheckSum(); err != nil {
				return y.Wrapf(err,
					"checksum validation failed for table: %s, block: %d, offset:%d",
					t.Filename(), i, b.offset)
			}
		}
	}
	return nil
}

// KeyID returns data key id.
func (t *Table) KeyID() uint64 {
	return 0
}

func ParseFileID(name string) (string, bool) {
	name = filepath.Base(name)
	if !strings.HasSuffix(name, fileSuffix) {
		return "", false
	}
	return strings.TrimSuffix(name, fileSuffix), true
}

// IDToFilename does the inverse of ParseFileID
func IDToFilename(id uint64) string {
	return fmt.Sprintf("%06d", id) + fileSuffix
}

// NewFilename should be named TableFilepath -- it combines the dir with the ID to make a table
// filepath.
func NewFilename(id uint64) string {
	return filepath.Join(IDToFilename(id))
}

func NewFilenameFromStrID(id string) string {
	return filepath.Join(id + fileSuffix)
}

// decompress decompresses the data stored in a block.
func (t *Table) decompress(b *block) error {
	var dst []byte
	var err error

	// Point to the original b.data
	src := b.data

	switch t.opt.Compression {
	case options.None:
		// Nothing to be done here.
		return nil
	case options.Snappy:
		if sz, err := snappy.DecodedLen(b.data); err == nil {
			dst = z.Calloc(sz, "Table.Decompress")
		} else {
			dst = z.Calloc(len(b.data)*4, "Table.Decompress") // Take a guess.
		}
		b.data, err = snappy.Decode(dst, b.data)
		if err != nil {
			z.Free(dst)
			return y.Wrap(err, "failed to decompress")
		}
	case options.ZSTD:
		sz := int(float64(t.opt.BlockSize) * 1.2)
		dst = z.Calloc(sz, "Table.Decompress")
		b.data, err = y.ZSTDDecompress(dst, b.data)
		if err != nil {
			z.Free(dst)
			return y.Wrap(err, "failed to decompress")
		}
	default:
		return errors.New("Unsupported compression type")
	}

	if b.freeMe {
		z.Free(src)
		b.freeMe = false
	}

	if len(b.data) > 0 && len(dst) > 0 && &dst[0] != &b.data[0] {
		z.Free(dst)
	} else {
		b.freeMe = true
	}
	return nil
}
