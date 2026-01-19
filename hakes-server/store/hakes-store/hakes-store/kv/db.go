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
	"expvar"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"hakes-store/hakes-store/table"

	"github.com/dgraph-io/badger/v3/options"
	"github.com/dgraph-io/badger/v3/skl"
	"github.com/dgraph-io/badger/v3/y"
	"github.com/dgraph-io/ristretto"
	"github.com/dgraph-io/ristretto/z"
	humanize "github.com/dustin/go-humanize"
	"github.com/pkg/errors"
)

var (
	hakesKVPrefix = []byte("!hakeskv!")
)

type closers struct {
	updateSize  *z.Closer
	compactors  *z.Closer
	memtable    *z.Closer
	writes      *z.Closer
	cacheHealth *z.Closer
	ioMonitor   *z.Closer
}

// DB provides the various functions required to interact with Badger.
// DB is thread-safe.
type DB struct {
	lock sync.RWMutex // Guards list of inmemory tables, not individual reads and writes.

	closers closers

	mt  *memTable   // Our latest (actively written) in-memory table
	imm []*memTable // Add here only AFTER pushing to flushChan.

	// Initialized via openMemTables.
	nextMemFid int

	opt       Options
	manifest  *manifestFile
	lc        *levelsController
	writeCh   chan *request
	flushChan chan flushTask // For flushing memtables.
	closeOnce sync.Once      // For closing DB only once.

	blockWrites int32
	isClosed    uint32

	orc *oracle

	blockCache *ristretto.Cache
	indexCache *ristretto.Cache
	allocPool  *z.AllocatorPool

	// id allocator
	idAlloc IdAllocator
}

const (
	kvWriteChCapacity = 5000
)

func checkAndSetOptions(opt *Options) error {
	// It's okay to have zero compactors which will disable all compactions but
	// we cannot have just one compactor otherwise we will end up with all data
	// on level 2.
	if opt.NumCompactors == 1 {
		return errors.New("Cannot have 1 compactor. Need at least 2")
	}

	opt.maxBatchSize = (15 * opt.MemTableSize) / 100
	opt.maxBatchCount = opt.maxBatchSize / int64(skl.MaxNodeSize)
	if opt.ValueThreshold > opt.maxBatchSize {
		return errors.Errorf("Valuethreshold %d greater than max batch size of %d. Either "+
			"reduce opt.ValueThreshold or increase opt.MaxTableSize.",
			opt.ValueThreshold, opt.maxBatchSize)
	}

	// set value threshold at least 1KB
	if opt.ValueThreshold < (1 << 10) {
		opt.ValueThreshold = 1 << 10
	}

	// ValueLogFileSize should be stricly LESS than 2<<30 otherwise we will
	// overflow the uint32 when we mmap it in OpenMemtable.
	if !(opt.ValueLogFileSize < 2<<30 && opt.ValueLogFileSize >= 1<<20) {
		return ErrValueLogSize
	}

	needCache := (opt.Compression != options.None) || (len(opt.EncryptionKey) > 0)
	if needCache && opt.BlockCacheSize == 0 {
		panic("BlockCacheSize should be set since compression/encryption are enabled")
	}

	// check the csscli, dloghandler, manifesthandler
	if !(opt.cssSetup && opt.dlsSetup && opt.manifestHandlerSetup) {
		return errors.Errorf("external io service are not setup")
	}

	return nil
}

// Open returns a new DB object.
func Open(opt Options) (*DB, error) {
	if err := checkAndSetOptions(&opt); err != nil {
		return nil, err
	}

	manifestFile, manifest, err := openOrCreateManifestFile(opt)
	if err != nil {
		return nil, err
	}
	defer func() {
		if manifestFile != nil {
			_ = manifestFile.close()
		}
	}()

	db := &DB{
		imm:       make([]*memTable, 0, opt.NumMemtables),
		flushChan: make(chan flushTask, opt.NumMemtables),
		writeCh:   make(chan *request, kvWriteChCapacity),
		opt:       opt,
		manifest:  manifestFile,
		orc:       newOracle(opt),
		allocPool: z.NewAllocatorPool(8),
		idAlloc:   NewEpochPrefixIdAllocatorWithEpoch(manifest.GetEpoch()),
	}
	// Cleanup all the goroutines started by badger in case of an error.
	defer func() {
		if err != nil {
			opt.Errorf("Received err: %v. Cleaning up...", err)
			db.cleanup()
			db = nil
		}
	}()

	if opt.BlockCacheSize > 0 {
		numInCache := opt.BlockCacheSize / int64(opt.BlockSize)
		if numInCache == 0 {
			// Make the value of this variable at least one since the cache requires
			// the number of counters to be greater than zero.
			numInCache = 1
		}

		config := ristretto.Config{
			NumCounters: numInCache * 8,
			MaxCost:     opt.BlockCacheSize,
			BufferItems: 64,
			Metrics:     true,
			OnExit:      table.BlockEvictHandler,
		}
		db.opt.Logger.Infof("Block cache size:  %d MB used", opt.BlockCacheSize)
		db.blockCache, err = ristretto.NewCache(&config)
		if err != nil {
			return nil, y.Wrap(err, "failed to create data cache")
		}
	}

	if opt.IndexCacheSize > 0 {
		// Index size is around 5% of the table size.
		indexSz := int64(float64(opt.MemTableSize) * 0.05)
		numInCache := opt.IndexCacheSize / indexSz
		if numInCache == 0 {
			// Make the value of this variable at least one since the cache requires
			// the number of counters to be greater than zero.
			numInCache = 1
		}

		config := ristretto.Config{
			NumCounters: numInCache * 8,
			MaxCost:     opt.IndexCacheSize,
			BufferItems: 64,
			Metrics:     true,
		}
		db.indexCache, err = ristretto.NewCache(&config)
		if err != nil {
			return nil, y.Wrap(err, "failed to create bf cache")
		}
	}

	db.closers.cacheHealth = z.NewCloser(1)
	go db.monitorCache(db.closers.cacheHealth)

	db.calculateSize()
	db.closers.updateSize = z.NewCloser(1)
	go db.updateSize(db.closers.updateSize)

	if err := db.openMemTables(db.opt, manifest.minWal); err != nil {
		return nil, y.Wrapf(err, "while opening memtables")
	}

	if db.mt, err = db.newMemTable(); err != nil {
		return nil, y.Wrapf(err, "cannot create memtable")
	}

	// newLevelsController potentially loads files in directory.
	if db.lc, err = newLevelsController(db, &manifest); err != nil {
		return db, err
	}

	db.closers.compactors = z.NewCloser(1)
	db.lc.startCompact(db.closers.compactors)

	db.closers.memtable = z.NewCloser(1)
	go func() {
		_ = db.flushMemtable(db.closers.memtable) // Need levels controller to be up.
	}()
	// Flush them to disk asap.
	for _, mt := range db.imm {
		db.flushChan <- flushTask{mt: mt}
	}

	// In normal mode, we must update readMark so older versions of keys can be removed during
	// compaction when run in offline mode via the flatten tool.
	db.orc.readMark.Done(db.MaxVersion())
	db.orc.incrementNextTs()

	db.closers.writes = z.NewCloser(1)
	go db.doWrites(db.closers.writes)

	// launch io stats logging
	db.closers.ioMonitor = z.NewCloser(1)
	go db.monitorIO(db.closers.ioMonitor)

	manifestFile = nil
	return db, nil
}

func (db *DB) monitorIO(lc *z.Closer) {
	defer lc.Done()

	statsTicker := time.NewTicker(time.Second)
	defer statsTicker.Stop()
	db.opt.Logger.Infof("io monitor started")

	var lastCssStats, lastDlhStats, lastMhStats string

	for {
		select {
		case <-statsTicker.C:
			if cssStats := db.opt.cssCli.GetStats(); len(cssStats) > 0 && cssStats != lastCssStats {
				lastCssStats = cssStats
				db.opt.Logger.Infof("CSS stats: %s", cssStats)
			}
			if dlhStats := db.opt.dlogHandler.GetStats(); len(dlhStats) > 0 && dlhStats != lastDlhStats {
				lastDlhStats = dlhStats
				db.opt.Logger.Infof("DLH stats: %s", dlhStats)
			}
			if mhStats := db.opt.manifestHandler.GetStats(); len(mhStats) > 0 && mhStats != lastMhStats {
				lastMhStats = mhStats
				db.opt.Logger.Infof("mh stats: %s", mhStats)
			}
		case <-lc.HasBeenClosed():
			db.opt.Logger.Infof("io monitor stopped")
			return
		}
	}
}

func (db *DB) MaxVersion() uint64 {
	var maxVersion uint64
	update := func(a uint64) {
		if a > maxVersion {
			maxVersion = a
		}
	}
	db.lock.Lock()
	// In read only mode, we do not create new mem table.
	update(db.mt.maxVersion)
	for _, mt := range db.imm {
		update(mt.maxVersion)
	}
	db.lock.Unlock()
	for _, ti := range db.Tables() {
		update(ti.MaxVersion)
	}
	return maxVersion
}

func (db *DB) monitorCache(c *z.Closer) {
	defer c.Done()
	count := 0
	analyze := func(name string, metrics *ristretto.Metrics) {
		// If the mean life expectancy is less than 10 seconds, the cache
		// might be too small.
		le := metrics.LifeExpectancySeconds()
		if le == nil {
			return
		}
		lifeTooShort := le.Count > 0 && float64(le.Sum)/float64(le.Count) < 10
		hitRatioTooLow := metrics.Ratio() > 0 && metrics.Ratio() < 0.4
		if lifeTooShort && hitRatioTooLow {
			db.opt.Warningf("%s might be too small. Metrics: %s\n", name, metrics)
			db.opt.Warningf("Cache life expectancy (in seconds): %+v\n", le)
		}
		// always log cache metric
		db.opt.Infof("%s metrics: %s\n", name, metrics)
	}

	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	for {
		select {
		case <-c.HasBeenClosed():
			return
		case <-ticker.C:
		}

		analyze("Block cache", db.BlockCacheMetrics())
		analyze("Index cache", db.IndexCacheMetrics())
		count++
	}
}

// cleanup stops all the goroutines started by badger. This is used in open to
// cleanup goroutines in case of an error.
func (db *DB) cleanup() {
	db.stopMemoryFlush()
	db.stopCompactions()

	db.blockCache.Close()
	db.indexCache.Close()
	if db.closers.updateSize != nil {
		db.closers.updateSize.Signal()
	}
	if db.closers.writes != nil {
		db.closers.writes.Signal()
	}

	db.orc.Stop()
}

// BlockCacheMetrics returns the metrics for the underlying block cache.
func (db *DB) BlockCacheMetrics() *ristretto.Metrics {
	if db.blockCache != nil {
		return db.blockCache.Metrics
	}
	return nil
}

// IndexCacheMetrics returns the metrics for the underlying index cache.
func (db *DB) IndexCacheMetrics() *ristretto.Metrics {
	if db.indexCache != nil {
		return db.indexCache.Metrics
	}
	return nil
}

// Close closes a DB. It's crucial to call it to ensure all the pending updates make their way to
// disk. Calling DB.Close() multiple times would still only close the DB once.
func (db *DB) Close() error {
	var err error
	db.closeOnce.Do(func() {
		err = db.close()
	})
	return err
}

// IsClosed denotes if the badger DB is closed or not. A DB instance should not
// be used after closing it.
func (db *DB) IsClosed() bool {
	return atomic.LoadUint32(&db.isClosed) == 1
}

func (db *DB) balanceLSM() {
	for {
		if db.lc.levels[0].numTables() == 0 {
			break
		}
		err := db.lc.doCompact(173, compactionPriority{level: 0, score: 1.73})
		switch err {
		case errFillTables:
			// This error only means that there might be enough tables to do a compaction. So, we
			// should not report it to the end user to avoid confusing them.
		case nil:
			db.opt.Infof("Force compaction on level 0 done")
		default:
			db.opt.Warningf("While forcing compaction on level 0: %v", err)
		}
	}
	db.opt.Infof("L0 cleared\n")

	for {
		time.Sleep(10 * time.Millisecond)
		prios := db.lc.pickCompactLevels()
		// no prios will return if all levels have score below 1.0
		if len(prios) == 0 {
			break
		}
	}
	db.opt.Infof("LSM tree is balanced\n")
}

func (db *DB) close() (err error) {
	defer db.allocPool.Release()

	db.opt.Debugf("Closing database")
	db.opt.Infof("Lifetime L0 stalled for: %s\n", time.Duration(atomic.LoadInt64(&db.lc.l0stallsMs)))

	atomic.StoreInt32(&db.blockWrites, 1)
	// Stop writes next.
	db.closers.writes.SignalAndWait()

	// Don't accept any more write.
	close(db.writeCh)

	db.closers.cacheHealth.Signal()

	// Make sure that block writer is done pushing stuff into memtable!
	// Otherwise, you will have a race condition: we are trying to flush memtables
	// and remove them completely, while the block / memtable writer is still
	// trying to push stuff into the memtable. This will also resolve the value
	// offset problem: as we push into memtable, we update value offsets there.
	if db.mt != nil {
		if db.mt.sl.Empty() {
			// Remove the memtable if empty.
			db.mt.DecrRef()
		} else {
			db.opt.Debugf("Flushing memtable")
			for {
				pushedFlushTask := func() bool {
					db.lock.Lock()
					defer db.lock.Unlock()
					y.AssertTrue(db.mt != nil)
					select {
					case db.flushChan <- flushTask{mt: db.mt}:
						db.imm = append(db.imm, db.mt) // Flusher will attempt to remove this from s.imm.
						db.mt = nil                    // Will segfault if we try writing!
						db.opt.Debugf("pushed to flush chan\n")
						return true
					default:
						// If we fail to push, we need to unlock and wait for a short while.
						// The flushing operation needs to update s.imm. Otherwise, we have a
						// deadlock.
						// TODO: Think about how to do this more cleanly, maybe without any locks.
					}
					return false
				}()
				if pushedFlushTask {
					break
				}
				time.Sleep(10 * time.Millisecond)
			}
		}
	}
	db.stopMemoryFlush()
	db.balanceLSM()
	db.stopCompactions()

	// Force Compact L0
	// We don't need to care about cstatus since no parallel compaction is running.
	if db.opt.CompactL0OnClose {
		err := db.lc.doCompact(173, compactionPriority{level: 0, score: 1.73})
		switch err {
		case errFillTables:
			// This error only means that there might be enough tables to do a compaction. So, we
			// should not report it to the end user to avoid confusing them.
		case nil:
			db.opt.Debugf("Force compaction on level 0 done")
		default:
			db.opt.Warningf("While forcing compaction on level 0: %v", err)
		}
	}

	db.opt.Infof(db.LevelsToString())
	if lcErr := db.lc.close(); err == nil {
		err = y.Wrap(lcErr, "DB.Close")
	}
	db.opt.Debugf("Waiting for closer")
	db.closers.updateSize.SignalAndWait()
	db.orc.Stop()
	db.blockCache.Close()
	db.indexCache.Close()

	atomic.StoreUint32(&db.isClosed, 1)

	if manifestErr := db.manifest.close(); err == nil {
		err = y.Wrap(manifestErr, "DB.Close")
	}

	if e := db.opt.cssCli.Sync(); e != nil {
		err = y.Wrap(e, "DB.Close")
	}
	db.closers.ioMonitor.Signal()
	return err
}

// VerifyChecksum verifies checksum for all tables on all levels.
// This method can be used to verify checksum, if opt.ChecksumVerificationMode is NoVerification.
func (db *DB) VerifyChecksum() error {
	return db.lc.verifyChecksum()
}

// Sync syncs database content to disk. This function provides
// more control to user to sync data whenever required.
func (db *DB) Sync() error {
	return nil
}

// getMemtables returns the current memtables and get references.
func (db *DB) getMemTables() ([]*memTable, func()) {
	db.lock.RLock()
	defer db.lock.RUnlock()

	var tables []*memTable

	// Get mutable memtable.
	tables = append(tables, db.mt)
	db.mt.IncrRef()

	// Get immutable memtables.
	last := len(db.imm) - 1
	for i := range db.imm {
		tables = append(tables, db.imm[last-i])
		db.imm[last-i].IncrRef()
	}
	return tables, func() {
		for _, tbl := range tables {
			tbl.DecrRef()
		}
	}
}

// get returns the value in memtable or disk for given key.
func (db *DB) get(key []byte) (y.ValueStruct, error) {
	if db.IsClosed() {
		return y.ValueStruct{}, ErrDBClosed
	}
	tables, decr := db.getMemTables() // Lock should be released.
	defer decr()

	var maxVs y.ValueStruct

	y.NumGetsAdd(db.opt.MetricsEnabled, 1)
	for i := 0; i < len(tables); i++ {
		vs := tables[i].sl.Get(key)
		y.NumMemtableGetsAdd(db.opt.MetricsEnabled, 1)
		if vs.Meta == 0 && vs.Value == nil {
			// not found
			continue
		}
		return vs, nil
	}
	return db.lc.get(key, maxVs, 0)
}

func (db *DB) getValueBlob(key []byte, vp valueBlobPtr) ([]byte, func(), error) {
	fp, err := db.opt.cssCli.OpenFile(createVBlobId(key, vp.ts), int(vp.valLen))
	if err == nil {
		return fp.ReadAll(), func() {
			fp.Close(int64(vp.valLen))
		}, nil
	} else {
		return nil, nil, y.Wrap(err, "fail to get value blob: ")
	}
}

func (db *DB) writeToLSM(b *request) error {
	// per request same ts.
	reqTs := rtts.GetTs()
	for _, entry := range b.Entries {

		// use system time as timestamp
		entry.Key = y.KeyWithTs(entry.Key, reqTs)
		err := db.mt.Put(entry.Key,
			y.ValueStruct{
				Value:     entry.Value,
				Meta:      entry.meta,
				UserMeta:  entry.UserMeta,
				ExpiresAt: entry.ExpiresAt,
			})
		if err != nil {
			return y.Wrapf(err, "while writing to memTable")
		}
	}
	return nil
}

func (db *DB) writeRequestsImpl(reqs []*request) ([]*request, error) {
	if len(reqs) == 0 {
		return nil, nil
	}

	// complete the requests and assign with errors if there is.
	done := func(doneUtil int, err error) {
		for _, r := range reqs[:doneUtil] {
			r.Err = err
			r.Wg.Done()
		}
	}
	var err error
	completed := 0

	var count int
	for reqIdx, b := range reqs {
		if len(b.Entries) == 0 {
			completed++
			continue
		}
		count += len(b.Entries)
		var i uint64
		var mt *memTable
		for mt, err = db.ensureRoomForWrite(); err == errNoRoom; {
			if mt, err = db.ensureRoomForWrite(); err != errNoRoom {
				break // succeed in scheduling the flush task.
			}
			i++
			if i%100 == 0 {
				db.opt.Debugf("Making room for writes")
			}
			time.Sleep(10 * time.Millisecond)
		}
		if mt != nil {
			// the mt is scheduled to flush, but flushing can takes long. We sync the backing wal of this memtable to commit the requests first
			var syncErr error
			if db.opt.SyncWrites {
				syncErr = mt.SyncWAL()
			}
			go mt.DecrRef() // release the ref count after sync call
			if syncErr != nil {
				err = syncErr // overwrite the error with sync error.
			}
			done(reqIdx, syncErr)
			// return to do the next memtable inserts from a fresh call of writeRequestsImpl
			return reqs[reqIdx:], err
		}
		// errors can only be switch error which definitely returns the last memtable and is already handled in the mt != nil block
		y.AssertTrue(err == nil)
		if err = db.writeToLSM(b); err != nil {
			// current request failed. set error for it and break the loop
			err = y.Wrap(err, "writeRequests") // this error will be set for later requests.
			// request from this onward will be set error and aborted by the caller writeRequests.
			break
		}
		completed++
	}
	if db.opt.SyncWrites {
		syncErr := db.mt.SyncWAL()
		if syncErr != nil {
			// syncErr will override err.
			err = syncErr
		}
	}
	// done the completed requests with no error
	done(completed, err)
	db.opt.Debugf("%d entries written", count)
	if completed == len(reqs) {
		// finished all
		return nil, err
	}
	// return the unfinished set, and the error status so far
	return reqs[completed:], err
}

// writeRequests is called serially by only one goroutine.
func (db *DB) writeRequests(reqs []*request) error {
	var err error
	for len(reqs) > 0 {
		reqs, err = db.writeRequestsImpl(reqs)
		if err != nil {
			// abort any unfinished requests
			for _, r := range reqs {
				r.Err = err
				r.Wg.Done()
			}
			return err
		}
	}

	return nil
}

func (db *DB) sendToWriteCh(entries []*Entry) (*request, error) {
	if atomic.LoadInt32(&db.blockWrites) == 1 {
		return nil, ErrBlockedWrites
	}
	var count, size int64
	vBlobs := make([]*valueBlob, 0, len(entries))
	for _, e := range entries {
		estimatedSize, vBlob := e.maybeSeparateValue(db.valueThreshold())
		size += estimatedSize
		if vBlob != nil {
			vBlobs = append(vBlobs, vBlob)
		}
		count++
	}
	if count >= db.opt.maxBatchCount || size >= db.opt.maxBatchSize {
		return nil, ErrBatchTooBig
	}

	req := requestPool.Get().(*request)
	req.reset()
	req.Entries = entries

	// persist the large value blocks and set pointer to entry first by individual writer
	errArr := make([]error, len(vBlobs))
	var wg sync.WaitGroup
	persistVBlob := func(vb *valueBlob, idx int) {
		defer wg.Done()
		// need to deal with error
		vlen := len(vb.val)
		fp, err := db.opt.cssCli.OpenNewFile(vb.id, vlen)
		if err != nil {
			errArr[idx] = err
			return
		}
		fp.Write(vb.val)
		errArr[idx] = fp.Sync()
	}

	// no rate limiting here, we expect the value separation here is only used for very large value blobs.
	for idx, vb := range vBlobs {
		wg.Add(1)
		go persistVBlob(vb, idx)
	}
	wg.Wait()

	for _, err := range errArr {
		if err != nil {
			fmt.Printf("persist blob: %v", err)
			return nil, fmt.Errorf("persist vblob error")
		}
	}

	req.Wg.Add(1)
	req.IncrRef()     // for db write
	db.writeCh <- req // Handled in doWrites.
	y.NumPutsAdd(db.opt.MetricsEnabled, int64(len(entries)))

	return req, nil
}

func (db *DB) doWrites(lc *z.Closer) {
	defer lc.Done()
	pendingCh := make(chan struct{}, 1)

	writeRequests := func(reqs []*request) {
		if err := db.writeRequests(reqs); err != nil {
			db.opt.Errorf("writeRequests: %v", err)
		}
		<-pendingCh
	}

	// This variable tracks the number of pending writes.
	reqLen := new(expvar.Int)
	// the key using dir serves no purpose
	y.PendingWritesSet(db.opt.MetricsEnabled, "pending-writes", reqLen)

	reqs := make([]*request, 0, 10)
	for {
		var r *request
		select {
		case r = <-db.writeCh:
		case <-lc.HasBeenClosed():
			goto closedCase
		}

		for {
			reqs = append(reqs, r)
			reqLen.Set(int64(len(reqs)))

			if len(reqs) >= 3*kvWriteChCapacity {
				pendingCh <- struct{}{} // blocking.
				goto writeCase
			}

			select {
			// Either push to pending, or continue to pick from writeCh.
			case r = <-db.writeCh:
			case pendingCh <- struct{}{}:
				goto writeCase
			case <-lc.HasBeenClosed():
				goto closedCase
			}
		}

	closedCase:
		// All the pending request are drained.
		// Don't close the writeCh, because it has be used in several places.
		for {
			select {
			case r = <-db.writeCh:
				reqs = append(reqs, r)
			default:
				pendingCh <- struct{}{} // Push to pending before doing a write.
				writeRequests(reqs)
				return
			}
		}

	writeCase:
		go writeRequests(reqs)
		reqs = make([]*request, 0, 10)
		reqLen.Set(0)
	}
}

func (db *DB) batchSet(entries []*Entry) error {
	req, err := db.sendToWriteCh(entries)
	if err != nil {
		return err
	}

	return req.Wait()
}

func (db *DB) batchSetAsync(entries []*Entry, f func(error)) error {
	req, err := db.sendToWriteCh(entries)
	if err != nil {
		return err
	}
	go func() {
		err := req.Wait()
		// Write is complete. Let's call the callback function now.
		f(err)
	}()
	return nil
}

var errNoRoom = errors.New("No room for write")

// ensureRoomForWrite is always called serially.
// the memtable reference will be incresed and sent back so caller can finish any task not completed.
func (db *DB) ensureRoomForWrite() (*memTable, error) {
	var err error
	db.lock.Lock()
	defer db.lock.Unlock()

	y.AssertTrue(db.mt != nil) // A nil mt indicates that DB is being closed.
	if (!db.mt.exceedSize(db.opt.MemTableSize)) && (!db.opt.dlogHandler.IsSwitchLogRequired()) {
		return nil, nil
	}

	select {
	case db.flushChan <- flushTask{mt: db.mt}:
		db.opt.Debugf("Flushing memtable, mt.size=%d size of flushChan: %d\n",
			db.mt.sl.MemSize(), len(db.flushChan))
		// We manage to push this task. Let's modify imm.
		db.imm = append(db.imm, db.mt)
		retmt := db.mt
		retmt.IncrRef() // we are still protected by db mutex. return to caller, caller is responsible to release after processing left over task
		db.mt, err = db.newMemTable()
		if err != nil {
			return retmt, y.Wrapf(err, "cannot create new mem table")
		}
		// New memtable is empty. We certainly have room.
		return retmt, nil
	default:
		// We need to do this to unlock and allow the flusher to modify imm.
		return nil, errNoRoom
	}
}

func arenaSize(opt Options) int64 {
	return opt.MemTableSize + opt.maxBatchSize + opt.maxBatchCount*int64(skl.MaxNodeSize)
}

// buildL0Table builds a new table from the memtable.
func buildL0Table(ft flushTask, bopts table.Options, uid string) *table.Builder {
	iter := ft.mt.sl.NewIterator()
	defer iter.Close()
	b := table.NewTableBuilder(bopts, uid)
	for iter.SeekToFirst(); iter.Valid(); iter.Next() {
		if len(ft.dropPrefixes) > 0 && hasAnyPrefixes(iter.Key(), ft.dropPrefixes) {
			continue
		}
		vs := iter.Value()
		var vp valueBlobPtr
		if vs.Meta&bitValuePointer > 0 {
			vp.Decode(vs.Value)
		}
		b.Add(iter.Key(), iter.Value(), uint32(vp.valLen))
	}
	return b
}

type flushTask struct {
	mt           *memTable
	dropPrefixes [][]byte
}

func (db *DB) maybeLogFlushStats(tableId string, timeStart time.Time) {
	if !db.opt.AlwaysLogFlushStats {
		return
	}
	dur := time.Since(timeStart)
	db.opt.Infof("flush to table: %s, took: %v\n", tableId, dur.Round(time.Millisecond))
}

// handleFlushTask must be run serially.
func (db *DB) handleFlushTask(ft flushTask) error {
	// There can be a scenario, when empty memtable is flushed.
	if ft.mt.sl.Empty() {
		return nil
	}

	timeStart := time.Now()

	fileID := db.lc.reserveFileID()

	bopts := buildTableOptions(db)
	builder := buildL0Table(ft, bopts, fileID)
	defer builder.Close()

	// buildL0Table can return nil if the none of the items in the skiplist are
	// added to the builder. This can happen when drop prefix is set and all
	// the items are skipped.
	if builder.Empty() {
		// builder.Finish()
		builder.Done()
		return nil
	}

	var tbl *table.Table
	var err error
	builder.Finalize(true)
	// if there is SSTCache, always try to add L0 tables to sstcache.
	tbl, err = table.CreateTable(table.NewFilenameFromStrID(fileID), builder, db.opt.cssCli, db.opt.sstCache)
	if err != nil {
		return y.Wrap(err, "error while creating table")
	}
	// We own a ref on tbl.
	// set the minWal.
	fid := ft.mt.wal.path
	id, err := strconv.ParseInt(fid[:len(fid)-len(memFileExt)], 10, 64)
	y.AssertTrue(err == nil)                      // parse should never fail.
	err = db.lc.addLevel0Table(uint64(id+1), tbl) // This will incrRef
	_ = tbl.DecrRef()                             // Releases our ref.
	if err == nil {
		db.maybeLogFlushStats(fileID, timeStart)
	}
	return err
}

// flushMemtable must keep running until we send it an empty flushTask. If there
// are errors during handling the flush task, we'll retry indefinitely.
func (db *DB) flushMemtable(lc *z.Closer) error {
	defer lc.Done()

	for ft := range db.flushChan {
		if ft.mt == nil {
			// We close db.flushChan now, instead of sending a nil ft.mt.
			continue
		}
		for {
			err := db.handleFlushTask(ft)
			if err == nil {
				// Update s.imm. Need a lock.
				db.lock.Lock()
				y.AssertTrue(ft.mt == db.imm[0])
				db.imm = db.imm[1:]
				ft.mt.DecrRef() // Return memory.
				db.lock.Unlock()

				break
			}
			// Encountered error. Retry indefinitely.
			db.opt.Errorf("Failure while flushing memtable to disk: %v. Retrying...\n", err)
			time.Sleep(time.Second)
		}
	}
	return nil
}

func (db *DB) calculateSize() {
	newInt := func(val int64) *expvar.Int {
		v := new(expvar.Int)
		v.Add(val)
		return v
	}

	lsmSize, vlogSize, err := db.opt.cssCli.EstimateSize()
	if err != nil {
		db.opt.Debugf("Got error while calculating total size of storage")
	}
	y.LSMSizeSet(db.opt.MetricsEnabled, "lsmsize", newInt(lsmSize))
	y.VlogSizeSet(db.opt.MetricsEnabled, "vlogsize", newInt(vlogSize))
}

func (db *DB) updateSize(lc *z.Closer) {
	defer lc.Done()

	metricsTicker := time.NewTicker(time.Minute)
	defer metricsTicker.Stop()

	for {
		select {
		case <-metricsTicker.C:
			db.calculateSize()
		case <-lc.HasBeenClosed():
			return
		}
	}
}

func (db *DB) Size() (lsm, vlog int64) {
	if y.LSMSizeGet(db.opt.MetricsEnabled, "lsmsize") == nil {
		lsm, vlog = 0, 0
		return
	}
	lsm = y.LSMSizeGet(db.opt.MetricsEnabled, "lsmsize").(*expvar.Int).Value()
	vlog = y.VlogSizeGet(db.opt.MetricsEnabled, "vlogsize").(*expvar.Int).Value()
	return
}

// Tables gets the TableInfo objects from the level controller. If withKeysCount
// is true, TableInfo objects also contain counts of keys for the tables.
func (db *DB) Tables() []TableInfo {
	return db.lc.getTableInfo()
}

// Levels gets the LevelInfo.
func (db *DB) Levels() []LevelInfo {
	return db.lc.getLevelInfo()
}

// EstimateSize can be used to get rough estimate of data size for a given prefix.
func (db *DB) EstimateSize(prefix []byte) (uint64, uint64) {
	var onDiskSize, uncompressedSize uint64
	tables := db.Tables()
	for _, ti := range tables {
		if bytes.HasPrefix(ti.Left, prefix) && bytes.HasPrefix(ti.Right, prefix) {
			onDiskSize += uint64(ti.OnDiskSize)
			uncompressedSize += uint64(ti.UncompressedSize)
		}
	}
	return onDiskSize, uncompressedSize
}

// MaxBatchCount returns max possible entries in batch
func (db *DB) MaxBatchCount() int64 {
	return db.opt.maxBatchCount
}

// MaxBatchSize returns max possible batch size
func (db *DB) MaxBatchSize() int64 {
	return db.opt.maxBatchSize
}

func (db *DB) stopMemoryFlush() {
	// Stop memtable flushes.
	if db.closers.memtable != nil {
		close(db.flushChan)
		db.closers.memtable.SignalAndWait()
	}
}

func (db *DB) stopCompactions() {
	// Stop compactions.
	if db.closers.compactors != nil {
		db.closers.compactors.SignalAndWait()
	}
}

// Opts returns a copy of the DB options.
func (db *DB) Opts() Options {
	return db.opt
}

type CacheType int

const (
	BlockCache CacheType = iota
	IndexCache
)

// CacheMaxCost updates the max cost of the given cache (either block or index cache).
// The call will have an effect only if the DB was created with the cache. Otherwise it is
// a no-op. If you pass a negative value, the function will return the current value
// without updating it.
func (db *DB) CacheMaxCost(cache CacheType, maxCost int64) (int64, error) {
	if db == nil {
		return 0, nil
	}

	if maxCost < 0 {
		switch cache {
		case BlockCache:
			return db.blockCache.MaxCost(), nil
		case IndexCache:
			return db.indexCache.MaxCost(), nil
		default:
			return 0, errors.Errorf("invalid cache type")
		}
	}

	switch cache {
	case BlockCache:
		db.blockCache.UpdateMaxCost(maxCost)
		return maxCost, nil
	case IndexCache:
		db.indexCache.UpdateMaxCost(maxCost)
		return maxCost, nil
	default:
		return 0, errors.Errorf("invalid cache type")
	}
}

func (db *DB) LevelsToString() string {
	levels := db.Levels()
	h := func(sz int64) string {
		return humanize.IBytes(uint64(sz))
	}
	base := func(b bool) string {
		if b {
			return "B"
		}
		return " "
	}

	var b strings.Builder
	b.WriteRune('\n')
	for _, li := range levels {
		b.WriteString(fmt.Sprintf(
			"Level %d [%s]: NumTables: %02d. Size: %s of %s. Score: %.2f->%.2f"+
				" StaleData: %s Target FileSize: %s\n",
			li.Level, base(li.IsBaseLevel), li.NumTables,
			h(li.Size), h(li.TargetSize), li.Score, li.Adjusted, h(li.StaleDatSize),
			h(li.TargetFileSize)))
	}
	b.WriteString("Level Done\n")
	return b.String()
}
