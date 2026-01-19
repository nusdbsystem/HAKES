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
	"context"
	"encoding/hex"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	otrace "go.opencensus.io/trace"

	pb "hakes-store/hakes-store/kv/proto"

	"hakes-store/hakes-store/io"
	"hakes-store/hakes-store/table"

	"github.com/dgraph-io/badger/v3/y"
	"github.com/dgraph-io/ristretto/z"
	"github.com/pkg/errors"
)

type levelsController struct {
	l0stallsMs int64 // Atomic

	// The following are initialized once and const.
	levels []*levelHandler
	kv     *DB

	cstatus compactStatus
}

// revertToManifest checks that all necessary table files exist and removes all table files not
// referenced by the manifest. idMap is a set of table file id's that were read from the directory
// listing.
func revertToManifest(kv *DB, mf *Manifest, idMap map[string]struct{}) error {
	// 1. Check all files in manifest exist.
	for id := range mf.Tables {
		if _, ok := idMap[id]; !ok {
			return fmt.Errorf("file does not exist for table %v", id)
		}
	}

	// 2. Delete files that shouldn't exist.
	for id := range idMap {
		if _, ok := mf.Tables[id]; !ok {
			kv.opt.Debugf("Table file %d not referenced in MANIFEST\n", id)
			filename := table.NewFilenameFromStrID(id)
			if err := kv.opt.cssCli.Delete(filename); err != nil {
				return y.Wrapf(err, "While removing table %v", id)
			}
		}
	}

	return nil
}

func newLevelsController(db *DB, mf *Manifest) (*levelsController, error) {
	y.AssertTrue(db.opt.NumLevelZeroTablesStall > db.opt.NumLevelZeroTables)
	s := &levelsController{
		kv:     db,
		levels: make([]*levelHandler, db.opt.MaxLevels),
	}
	s.cstatus.tables = make(map[string]struct{})
	s.cstatus.levels = make([]*levelCompactStatus, db.opt.MaxLevels)

	for i := 0; i < db.opt.MaxLevels; i++ {
		s.levels[i] = newLevelHandler(db, i)
		s.cstatus.levels[i] = new(levelCompactStatus)
	}

	if err := revertToManifest(db, mf, getIDMap(db.opt.cssCli)); err != nil {
		return nil, err
	}

	var mu sync.Mutex
	tables := make([][]*table.Table, db.opt.MaxLevels)
	throttle := y.NewThrottle(32)

	start := time.Now()
	var numOpened int32
	tick := time.NewTicker(3 * time.Second)
	defer tick.Stop()

	// collect valid ssb

	for fileID, tf := range mf.Tables {
		// fname := table.NewFilename(fileID, db.opt.Dir)
		fname := table.NewFilenameFromStrID(fileID)
		select {
		case <-tick.C:
			db.opt.Infof("%d tables out of %d opened in %s\n", atomic.LoadInt32(&numOpened),
				len(mf.Tables), time.Since(start).Round(time.Millisecond))
		default:
		}
		if err := throttle.Do(); err != nil {
			closeAllTables(tables)
			return nil, err
		}
		db.idAlloc.updateWithAllocated(fileID)
		go func(fname string, tf TableManifest) {
			var rerr error
			defer func() {
				throttle.Done(rerr)
				atomic.AddInt32(&numOpened, 1)
			}()
			topt := buildTableOptions(db)
			// Explicitly set Compression and DataKey based on how the table was generated.
			topt.Compression = tf.Compression

			mf, err := db.opt.cssCli.OpenFile(fname, 0)
			if err != nil {
				rerr = y.Wrapf(err, "Opening file: %q", fname)
				return
			}
			t, err := table.OpenTable(mf, topt)
			if err != nil {
				if strings.HasPrefix(err.Error(), "CHECKSUM_MISMATCH:") {
					db.opt.Errorf(err.Error())
					db.opt.Errorf("Ignoring table %s", mf.Name())
					// Do not set rerr. We will continue without this table.
				} else {
					rerr = y.Wrapf(err, "Opening table: %q", fname)
				}
				return
			}

			mu.Lock()
			tables[tf.Level] = append(tables[tf.Level], t)
			mu.Unlock()
		}(fname, tf)
	}
	if err := throttle.Finish(); err != nil {
		closeAllTables(tables)
		return nil, err
	}

	db.opt.Infof("All %d tables opened in %s\n", atomic.LoadInt32(&numOpened),
		time.Since(start).Round(time.Millisecond))
	for i, tbls := range tables {
		s.levels[i].initTables(tbls)
	}

	// Make sure key ranges do not overlap etc.
	if err := s.validate(); err != nil {
		_ = s.cleanupLevels()
		return nil, y.Wrap(err, "Level validation")
	}

	if err := db.opt.cssCli.Sync(); err != nil {
		_ = s.close()
		return nil, err
	}

	if db.opt.sstCache != nil {
		// add L0 and baselevel table to SSTCache
		baselevel := 0
		for i := 1; i < len(mf.Levels); i++ {
			if len(mf.Levels[i].Tables) > 0 {
				baselevel = i
				break
			}
		}
		cacheThrottle := y.NewThrottle(32)
		// load set
		cacheTables := make([]*table.Table, 0)
		cacheTables = append(cacheTables, s.levels[0].tables...)
		if baselevel != 0 {
			cacheTables = append(cacheTables, s.levels[baselevel].tables...)
		}
		for _, t := range cacheTables {
			cacheThrottle.Do()
			go func(t *table.Table) {
				defer cacheThrottle.Done(nil)
				// reserve the space first
				fname := t.CSSF.Name()
				charge := int(t.Size())
				if f := db.opt.sstCache.Add(fname, charge); f == nil {
					return
				} else {
					if data := t.CSSF.ReadAll(); data == nil {
						return
					} else {
						f.Write(data)
						if err := f.Sync(); err != nil {
							db.opt.Warningf("failed to sync %s to SSTCache during db open, dropping it: %v", fname, err)
							db.opt.sstCache.Drop(fname, charge)
						} else {
							// replace the fd in table meta
							t.CSSF.Close(-1)
							t.CSSF = f
							t.ReplaceDeleteCallBack(func() error {
								db.opt.sstCache.Drop(fname, charge)
								return db.opt.cssCli.Delete(fname)
							})
						}
					}
				}
			}(t)
		}
		cacheThrottle.Finish()
		db.opt.Infof("Loaded %d tables in L0 and %d tables in LBase\n")
		// we do not need to do sync for sstcache hold cssCli. No need to sync the directory.
	}

	return s, nil
}

// Closes the tables, for cleanup in newLevelsController.  (We Close() instead of using DecrRef()
// because that would delete the underlying files.)  We ignore errors, which is OK because tables
// are read-only.
func closeAllTables(tables [][]*table.Table) {
	for _, tableSlice := range tables {
		for _, table := range tableSlice {
			// _ = table.Close(-1)
			table.CloseTable()
		}
	}
}

func (s *levelsController) cleanupLevels() error {
	var firstErr error
	for _, l := range s.levels {
		if err := l.close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

func (s *levelsController) startCompact(lc *z.Closer) {
	n := s.kv.opt.NumCompactors
	lc.AddRunning(n - 1)
	for i := 0; i < n; i++ {
		go s.runCompactor(i, lc)
	}
}

type targets struct {
	baseLevel int
	targetSz  []int64
	fileSz    []int64
}

// levelTargets calculates the targets for levels in the LSM tree. The idea comes from Dynamic Level
// Sizes ( https://rocksdb.org/blog/2015/07/23/dynamic-level.html ) in RocksDB. The sizes of levels
// are calculated based on the size of the lowest level, typically L6. So, if L6 size is 1GB, then
// L5 target size is 100MB, L4 target size is 10MB and so on.
//
// L0 files don't automatically go to L1. Instead, they get compacted to Lbase, where Lbase is
// chosen based on the first level which is non-empty from top (check L1 through L6). For an empty
// DB, that would be L6.  So, L0 compactions go to L6, then L5, L4 and so on.
//
// Lbase is advanced to the upper levels when its target size exceeds BaseLevelSize. For
// example, when L6 reaches 1.1GB, then L4 target sizes becomes 11MB, thus exceeding the
// BaseLevelSize of 10MB. L3 would then become the new Lbase, with a target size of 1MB <
// BaseLevelSize.
func (s *levelsController) levelTargets() targets {
	adjust := func(sz int64) int64 {
		if sz < s.kv.opt.BaseLevelSize {
			return s.kv.opt.BaseLevelSize
		}
		return sz
	}

	t := targets{
		targetSz: make([]int64, len(s.levels)),
		fileSz:   make([]int64, len(s.levels)),
	}
	// DB size is the size of the last level.
	dbSize := s.lastLevel().getTotalSize()
	for i := len(s.levels) - 1; i > 0; i-- {
		ltarget := adjust(dbSize)
		t.targetSz[i] = ltarget
		if t.baseLevel == 0 && ltarget <= s.kv.opt.BaseLevelSize {
			t.baseLevel = i
		}
		dbSize /= int64(s.kv.opt.LevelSizeMultiplier)
	}

	tsz := s.kv.opt.BaseTableSize
	for i := 0; i < len(s.levels); i++ {
		if i == 0 {
			// Use MemTableSize for Level 0. Because at Level 0, we stop compactions based on the
			// number of tables, not the size of the level. So, having a 1:1 size ratio between
			// memtable size and the size of L0 files is better than churning out 32 files per
			// memtable (assuming 64MB MemTableSize and 2MB BaseTableSize).
			t.fileSz[i] = s.kv.opt.MemTableSize
		} else if i <= t.baseLevel {
			t.fileSz[i] = tsz
		} else {
			tsz *= int64(s.kv.opt.TableSizeMultiplier)
			t.fileSz[i] = tsz
		}
	}

	// Bring the base level down to the last empty level.
	for i := t.baseLevel + 1; i < len(s.levels)-1; i++ {
		if s.levels[i].getTotalSize() > 0 {
			break
		}
		t.baseLevel = i
	}

	// If the base level is empty and the next level size is less than the
	// target size, pick the next level as the base level.
	b := t.baseLevel
	lvl := s.levels
	if b < len(lvl)-1 && lvl[b].getTotalSize() == 0 && lvl[b+1].getTotalSize() < t.targetSz[b+1] {
		t.baseLevel++
	}
	return t
}

func (s *levelsController) runCompactor(id int, lc *z.Closer) {
	defer lc.Done()

	randomDelay := time.NewTimer(time.Duration(rand.Int31n(1000)) * time.Millisecond)
	select {
	case <-randomDelay.C:
	case <-lc.HasBeenClosed():
		randomDelay.Stop()
		return
	}

	moveL0toFront := func(prios []compactionPriority) []compactionPriority {
		idx := -1
		for i, p := range prios {
			if p.level == 0 {
				idx = i
				break
			}
		}
		// If idx == -1, we didn't find L0.
		// If idx == 0, then we don't need to do anything. L0 is already at the front.
		if idx > 0 {
			out := append([]compactionPriority{}, prios[idx])
			out = append(out, prios[:idx]...)
			out = append(out, prios[idx+1:]...)
			return out
		}
		return prios
	}

	run := func(p compactionPriority) bool {
		err := s.doCompact(id, p)
		switch err {
		case nil:
			return true
		case errFillTables:
			// pass
		default:
			s.kv.opt.Warningf("While running doCompact: %v\n", err)
		}
		return false
	}
	runOnce := func() bool {
		prios := s.pickCompactLevels()
		if id == 0 {
			// Worker ID zero prefers to compact L0 always.
			prios = moveL0toFront(prios)
		}
		for _, p := range prios {
			if id == 0 && p.level == 0 {
				// Allow worker zero to run level 0, irrespective of its adjusted score.
			} else if p.adjusted < 1.0 {
				break
			}
			if run(p) {
				return true
			}
		}

		return false
	}

	tryLmaxToLmaxCompaction := func() {
		p := compactionPriority{
			level: s.lastLevel().level,
			t:     s.levelTargets(),
		}
		run(p)

	}
	count := 0
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		// Can add a done channel or other stuff.
		case <-ticker.C:
			count++
			// Each ticker is 50ms so 50*200=10seconds.
			if s.kv.opt.LmaxCompaction && id == 2 && count >= 200 {
				tryLmaxToLmaxCompaction()
				count = 0
			} else {
				runOnce()
			}
		case <-lc.HasBeenClosed():
			return
		}
	}
}

type compactionPriority struct {
	level        int
	score        float64
	adjusted     float64
	dropPrefixes [][]byte
	t            targets
}

func (s *levelsController) lastLevel() *levelHandler {
	return s.levels[len(s.levels)-1]
}

// pickCompactLevel determines which level to compact.
// Based on: https://github.com/facebook/rocksdb/wiki/Leveled-Compaction
func (s *levelsController) pickCompactLevels() (prios []compactionPriority) {
	t := s.levelTargets()
	addPriority := func(level int, score float64) {
		pri := compactionPriority{
			level:    level,
			score:    score,
			adjusted: score,
			t:        t,
		}
		prios = append(prios, pri)
	}

	// Add L0 priority based on the number of tables.
	addPriority(0, float64(s.levels[0].numTables())/float64(s.kv.opt.NumLevelZeroTables))

	// All other levels use size to calculate priority.
	for i := 1; i < len(s.levels); i++ {
		// Don't consider those tables that are already being compacted right now.
		delSize := s.cstatus.delSize(i)

		l := s.levels[i]
		sz := l.getTotalSize() - delSize
		addPriority(i, float64(sz)/float64(t.targetSz[i]))
	}
	y.AssertTrue(len(prios) == len(s.levels))

	// The following code is borrowed from PebbleDB and results in healthier LSM tree structure.
	// If Li-1 has score > 1.0, then we'll divide Li-1 score by Li. If Li score is >= 1.0, then Li-1
	// score is reduced, which means we'll prioritize the compaction of lower levels (L5, L4 and so
	// on) over the higher levels (L0, L1 and so on). On the other hand, if Li score is < 1.0, then
	// we'll increase the priority of Li-1.
	// Overall what this means is, if the bottom level is already overflowing, then de-prioritize
	// compaction of the above level. If the bottom level is not full, then increase the priority of
	// above level.
	var prevLevel int
	for level := t.baseLevel; level < len(s.levels); level++ {
		if prios[prevLevel].adjusted >= 1 {
			// Avoid absurdly large scores by placing a floor on the score that we'll
			// adjust a level by. The value of 0.01 was chosen somewhat arbitrarily
			const minScore = 0.01
			if prios[level].score >= minScore {
				prios[prevLevel].adjusted /= prios[level].adjusted
			} else {
				prios[prevLevel].adjusted /= minScore
			}
		}
		prevLevel = level
	}

	// Pick all the levels whose original score is >= 1.0, irrespective of their adjusted score.
	// We'll still sort them by their adjusted score below. Having both these scores allows us to
	// make better decisions about compacting L0. If we see a score >= 1.0, we can do L0->L0
	// compactions. If the adjusted score >= 1.0, then we can do L0->Lbase compactions.
	out := prios[:0]
	for _, p := range prios[:len(prios)-1] {
		if p.score >= 1.0 {
			out = append(out, p)
		}
	}
	prios = out

	// Sort by the adjusted score.
	sort.Slice(prios, func(i, j int) bool {
		return prios[i].adjusted > prios[j].adjusted
	})
	return prios
}

// checkOverlap checks if the given tables overlap with any level from the given "lev" onwards.
func (s *levelsController) checkOverlap(tables []*table.Table, lev int) bool {
	kr := getKeyRange(tables...)
	for i, lh := range s.levels {
		if i < lev { // Skip upper levels.
			continue
		}
		lh.RLock()
		left, right := lh.overlappingTables(levelHandlerRLocked{}, kr)
		lh.RUnlock()
		if right-left > 0 {
			return true
		}
	}
	return false
}

func maybeDeleteValueBlob(key []byte, vs y.ValueStruct, cssCli io.CSSCli) {
	if vs.Meta&bitValuePointer == 0 {
		// only delete blob when this record is tied to one.
		return
	}
	var vb valueBlobPtr
	vb.Decode(vs.Value)
	keyNoTs := key[:len(key)-8]
	fmt.Printf("\ndeleting value blob %v", createVBlobId(keyNoTs, vb.ts))
	cssCli.Delete(createVBlobId(keyNoTs, vb.ts))
}

// add keys add a set of the valid keys to the new table.
// it will break if
//  1. the table capacity is filled up
//  2. overlap with too many tables in the next level
//  3. exceeds the specified key range for this subcompact
//
// after it returns and the outer for loop will create a new table to continue the compaction if there are still valid records.
func addKeys(it y.Iterator, kr keyRange, builder *table.Builder, cssCli io.CSSCli, hasOverlap bool, numVersionsToKeep int, discardTs uint64, dropPrefixes [][]byte, exceedsAllowedOverlap func(keyRange) bool) (numKeys, numSkips uint64) {
	var (
		lastKey, skipKey []byte
		numVersions      int
		// Denotes if the first key is a series of duplicate keys had
		// "DiscardEarlierVersions" set
		firstKeyHasDiscardSet bool
	)

	var rangeCheck int
	var tableKr keyRange
	for ; it.Valid(); it.Next() {
		// See if we need to skip the prefix.
		if len(dropPrefixes) > 0 && hasAnyPrefixes(it.Key(), dropPrefixes) {
			numSkips++
			// the record is dropped in this compaction, we need to clean its value blob if there is one.
			maybeDeleteValueBlob(it.Key(), it.Value(), cssCli)
			continue
		}

		// See if we need to skip this key.
		if len(skipKey) > 0 {
			if y.SameKey(it.Key(), skipKey) {
				numSkips++
				// the record is dropped in this compaction, we need to clean its value blob if there is one.
				maybeDeleteValueBlob(it.Key(), it.Value(), cssCli)
				continue
			} else {
				// reset skip key when a different key is encountered
				skipKey = skipKey[:0]
			}
		}

		if !y.SameKey(it.Key(), lastKey) {
			firstKeyHasDiscardSet = false
			if len(kr.right) > 0 && y.CompareKeys(it.Key(), kr.right) >= 0 {
				break
			}
			if builder.ReachedCapacity() {
				// Only break if we are on a different key, and have reached capacity. We want
				// to ensure that all versions of the key are stored in the same sstable, and
				// not divided across multiple tables at the same level.
				break
			}
			lastKey = y.SafeCopy(lastKey, it.Key())
			numVersions = 0
			firstKeyHasDiscardSet = it.Value().Meta&bitDiscardEarlierVersions > 0

			if len(tableKr.left) == 0 {
				tableKr.left = y.SafeCopy(tableKr.left, it.Key())
			}
			tableKr.right = lastKey

			rangeCheck++
			if rangeCheck%5000 == 0 {
				// This table's range exceeds the allowed range overlap with the level after
				// next. So, we stop writing to this table. If we don't do this, then we end up
				// doing very expensive compactions involving too many tables. To amortize the
				// cost of this check, we do it only every N keys.
				if exceedsAllowedOverlap(tableKr) {
					log.Print("Breaking due to exceedsAllowedOverlap")
					break
				}
			}
		}

		vs := it.Value()
		version := y.ParseTs(it.Key())

		isExpired := isDeletedOrExpired(vs.Meta, vs.ExpiresAt)

		// Do not discard entries inserted by merge operator. These entries will be
		// discarded once they're merged
		if version <= discardTs && vs.Meta&bitMergeEntry == 0 {
			// Keep track of the number of versions encountered for this key. Only consider the
			// versions which are below the minReadTs, otherwise, we might end up discarding the
			// only valid version for a running transaction.
			numVersions++
			// Keep the current version and discard all the next versions if
			// - The `discardEarlierVersions` bit is set OR
			// - We've already processed `NumVersionsToKeep` number of versions
			// (including the current item being processed)
			lastValidVersion := vs.Meta&bitDiscardEarlierVersions > 0 ||
				numVersions == numVersionsToKeep

			if isExpired || lastValidVersion {
				// If this version of the key is deleted or expired, skip all the rest of the
				// versions. Ensure that we're only removing versions below readTs.
				skipKey = y.SafeCopy(skipKey, it.Key())

				switch {
				// Add the key to the table only if it has not expired.
				// We don't want to add the deleted/expired keys.
				case !isExpired && lastValidVersion:
					// Add this key. We have set skipKey, so the following key versions
					// would be skipped.
				case hasOverlap:
					// If this key range has overlap with lower levels, then keep the deletion
					// marker with the latest version, discarding the rest. We have set skipKey,
					// so the following key versions would be skipped.
				default:
					// If no overlap, we can skip all the versions, by continuing here.
					numSkips++
					// expired records / deletion records fall here. for expired record we need to clean its value blob if there is one.
					maybeDeleteValueBlob(it.Key(), it.Value(), cssCli)
					continue // Skip adding this key.
				}
			}
		}
		// keep the record, add it to the new table builder.
		numKeys++
		var vp valueBlobPtr
		if vs.Meta&bitValuePointer > 0 {
			vp.Decode(vs.Value)
		}
		switch {
		case firstKeyHasDiscardSet:
			// This key is same as the last key which had "DiscardEarlierVersions" set. The
			// the next compactions will drop this key if its ts >
			// discardTs (of the next compaction).
			builder.AddStaleKey(it.Key(), vs, uint32(vp.valLen))
		case isExpired:
			// If the key is expired, the next compaction will drop it if
			// its ts > discardTs (of the next compaction).
			builder.AddStaleKey(it.Key(), vs, uint32(vp.valLen))
		default:
			builder.Add(it.Key(), vs, uint32(vp.valLen))
		}
	}
	return
}

// subcompact runs a single sub-compaction, iterating over the specified key-range only.
//
// We use splits to do a single compaction concurrently. If we have >= 3 tables
// involved in the bottom level during compaction, we choose key ranges to
// split the main compaction up into sub-compactions. Each sub-compaction runs
// concurrently, only iterating over the provided key range, generating tables.
// This speeds up the compaction significantly.
func (s *levelsController) subcompact(it y.Iterator, kr keyRange, cd compactDef,
	inflightBuilders *y.Throttle, res chan<- *table.Table) error {
	// exceedsAllowedOverlap returns true if the given key range would overlap with more than 10
	// tables from level below nextLevel (nextLevel+1). This helps avoid generating tables at Li
	// with huge overlaps with Li+1.
	exceedsAllowedOverlap := func(kr keyRange) bool {
		n2n := cd.nextLevel.level + 1
		if n2n <= 1 || n2n >= len(s.levels) {
			return false
		}
		n2nl := s.levels[n2n]
		n2nl.RLock()
		defer n2nl.RUnlock()

		l, r := n2nl.overlappingTables(levelHandlerRLocked{}, kr)
		return r-l >= 10
	}

	if len(kr.left) > 0 {
		it.Seek(kr.left)
	} else {
		it.Rewind()
	}
	continueItr := it.Valid() && (len(kr.right) == 0 || y.CompareKeys(it.Key(), kr.right) < 0)
	for continueItr {
		bopts := buildTableOptions(s.kv)
		// Set TableSize to the target file size for that level.
		bopts.TableSize = uint64(cd.t.fileSz[cd.nextLevel.level])
		fileID := s.reserveFileID()
		builder := table.NewTableBuilder(bopts, fileID)

		if !builder.ReachedCapacity() {
			// This would do the iteration and add keys to builder.
			timeStart := time.Now()
			numKeys, numSkips := addKeys(it, kr, builder, s.kv.opt.cssCli, cd.hasOverlap, s.kv.opt.NumVersionsToKeep, cd.discardTs, cd.dropPrefixes, exceedsAllowedOverlap)
			s.kv.opt.Debugf("[%d] LOG Compact. Added %d keys. Skipped %d keys. Iteration took: %v",
				cd.compactorId, numKeys, numSkips, time.Since(timeStart).Round(time.Millisecond))
		}

		if builder.Empty() {
			// Cleanup builder resources:
			builder.Done()
			builder.Close()
			continueItr = it.Valid() && (len(kr.right) == 0 || y.CompareKeys(it.Key(), kr.right) < 0)
			continue
		}

		// table build are done parallely in multiple goroutines, rate-limited by the inflightBuilders
		if err := inflightBuilders.Do(); err != nil {
			return err
		}
		builder.Finalize(!continueItr) // finalize maxkey before launching separate goroutine
		go func(builder *table.Builder, fileID string) {
			var err error
			defer inflightBuilders.Done(err)
			defer builder.Close()

			var tbl *table.Table
			fname := table.NewFilenameFromStrID(fileID)
			if cd.useSstCache {
				tbl, err = table.CreateTable(fname, builder, s.kv.opt.cssCli, s.kv.opt.sstCache)
			} else {
				tbl, err = table.CreateTable(fname, builder, s.kv.opt.cssCli, nil)
			}

			// If we couldn't build the table, return fast.
			if err != nil {
				return
			}
			res <- tbl
		}(builder, fileID)
		continueItr = it.Valid() && (len(kr.right) == 0 || y.CompareKeys(it.Key(), kr.right) < 0)
	}
	return nil
}

func getCompactionItrNoBA(lev int, topTables, botTables []*table.Table, prefetchOpts *table.PrefetchOptions) []y.Iterator {
	// Create iterators across all the tables involved first.
	var iters []y.Iterator
	if prefetchOpts.Type != table.NoPrefetch {
		switch {
		case lev == 0:
			iters = appendIteratorsReversedWithPrefetchBuffer(iters, topTables, table.NOCACHE, *prefetchOpts)
		case len(topTables) > 0:
			y.AssertTrue(len(topTables) == 1)
			iters = []y.Iterator{topTables[0].NewIteratorWithPrefetchBuffer(table.NOCACHE, *prefetchOpts)}
		}
		return append(iters, table.NewConcatIterator(botTables, table.NOCACHE, prefetchOpts))
	} else {
		// old logic
		switch {
		case lev == 0:
			iters = appendIteratorsReversed(iters, topTables, table.NOCACHE)
		case len(topTables) > 0:
			y.AssertTrue(len(topTables) == 1)
			iters = []y.Iterator{topTables[0].NewIterator(table.NOCACHE)}
		}
		// Next level has level>=1 and we can use ConcatIterator as key ranges do not overlap.
		return append(iters, table.NewConcatIterator(botTables, table.NOCACHE, nil))
	}
}

// compactBuildTables merges topTables and botTables to form a list of new tables.
func (s *levelsController) compactBuildTables(
	lev int, cd compactDef) ([]*table.Table, func() error, error) {
	topTables := cd.top
	botTables := cd.bot
	valid := botTables

	s.addSplits(&cd)

	var err error
	res := make(chan *table.Table, 3)
	inflightBuilders := y.NewThrottle(8 + len(cd.splits))
	for id, kr := range cd.splits {
		// Initiate Do here so we can register the goroutines for buildTables too.
		if err = inflightBuilders.Do(); err != nil {
			s.kv.opt.Errorf("cannot start subcompaction: %+v", err)
			break
		}
		go func(kr keyRange, id int) {
			var err error
			it := table.NewMergeIterator(getCompactionItrNoBA(lev, topTables, valid, &s.kv.opt.compactionPrefetch), false)
			defer it.Close()
			err = s.subcompact(it, kr, cd, inflightBuilders, res)
			inflightBuilders.Done(err)
		}(kr, id)
	}

	var newTables []*table.Table
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for t := range res {
			newTables = append(newTables, t)
		}
	}()

	// Wait for all table builders to finish and also for newTables accumulator to finish.
	if e := inflightBuilders.Finish(); e != nil {
		err = e
	}
	close(res)
	wg.Wait() // Wait for all tables to be picked up.

	if err == nil {
		// Ensure created files' directory entries are visible.  We don't mind the extra latency
		// from not doing this ASAP after all file creation has finished because this is a
		// background operation.
		err = s.kv.opt.cssCli.Sync()
	}

	if err != nil {
		// An error happened.  Delete all the newly created table files (by calling DecrRef
		// -- we're the only holders of a ref).
		_ = decrRefs(newTables)
		return nil, nil, y.Wrapf(err, "while running compactions for: %+v", cd)
	}

	sort.Slice(newTables, func(i, j int) bool {
		return y.CompareKeys(newTables[i].Biggest(), newTables[j].Biggest()) < 0
	})
	return newTables, func() error { return decrRefs(newTables) }, nil
}

func buildChangeSet(cd *compactDef, newTables []*table.Table) pb.ManifestChangeSet {
	changes := []*pb.ManifestChange{}
	for _, table := range newTables {
		changes = append(changes,
			newCreateChange(table.ID(), cd.nextLevel.level, table.CompressionType()))
	}
	for _, table := range cd.top {
		// Add a delete change only if the table is not in memory.
		changes = append(changes, newDeleteChange(table.ID()))
	}
	for _, table := range cd.bot {
		changes = append(changes, newDeleteChange(table.ID()))
	}
	return pb.ManifestChangeSet{Changes: changes}
}

func buildMoveChangeSet(moved []*table.Table, targetLevel int) pb.ManifestChangeSet {
	changes := []*pb.ManifestChange{}
	for _, table := range moved {
		changes = append(changes,
			newMoveChange(table.ID(), targetLevel, table.CompressionType()))
	}
	return pb.ManifestChangeSet{Changes: changes}
}

func hasAnyPrefixes(s []byte, listOfPrefixes [][]byte) bool {
	for _, prefix := range listOfPrefixes {
		if bytes.HasPrefix(s, prefix) {
			return true
		}
	}

	return false
}

type compactDef struct {
	span *otrace.Span

	compactorId int
	t           targets
	p           compactionPriority
	thisLevel   *levelHandler
	nextLevel   *levelHandler

	top []*table.Table
	bot []*table.Table

	thisRange keyRange // for upper level table
	nextRange keyRange // for bottom level table
	splits    []keyRange

	thisSize int64

	dropPrefixes [][]byte
	// added that are only used in compaction and filled before executing the compaction.
	hasOverlap bool
	discardTs  uint64

	useSstCache bool
}

func addSplitsImpl(topRange, bottomRange keyRange, bottomTables []*table.Table) (ret []keyRange) {
	// Let's say we have 10 tables in cd.bot and min width = 3. Then, we'll pick
	// 0, 1, 2 (pick), 3, 4, 5 (pick), 6, 7, 8 (pick), 9 (pick, because last table).
	// This gives us 4 picks for 10 tables.
	// In an edge case, 142 tables in bottom led to 48 splits. That's too many splits, because it
	// then uses up a lot of memory for table builder.
	// We should keep it so we have at max 5 splits.
	width := int(math.Ceil(float64(len(bottomTables)) / 5.0))
	if width < 3 {
		width = 3
	}
	skr := topRange
	skr.extend(bottomRange)

	addRange := func(right []byte) {
		skr.right = y.Copy(right)
		ret = append(ret, skr)

		skr.left = skr.right
	}

	for i, t := range bottomTables {
		// last entry in bottom table.
		if i == len(bottomTables)-1 {
			addRange([]byte{}) // empty right key means no need to check if the current iterate over the current key range. (which is not needed for the last key range.)
			return
		}
		if i%width == width-1 {
			// Right should always have ts=maxUint64 otherwise we'll lose keys
			// in subcompaction. Consider the following.
			// Top table is [A1...C3(deleted)]
			// bot table is [B1....C2]
			// This will generate splits like [A1 ... C2] . Notice that we
			// dropped the C3 which is the last key of the top table.
			// See TestCompaction/with_split test.
			right := y.KeyWithTs(y.ParseKey(t.Biggest()), 0)
			addRange(right)
		}
	}
	return
}

// addSplits can allow us to run multiple sub-compactions in parallel across the split key ranges.
// generate a) at most 5 splits b) one split with <= 3 tables (last one) and zero or more split with three tables (when there is > 3 but < 14 tables leading tables are grouped into splits with each having three tables).
func (s *levelsController) addSplits(cd *compactDef) {
	y.AssertTrue(len(cd.splits) == 0)
	if cd.thisLevel.level == cd.nextLevel.level {
		// don't do anything for L0 -> L0 and Lmax -> Lmax.
	} else {
		cd.splits = addSplitsImpl(cd.thisRange, cd.nextRange, cd.bot)
	}
	// it is not split, run a monolithic compaction
	// an empty key range means that iterator start from the beginning and there is no early return from the split but run till the end
	if len(cd.splits) == 0 {
		cd.splits = append(cd.splits, keyRange{})
	}
}

func (cd *compactDef) lockLevels() {
	cd.thisLevel.RLock()
	cd.nextLevel.RLock()
}

func (cd *compactDef) unlockLevels() {
	cd.nextLevel.RUnlock()
	cd.thisLevel.RUnlock()
}

func (cd *compactDef) allTables() []*table.Table {
	ret := make([]*table.Table, 0, len(cd.top)+len(cd.bot))
	ret = append(ret, cd.top...)
	ret = append(ret, cd.bot...)
	return ret
}

func (cd *compactDef) isTrivialMove() bool {
	if cd.nextLevel.level == 0 {
		// intra-L0 compaction
		return false
	}

	// return false for intra-level compaction
	if cd.nextLevel == nil || cd.thisLevel.level == cd.nextLevel.level {
		return false
	}

	// if there are overlapping files in bot level.
	if len(cd.bot) > 0 {
		return false
	}

	if cd.thisLevel.level == 0 && len(cd.top) > 1 {
		// L0 -> LBase but L0 has overlapping source files.
		return false
	}

	// compression is uniform so no invalidation of trivial move due to compression mismatch between two levels.

	// check next large overlapping with next level is not supported here

	return true
}

func (s *levelsController) fillTablesL0ToL0(cd *compactDef) bool {
	if cd.compactorId != 0 {
		// Only compactor zero can work on this.
		return false
	}

	cd.nextLevel = s.levels[0]
	cd.nextRange = keyRange{}
	cd.bot = nil

	// Because this level and next level are both level 0, we should NOT acquire
	// the read lock twice, because it can result in a deadlock. So, we don't
	// call compactDef.lockLevels, instead locking the level only once and
	// directly here.
	//
	// As per godocs on RWMutex:
	// If a goroutine holds a RWMutex for reading and another goroutine might
	// call Lock, no goroutine should expect to be able to acquire a read lock
	// until the initial read lock is released. In particular, this prohibits
	// recursive read locking. This is to ensure that the lock eventually
	// becomes available; a blocked Lock call excludes new readers from
	// acquiring the lock.
	y.AssertTrue(cd.thisLevel.level == 0)
	y.AssertTrue(cd.nextLevel.level == 0)
	s.levels[0].RLock()
	defer s.levels[0].RUnlock()

	s.cstatus.Lock()
	defer s.cstatus.Unlock()

	top := cd.thisLevel.tables
	var out []*table.Table
	now := time.Now()
	for _, t := range top {
		if t.Size() >= 2*cd.t.fileSz[0] {
			// This file is already big, don't include it.
			continue
		}
		if now.Sub(t.CreatedAt) < 10*time.Second {
			// Just created it 10s ago. Don't pick for compaction.
			continue
		}
		if _, beingCompacted := s.cstatus.tables[t.ID()]; beingCompacted {
			continue
		}
		out = append(out, t)
	}

	if len(out) < 4 {
		// If we don't have enough tables to merge in L0, don't do it.
		return false
	}
	cd.thisRange = infRange
	cd.top = out

	// Avoid any other L0 -> Lbase from happening, while this is going on.
	thisLevel := s.cstatus.levels[cd.thisLevel.level]
	thisLevel.ranges = append(thisLevel.ranges, infRange)
	for _, t := range out {
		s.cstatus.tables[t.ID()] = struct{}{}
	}

	// For L0->L0 compaction, we set the target file size to max, so the output is always one file.
	// This significantly decreases the L0 table stalls and improves the performance.
	cd.t.fileSz[0] = math.MaxUint32
	return true
}

func (s *levelsController) fillTablesL0ToLbase(cd *compactDef) bool {
	if cd.nextLevel.level == 0 {
		panic("Base level can't be zero.")
	}
	// We keep cd.p.adjusted > 0.0 here to allow functions in db.go to artificially trigger
	// L0->Lbase compactions. Those functions wouldn't be setting the adjusted score.
	if cd.p.adjusted > 0.0 && cd.p.adjusted < 1.0 {
		// Do not compact to Lbase if adjusted score is less than 1.0.
		return false
	}
	cd.lockLevels()
	defer cd.unlockLevels()

	top := cd.thisLevel.tables
	if len(top) == 0 {
		return false
	}

	var out []*table.Table
	if len(cd.dropPrefixes) > 0 {
		// Use all tables if drop prefix is set. We don't want to compact only a
		// sub-range. We want to compact all the tables.
		out = top

	} else {
		var kr keyRange
		// cd.top[0] is the oldest file. So we start from the oldest file first.
		for _, t := range top {
			dkr := getKeyRange(t)
			if kr.overlapsWith(dkr) {
				out = append(out, t)
				kr.extend(dkr)
			} else {
				break
			}
		}
	}
	cd.thisRange = getKeyRange(out...)
	cd.top = out

	left, right := cd.nextLevel.overlappingTables(levelHandlerRLocked{}, cd.thisRange)
	cd.bot = make([]*table.Table, right-left)
	copy(cd.bot, cd.nextLevel.tables[left:right])

	if len(cd.bot) == 0 {
		cd.nextRange = cd.thisRange
	} else {
		cd.nextRange = getKeyRange(cd.bot...)
	}
	return s.cstatus.compareAndAdd(thisAndNextLevelRLocked{}, *cd)
}

// fillTablesL0 would try to fill tables from L0 to be compacted with Lbase. If
// it can not do that, it would try to compact tables from L0 -> L0.
//
// Say L0 has 10 tables.
// fillTablesL0ToLbase picks up 5 tables to compact from L0 -> L5.
// Next call to fillTablesL0 would run L0ToLbase again, which fails this time.
// So, instead, we run fillTablesL0ToL0, which picks up rest of the 5 tables to
// be compacted within L0. Additionally, it would set the compaction range in
// cstatus to inf, so no other L0 -> Lbase compactions can happen.
// Thus, L0 -> L0 must finish for the next L0 -> Lbase to begin.
func (s *levelsController) fillTablesL0(cd *compactDef) bool {
	if ok := s.fillTablesL0ToLbase(cd); ok {
		return true
	}
	return s.fillTablesL0ToL0(cd)
}

// sortByStaleData sorts tables based on the amount of stale data they have.
// This is useful in removing tombstones.
func (s *levelsController) sortByStaleDataSize(tables []*table.Table, cd *compactDef) {
	if len(tables) == 0 || cd.nextLevel == nil {
		return
	}

	sort.Slice(tables, func(i, j int) bool {
		return tables[i].StaleDataSize() > tables[j].StaleDataSize()
	})
}

// sortByHeuristic sorts tables in increasing order of MaxVersion, so we
// compact older tables first.
func (s *levelsController) sortByHeuristic(tables []*table.Table, cd *compactDef) {
	if len(tables) == 0 || cd.nextLevel == nil {
		return
	}

	// Sort tables by max version. This is what RocksDB does.
	sort.Slice(tables, func(i, j int) bool {
		return tables[i].MaxVersion() < tables[j].MaxVersion()
	})
}

// This function should be called with lock on levels.
func (s *levelsController) fillMaxLevelTables(tables []*table.Table, cd *compactDef) bool {
	sortedTables := make([]*table.Table, len(tables))
	copy(sortedTables, tables)
	s.sortByStaleDataSize(sortedTables, cd)

	if len(sortedTables) > 0 && sortedTables[0].StaleDataSize() == 0 {
		// This is a maxLevel to maxLevel compaction and we don't have any stale data.
		return false
	}
	cd.bot = []*table.Table{}
	collectBotTables := func(t *table.Table, needSz int64) {
		totalSize := t.Size()

		j := sort.Search(len(tables), func(i int) bool {
			return y.CompareKeys(tables[i].Smallest(), t.Smallest()) >= 0
		})
		y.AssertTrue(tables[j].ID() == t.ID())
		j++
		// Collect tables until we reach the the required size.
		for j < len(tables) {
			newT := tables[j]
			totalSize += newT.Size()

			if totalSize >= needSz {
				break
			}
			cd.bot = append(cd.bot, newT)
			cd.nextRange.extend(getKeyRange(newT))
			j++
		}
	}
	now := time.Now()
	for _, t := range sortedTables {
		// If the maxVersion is above the discardTs, we won't clean anything in
		// the compaction. So skip this table.
		if t.MaxVersion() > s.kv.orc.discardAtOrBelow() {
			continue
		}
		if now.Sub(t.CreatedAt) < time.Hour {
			// Just created it an hour ago. Don't pick for compaction.
			continue
		}
		// If the stale data size is less than 10 MB, it might not be worth
		// rewriting the table. Skip it.
		if t.StaleDataSize() < 10<<20 {
			continue
		}

		cd.thisSize = t.Size()
		cd.thisRange = getKeyRange(t)
		// Set the next range as the same as the current range. If we don't do
		// this, we won't be able to run more than one max level compactions.
		cd.nextRange = cd.thisRange
		// If we're already compacting this range, don't do anything.
		if s.cstatus.overlapsWith(cd.thisLevel.level, cd.thisRange) {
			continue
		}

		// Found a valid table!
		cd.top = []*table.Table{t}

		needFileSz := cd.t.fileSz[cd.thisLevel.level]
		// The table size is what we want so no need to collect more tables.
		if t.Size() >= needFileSz {
			break
		}
		// TableSize is less than what we want. Collect more tables for compaction.
		// If the level has multiple small tables, we collect all of them
		// together to form a bigger table.
		collectBotTables(t, needFileSz)
		if !s.cstatus.compareAndAdd(thisAndNextLevelRLocked{}, *cd) {
			cd.bot = cd.bot[:0]
			cd.nextRange = keyRange{}
			continue
		}
		return true
	}
	if len(cd.top) == 0 {
		return false
	}

	return s.cstatus.compareAndAdd(thisAndNextLevelRLocked{}, *cd)
}

func (s *levelsController) fillTables(cd *compactDef) bool {
	cd.lockLevels()
	defer cd.unlockLevels()

	tables := make([]*table.Table, len(cd.thisLevel.tables))
	copy(tables, cd.thisLevel.tables)
	if len(tables) == 0 {
		return false
	}
	// We're doing a maxLevel to maxLevel compaction. Pick tables based on the stale data size.
	if cd.thisLevel.isLastLevel() {
		return s.fillMaxLevelTables(tables, cd)
	}
	// We pick tables, so we compact older tables first. This is similar to
	// kOldestLargestSeqFirst in RocksDB.
	s.sortByHeuristic(tables, cd)

	for _, t := range tables {
		cd.thisSize = t.Size()
		cd.thisRange = getKeyRange(t)
		// If we're already compacting this range, don't do anything.
		if s.cstatus.overlapsWith(cd.thisLevel.level, cd.thisRange) {
			continue
		}
		cd.top = []*table.Table{t}
		left, right := cd.nextLevel.overlappingTables(levelHandlerRLocked{}, cd.thisRange)

		cd.bot = make([]*table.Table, right-left)
		copy(cd.bot, cd.nextLevel.tables[left:right])

		if len(cd.bot) == 0 {
			cd.bot = []*table.Table{}
			cd.nextRange = cd.thisRange
			if !s.cstatus.compareAndAdd(thisAndNextLevelRLocked{}, *cd) {
				continue
			}
			return true
		}
		cd.nextRange = getKeyRange(cd.bot...)

		if s.cstatus.overlapsWith(cd.nextLevel.level, cd.nextRange) {
			continue
		}
		if !s.cstatus.compareAndAdd(thisAndNextLevelRLocked{}, *cd) {
			continue
		}
		return true
	}
	return false
}

func filterPrefixDroppedTables(tables []*table.Table, dropPrefixes [][]byte) []*table.Table {
	keepTable := func(t *table.Table) bool {
		for _, prefix := range dropPrefixes {
			if bytes.HasPrefix(t.Smallest(), prefix) && bytes.HasPrefix(t.Biggest(), prefix) {
				return false
			}
		}
		return true
	}
	var ret []*table.Table
	for _, t := range tables {
		if keepTable(t) {
			ret = append(ret, t)
		}
	}
	return ret
}

func (s *levelsController) runCompactDef(id, l int, cd compactDef) (err error) {
	if len(cd.t.fileSz) == 0 {
		return errors.New("Filesizes cannot be zero. Targets are not set")
	}
	timeStart := time.Now()

	thisLevel := cd.thisLevel
	nextLevel := cd.nextLevel

	// stats (was in compactBuildTables)
	numTables := int64(len(cd.top) + len(cd.bot))
	y.NumCompactionTablesAdd(s.kv.opt.MetricsEnabled, numTables)
	defer y.NumCompactionTablesAdd(s.kv.opt.MetricsEnabled, -numTables)
	cd.span.Annotatef(nil, "Top tables count: %v Bottom tables count: %v",
		len(cd.top), len(cd.bot))

	botToDel := cd.bot
	// skip droped tables (was in compactBuildTable)
	// dropPrefix is only applied to the bottom tables because this is only used in levelController::dropPrefixes, which only does intra-level compaction.
	cd.bot = filterPrefixDroppedTables(cd.bot, cd.dropPrefixes)

	// decide if should cache generated tables
	if s.kv.opt.sstCache != nil && s.kv.opt.sstCachePolicy != nil {
		cd.useSstCache = s.kv.opt.sstCachePolicy.UseSstCache(cd.nextLevel.level, cd.t.baseLevel)
	}

	// Table should never be moved directly between levels, always be rewritten to allow discarding
	// invalid versions.
	if s.kv.opt.enableTrivialMove && cd.isTrivialMove() {
		return s.runTrivialMove(id, cd, timeStart)
	}

	// Check overlap of the top level with the levels which are not being
	// compacted in this compaction.
	cd.hasOverlap = s.checkOverlap(cd.allTables(), cd.nextLevel.level+1)

	// Pick a discard ts, so we can discard versions below this ts. We should
	// never discard any versions starting from above this timestamp, because
	// that would affect the snapshot view guarantee provided by transactions.
	cd.discardTs = s.kv.orc.discardAtOrBelow()

	var newTables []*table.Table
	var decr func() error
	remoteScheduled := false
	if s.kv.opt.compscheduler != nil {
		// maybe schedule remote
		rcc := RemoteCompactionContext{
			kv: s.kv,
			cd: &cd,
		}
		var err error
		remoteScheduled, newTables, err = s.kv.opt.compscheduler.scheduleRemote(rcc)
		if err != nil {
			log.Printf("remote compaction scheduling error (fall back to local): %v", err)
			// set remote scheduled false to retry compaction locally
			remoteScheduled = false
		} else if remoteScheduled && len(newTables) > 0 {
			decr = func() error { return decrRefs(newTables) }
		}
	}
	if !remoteScheduled {
		var err error
		newTables, decr, err = s.compactBuildTables(l, cd)
		if err != nil {
			return err
		}
	}
	defer func() {
		// Only assign to err, if it's not already nil.
		if decErr := decr(); err == nil {
			err = decErr
		}
	}()

	// reset bot table to include those being dropped.
	cd.bot = botToDel

	changeSet := buildChangeSet(&cd, newTables)

	// We write to the manifest _before_ we delete files (and after we created files)
	if err := s.kv.manifest.addChanges(changeSet.Changes); err != nil {
		return err
	}

	// See comment earlier in this function about the ordering of these ops, and the order in which
	// we access levels when reading.
	if err := nextLevel.replaceTables(cd.bot, newTables); err != nil {
		return err
	}
	if err := thisLevel.deleteTables(cd.top); err != nil {
		return err
	}

	// Note: For level 0, while doCompact is running, it is possible that new tables are added.
	// However, the tables are added only to the end, so it is ok to just delete the first table.

	s.maybeLogCompactionStats(id, &cd, newTables, timeStart)
	return nil
}

func (s *levelsController) runTrivialMove(id int, cd compactDef, timeStart time.Time) error {
	// build change set
	changeSet := buildMoveChangeSet(cd.top, cd.nextLevel.level)
	// add to manifest
	if err := s.kv.manifest.addChanges(changeSet.Changes); err != nil {
		return err
	}
	// update levelHandlers
	if err := cd.nextLevel.replaceTables(nil, cd.top); err != nil {
		// should never happen as error only originate from decRef on first argument, which is nil here.
		return err
	}
	if err := cd.thisLevel.deleteTables(cd.top); err != nil {
		return err
	}
	// ref count correctly transfer from this level to next level if both replace and delete proceeds with no error.
	s.kv.opt.Debugf("[%d] LOG Compact (trivial move) %d->%d (%d tables with %d splits). [%s], took %v\n",
		id, cd.thisLevel.level, cd.nextLevel.level, len(cd.top), len(cd.splits),
		strings.Join(tablesToString(cd.top), " "), time.Since(timeStart).Round(time.Millisecond))
	return nil
}

func (s *levelsController) maybeLogCompactionStats(id int, cd *compactDef, newTables []*table.Table, timeStart time.Time) {
	genCompactionLog := func() string {
		from := append(tablesToString(cd.top), tablesToString(cd.bot)...)
		to := tablesToString(newTables)
		return fmt.Sprintf("[%d] LOG Compact %d->%d (%d, %d -> %d tables with %d splits)."+
			" [%s] -> [%s]",
			id, cd.thisLevel.level, cd.nextLevel.level, len(cd.top), len(cd.bot),
			len(newTables), len(cd.splits), strings.Join(from, " "), strings.Join(to, " "))
	}

	if dur := time.Since(timeStart); dur > 2*time.Millisecond {
		var expensive string
		if dur > time.Second {
			expensive = " [E]"
		}
		s.kv.opt.Debugf("%s %s, took: %v\n", expensive, genCompactionLog(),
			dur.Round(time.Millisecond))
	}

	if s.kv.opt.AlwaysLogCompactionStats {
		dur := time.Since(timeStart)
		s.kv.opt.Infof("%s, took: %v\n", genCompactionLog(), dur.Round(time.Millisecond))
	}

	if cd.thisLevel.level != 0 && len(newTables) > 2*s.kv.opt.LevelSizeMultiplier {
		s.kv.opt.Debugf("This Range (numTables: %d)\nLeft:\n%s\nRight:\n%s\n",
			len(cd.top), hex.Dump(cd.thisRange.left), hex.Dump(cd.thisRange.right))
		s.kv.opt.Debugf("Next Range (numTables: %d)\nLeft:\n%s\nRight:\n%s\n",
			len(cd.bot), hex.Dump(cd.nextRange.left), hex.Dump(cd.nextRange.right))
	}
}

func tablesToString(tables []*table.Table) []string {
	var res []string
	for _, t := range tables {
		res = append(res, fmt.Sprintf("%v(%dMB)", t.ID(), t.Size()>>20))
	}
	res = append(res, ".")
	return res
}

// doCompact picks some table on level l and compacts it away to the next level.
func (s *levelsController) doCompact(id int, p compactionPriority) error {
	l := p.level
	y.AssertTrue(l < s.kv.opt.MaxLevels) // Sanity check.
	if p.t.baseLevel == 0 {
		p.t = s.levelTargets()
	}

	_, span := otrace.StartSpan(context.Background(), "HakesKV.Compaction")
	defer span.End()

	cd := compactDef{
		compactorId:  id,
		span:         span,
		p:            p,
		t:            p.t,
		thisLevel:    s.levels[l],
		dropPrefixes: p.dropPrefixes,
	}

	// While picking tables to be compacted, both levels' tables are expected to
	// remain unchanged.
	if l == 0 {
		cd.nextLevel = s.levels[p.t.baseLevel]
		if !s.fillTablesL0(&cd) {
			return errFillTables
		}
	} else {
		cd.nextLevel = cd.thisLevel
		// We're not compacting the last level so pick the next level.
		if !cd.thisLevel.isLastLevel() {
			cd.nextLevel = s.levels[l+1]
		}
		if !s.fillTables(&cd) {
			return errFillTables
		}
	}
	defer s.cstatus.delete(cd) // Remove the ranges from compaction status.

	span.Annotatef(nil, "Compaction: %+v", cd)
	if err := s.runCompactDef(id, l, cd); err != nil {
		// This compaction couldn't be done successfully.
		s.kv.opt.Warningf("[Compactor: %d] LOG Compact FAILED with error: %+v: %+v", id, err, cd)
		return err
	}

	s.kv.opt.Debugf("[Compactor: %d] Compaction for level: %d DONE", id, cd.thisLevel.level)
	return nil
}

func (s *levelsController) addLevel0Table(minWal uint64, t *table.Table) error {
	// Add table to manifest file only if it is not opened in memory. We don't want to add a table
	// to the manifest file if it exists only in memory.
	// if !t.IsInmemory {
	// We update the manifest _before_ the table becomes part of a levelHandler, because at that
	// point it could get used in some compaction.  This ensures the manifest file gets updated in
	// the proper order. (That means this update happens before that of some compaction which
	// deletes the table.)
	s.kv.manifest.manifest.setMinWal(minWal) // addChanges will persist manifest, so update of minWal should happen before.
	err := s.kv.manifest.addChanges([]*pb.ManifestChange{
		newCreateChange(t.ID(), 0, t.CompressionType()),
	})
	if err != nil {
		return err
	}
	// }

	for !s.levels[0].tryAddLevel0Table(t) {
		// Before we unstall, we need to make sure that level 0 is healthy.
		timeStart := time.Now()
		for s.levels[0].numTables() >= s.kv.opt.NumLevelZeroTablesStall {
			time.Sleep(10 * time.Millisecond)
		}
		dur := time.Since(timeStart)
		if dur > time.Second {
			s.kv.opt.Infof("L0 was stalled for %s\n", dur.Round(time.Millisecond))
		}
		atomic.AddInt64(&s.l0stallsMs, int64(dur.Round(time.Millisecond)))
	}

	return nil
}

func (s *levelsController) close() error {
	err := s.cleanupLevels()
	return y.Wrap(err, "levelsController.Close")
}

// get searches for a given key in all the levels of the LSM tree. It returns
// key version <= the expected version (maxVs). If not found, it returns an empty
// y.ValueStruct.
func (s *levelsController) get(key []byte, maxVs y.ValueStruct, startLevel int) (
	y.ValueStruct, error) {
	if s.kv.IsClosed() {
		return y.ValueStruct{}, ErrDBClosed
	}
	// It's important that we iterate the levels from 0 on upward. The reason is, if we iterated
	// in opposite order, or in parallel (naively calling all the h.RLock() in some order) we could
	// read level L's tables post-compaction and level L+1's tables pre-compaction. (If we do
	// parallelize this, we will need to call the h.RLock() function by increasing order of level
	// number.)
	// version := y.ParseTs(key)
	for _, h := range s.levels {
		// Ignore all levels below startLevel. This is useful for GC when L0 is kept in memory.
		if h.level < startLevel {
			continue
		}
		vs, err := h.get(key) // Calls h.RLock() and h.RUnlock().
		if err != nil {
			return y.ValueStruct{}, y.Wrapf(err, "get key: %q", key)
		}
		if vs.Value == nil && vs.Meta == 0 {
			// not found in this level
			continue
		}
		return vs, nil
	}
	return y.ValueStruct{}, nil
}

func appendIteratorsReversed(out []y.Iterator, th []*table.Table, opt int) []y.Iterator {
	for i := len(th) - 1; i >= 0; i-- {
		// This will increment the reference of the table handler.
		out = append(out, th[i].NewIterator(opt))
	}
	return out
}

func appendIteratorsReversedWithPrefetchBuffer(out []y.Iterator, th []*table.Table, opt int, prefetchOpts table.PrefetchOptions) []y.Iterator {
	for i := len(th) - 1; i >= 0; i-- {
		// This will increment the reference of the table handler.
		out = append(out, th[i].NewIteratorWithPrefetchBuffer(opt, prefetchOpts))
	}
	return out
}

// appendIterators appends iterators to an array of iterators, for merging.
// Note: This obtains references for the table handlers. Remember to close these iterators.
func (s *levelsController) appendIterators(
	iters []y.Iterator, opt *IteratorOptions) []y.Iterator {
	// Just like with get, it's important we iterate the levels from 0 on upward, to avoid missing
	// data when there's a compaction.
	for _, level := range s.levels {
		iters = level.appendIterators(iters, opt)
	}
	return iters
}

// TableInfo represents the information about a table.
type TableInfo struct {
	// ID    uint64
	Level int
	Left  []byte
	Right []byte
	// KeyCount         uint32 // Number of keys in the table
	OnDiskSize       uint32
	StaleDataSize    uint32
	UncompressedSize uint32
	MaxVersion       uint64
	IndexSz          int
	BloomFilterSize  int
	ID               string
}

func (s *levelsController) getTableInfo() (result []TableInfo) {
	for _, l := range s.levels {
		l.RLock()
		for _, t := range l.tables {
			info := TableInfo{
				Level:            l.level,
				Left:             t.Smallest(),
				Right:            t.Biggest(),
				OnDiskSize:       t.OnDiskSize(),
				StaleDataSize:    t.StaleDataSize(),
				IndexSz:          t.IndexSize(),
				BloomFilterSize:  t.BloomFilterSize(),
				UncompressedSize: t.UncompressedSize(),
				MaxVersion:       t.MaxVersion(),
				ID:               t.ID(),
			}
			result = append(result, info)
		}
		l.RUnlock()
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].Level != result[j].Level {
			return result[i].Level < result[j].Level
		}
		return result[i].ID < result[j].ID
	})
	return
}

type LevelInfo struct {
	Level          int
	NumTables      int
	Size           int64
	TargetSize     int64
	TargetFileSize int64
	IsBaseLevel    bool
	Score          float64
	Adjusted       float64
	StaleDatSize   int64
}

func (s *levelsController) getLevelInfo() []LevelInfo {
	t := s.levelTargets()
	prios := s.pickCompactLevels()
	result := make([]LevelInfo, len(s.levels))
	for i, l := range s.levels {
		l.RLock()
		result[i].Level = i
		result[i].Size = l.totalSize
		result[i].NumTables = len(l.tables)
		result[i].StaleDatSize = l.totalStaleSize

		l.RUnlock()

		result[i].TargetSize = t.targetSz[i]
		result[i].TargetFileSize = t.fileSz[i]
		result[i].IsBaseLevel = t.baseLevel == i
	}
	for _, p := range prios {
		result[p.level].Score = p.score
		result[p.level].Adjusted = p.adjusted
	}
	return result
}

// verifyChecksum verifies checksum for all tables on all levels.
func (s *levelsController) verifyChecksum() error {
	var tables []*table.Table
	for _, l := range s.levels {
		l.RLock()
		tables = tables[:0]
		for _, t := range l.tables {
			tables = append(tables, t)
			t.IncrRef()
		}
		l.RUnlock()

		for _, t := range tables {
			errChkVerify := t.VerifyChecksum()
			if err := t.DecrRef(); err != nil {
				s.kv.opt.Errorf("unable to decrease reference of table: %s while "+
					"verifying checksum with error: %s", t.Filename(), err)
			}

			if errChkVerify != nil {
				return errChkVerify
			}
		}
	}

	return nil
}
