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
	"fmt"
	"log"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"hakes-store/hakes-store/io"
	"hakes-store/hakes-store/table"

	"github.com/dgraph-io/badger/v3/options"
	"github.com/dgraph-io/badger/v3/y"
	"github.com/dgraph-io/ristretto/z"
)

// compactor is a compaction executor to handle offloaded compaction task from a HakesKV instance. The HakesKV instance is responsible to define a compaction definition, coordiates it with other conflicting operations and perfrom the GC when the compactor returns.
// it reimplements two functions that is key to compaction: `subcompact` and `compactBuildTables` avoiding the dependency of a running HakesKV instance.

type TableMeta struct {
	Tid        string
	Size       int
	InSstCache bool
}

func genMeta(t *table.Table) *TableMeta {
	return &TableMeta{t.Filename(), int(t.Size()), t.InSstCache()}
}

func (m *TableMeta) load(topts table.Options, sc io.SstCache) (*table.Table, error) {
	if m.InSstCache && sc != nil {
		if mf, err := sc.Get(m.Tid); err == nil {
			return table.OpenTable(mf, topts)
		} else {
			log.Printf("Failed to load a table from SSTCache, which is said cached: %v", err)
		}
	}
	// not in cache
	if mf, err := topts.CssCli.OpenFile(m.Tid, m.Size); err != nil {
		return nil, err
	} else {
		return table.OpenTable(mf, topts)
	}
}

func (m *TableMeta) String() string {
	return fmt.Sprintf("Table-%v (%dB)", m.Tid, m.Size)
}

func parallelLoadTablesWithRetry(tl []*TableMeta, topts table.Options, retryCount int, sc io.SstCache) ([]*table.Table, error) {
	ret := make([]*table.Table, len(tl))
	var wg sync.WaitGroup
	var hasErr atomic.Bool
	var retErr error
	for idx, meta := range tl {
		if hasErr.Load() {
			break
		}
		wg.Add(1)
		go func(idx int, meta *TableMeta) {
			defer wg.Done()
			for i := 0; i < retryCount; i++ {
				time.Sleep(time.Duration(i) * time.Millisecond)
				if t, err := meta.load(topts, sc); t != nil {
					ret[idx] = t
					return
				} else if err != nil {
					// debug print
					log.Printf("loading %v error: %v", meta, err)
				}
			}
			if !hasErr.Load() && hasErr.CompareAndSwap(false, true) {
				retErr = errFailedLoadTable
			}
		}(idx, meta)
	}
	wg.Wait()
	if hasErr.Load() {
		// close all tables
		for _, t := range ret {
			if t != nil {
				t.Close(-1)
			}
		}
		return nil, retErr
	}
	return ret, nil
}

// a subset of table option which can be readily serialized.
type CompactionOption struct {
	TableSize            uint64
	BlockSize            int
	BloomFalsePositive   float64
	ChkMode              options.ChecksumVerificationMode
	Compression          options.CompressionType
	ZstdCompressionLevel int
}

// encoding in json for payload.
type CompactionJobDef struct {
	OutTablePrefix    string // hakeskv instance id + job id.
	TablePath         string
	TopTables         []*TableMeta
	BotTables         []*TableMeta
	ThisLevel         int
	NextLevel         int
	Opts              CompactionOption
	NumVersionsToKeep int
	HasOverlap        bool
	DiscardTs         uint64
	DropPrefixes      [][]byte
	UseSstCache       bool
}

func (r *CompactionJobDef) Encode() ([]byte, error) {
	return json.Marshal(r)
}

func (r *CompactionJobDef) Decode(in []byte) error {
	return json.Unmarshal(in, r)
}

func prepareCompactionJobDef(cd compactDef, kv *DB) *CompactionJobDef {
	var top, bot []*TableMeta
	for _, t := range cd.top {
		top = append(top, genMeta(t))
	}
	for _, t := range cd.bot {
		bot = append(bot, genMeta(t))
	}

	return &CompactionJobDef{
		OutTablePrefix: kv.idAlloc.getNext(), // get next is the easiest way to avoid table name collision.
		TablePath:      kv.opt.cssCli.Config(),
		TopTables:      top,
		BotTables:      bot,
		ThisLevel:      cd.thisLevel.level,
		NextLevel:      cd.nextLevel.level,
		Opts: CompactionOption{
			TableSize:            uint64(cd.t.fileSz[cd.nextLevel.level]),
			BlockSize:            kv.opt.BlockSize,
			BloomFalsePositive:   kv.opt.BloomFalsePositive,
			ChkMode:              kv.opt.ChecksumVerificationMode,
			Compression:          kv.opt.Compression,
			ZstdCompressionLevel: kv.opt.ZSTDCompressionLevel,
		},
		NumVersionsToKeep: kv.opt.NumVersionsToKeep,
		HasOverlap:        cd.hasOverlap,
		DiscardTs:         cd.discardTs,
		DropPrefixes:      cd.dropPrefixes,
		UseSstCache:       cd.useSstCache,
	}
}

type CompactionJobReply struct {
	Success   bool
	NewTables []*TableMeta
}

func (r *CompactionJobReply) Encode() ([]byte, error) {
	return json.Marshal(r)
}

func (r *CompactionJobReply) Decode(in []byte) error {
	return json.Unmarshal(in, r)
}

type CompactionJob struct {
	*CompactionJobDef
	// prefetch setting
	prefetchOpts table.PrefetchOptions
	// return
	registered    bool
	disableSCLoad bool
	wg            sync.WaitGroup
	reply         CompactionJobReply
}

func NewCompactionJob(jd *CompactionJobDef, prefetchOpts table.PrefetchOptions, disableSCLoad bool) *CompactionJob {
	job := CompactionJob{}
	job.CompactionJobDef = jd
	job.prefetchOpts = prefetchOpts
	job.disableSCLoad = disableSCLoad
	return &job
}

// called once before scheduling the compaction job
func (j *CompactionJob) reg() {
	if !j.registered {
		j.registered = true
		j.wg.Add(1)
	}
}

// caller wait on the job's reply after scheduling this job
func (j *CompactionJob) GetResult() (bool, CompactionJobReply) {
	j.wg.Wait()
	return j.reply.Success, j.reply
}

func (j *CompactionJob) complete(r CompactionJobReply) {
	j.reply = r
	j.wg.Done()
}

func (j *CompactionJobDef) getTableOptions() table.Options {
	return table.Options{
		TableSize:            j.Opts.TableSize,
		BlockSize:            j.Opts.BlockSize,
		BloomFalsePositive:   j.Opts.BloomFalsePositive,
		ChkMode:              j.Opts.ChkMode,
		Compression:          j.Opts.Compression,
		ZSTDCompressionLevel: j.Opts.ZstdCompressionLevel,
	}
}

// a compactor run one job at a time
type Compactor struct {
	// context of a compaction run
	opts      table.Options
	topTables []*table.Table
	botTables []*table.Table
	splits    []keyRange

	gidAlloc  func() string
	cssCli    io.CSSCli
	allocPool *z.AllocatorPool
	sc        io.SstCache
}

// Compactors are created and launched as goroutines by calling run and fetch task and complete.
// The owner of compactor should provide the css client and alloc pool shared.
func NewCompactor(cssCli io.CSSCli, allocPool *z.AllocatorPool, gidAlloc func() string, sc io.SstCache) *Compactor {
	return &Compactor{
		cssCli:    cssCli,
		allocPool: allocPool,
		gidAlloc:  gidAlloc,
		sc:        sc,
	}
}

func (c *Compactor) reset() {
	for _, t := range c.topTables {
		if t != nil {
			t.Close(-1)
		}
	}
	for _, t := range c.botTables {
		if t != nil {
			t.Close(-1)
		}
	}
	c.topTables = c.topTables[:0]
	c.botTables = c.botTables[:0]
	c.splits = c.splits[:0]
}

func (c *Compactor) init(j *CompactionJob) {
	c.reset()
	c.opts = j.getTableOptions()
	c.opts.AllocPool = c.allocPool
	c.opts.CssCli = c.cssCli
	if c.cssCli.Config() != j.TablePath {
		log.Printf("reconnection from %v to %v", c.cssCli.Config(), j.TablePath)
		c.cssCli.Connect(j.TablePath)
	}
}

func (c *Compactor) loadTables(j *CompactionJob) error {

	// parallel load
	var hasError atomic.Bool
	var loadWg sync.WaitGroup
	loadWg.Add(2)

	sc := c.sc
	if j.disableSCLoad {
		sc = nil
	}

	// load top tables
	go func() {
		defer loadWg.Done()
		if loaded, err := parallelLoadTablesWithRetry(j.TopTables, c.opts, 10, c.sc); err != nil {
			hasError.Store(true)
		} else {
			c.topTables = loaded
		}
	}()

	// load bot tables
	go func() {
		defer loadWg.Done()
		if loaded, err := parallelLoadTablesWithRetry(j.BotTables, c.opts, 10, sc); err != nil {
			hasError.Store(true)
		} else {
			c.botTables = loaded
		}
	}()

	loadWg.Wait()

	if hasError.Load() {
		c.reset()
		return errFailedLoadTable
	}
	return nil
}

func (c *Compactor) subcompact(it y.Iterator, kr keyRange, j *CompactionJob, inflightBuilders *y.Throttle, res chan<- *table.Table) error {
	noExceedOverlapCheck := func(keyRange) bool {
		return false
	}

	// seek the iterator to correct position
	if len(kr.left) > 0 {
		it.Seek(kr.left)
	} else {
		it.Rewind()
	}

	// iterator valid condition. for looping
	continueItr := it.Valid() && (len(kr.right) == 0 || y.CompareKeys(it.Key(), kr.right) < 0)
	for continueItr {
		fileID := fmt.Sprintf("%v%v", j.OutTablePrefix, c.gidAlloc())
		builder := table.NewTableBuilder(c.opts, fileID)
		if !builder.ReachedCapacity() {
			addKeys(it, kr, builder, c.cssCli, j.HasOverlap, j.NumVersionsToKeep, j.DiscardTs, j.DropPrefixes, noExceedOverlapCheck)
		}
		if builder.Empty() {
			builder.Done()
			builder.Close()
			// update continue Itr condition
			continueItr = it.Valid() && (len(kr.right) == 0 || y.CompareKeys(it.Key(), kr.right) < 0)
			continue
		}

		if err := inflightBuilders.Do(); err != nil {
			return err
		}
		builder.Finalize(!continueItr)
		go func(builder *table.Builder, fileID string) {
			var err error
			defer inflightBuilders.Done(err)
			defer builder.Close()
			var tbl *table.Table
			fname := table.NewFilenameFromStrID(fileID)
			if j.UseSstCache {
				tbl, err = table.CreateTable(fname, builder, c.cssCli, c.sc)
			} else {
				tbl, err = table.CreateTable(fname, builder, c.cssCli, nil)
			}
			if err != nil {
				return
			}
			res <- tbl
		}(builder, fileID)
		continueItr = it.Valid() && (len(kr.right) == 0 || y.CompareKeys(it.Key(), kr.right) < 0)
	}
	return nil
}

// comapctBuildTables here is a simplified version of levelController::compactBuildTables
func (c *Compactor) compactBuildTables(j *CompactionJob) ([]*table.Table, error) {
	var err error
	res := make(chan *table.Table, 3)
	inflightBuilders := y.NewThrottle(8 + len(c.splits))
	for id, kr := range c.splits {
		if err = inflightBuilders.Do(); err != nil {
			log.Printf("cannot start subcompaction: %+v", err)
			break
		}
		go func(kr keyRange, id int) {
			var err error
			it := table.NewMergeIterator(getCompactionItrNoBA(j.ThisLevel, c.topTables, c.botTables, &j.prefetchOpts), false)
			defer it.Close()
			err = c.subcompact(it, kr, j, inflightBuilders, res)
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
	if e := inflightBuilders.Finish(); e != nil {
		err = e
	}
	close(res)
	wg.Wait()
	if err == nil {
		c.cssCli.Sync() // no-op, synced as tables are created.
	}
	if err != nil {
		_ = decrRefs(newTables)
		return nil, err
	}
	sort.Slice(newTables, func(i, j int) bool {
		return y.CompareKeys(newTables[i].Biggest(), newTables[j].Biggest()) < 0
	})
	return newTables, nil
}

func (c *Compactor) RunDirect(j *CompactionJob) ([]*TableMeta, error) {
	defer c.reset()
	// init options
	c.init(j)
	// load tables
	if err := c.loadTables(j); err != nil {
		return nil, err
	}
	// prepare splits
	if j.ThisLevel != j.NextLevel {
		topRange := getKeyRange(c.topTables...)
		botRange := getKeyRange(c.botTables...)
		c.splits = addSplitsImpl(topRange, botRange, c.botTables)
	}
	if len(c.splits) == 0 {
		c.splits = append(c.splits, keyRange{})
	}
	// compact tables
	newTables, err := c.compactBuildTables(j)
	if err != nil {
		return nil, err
	}
	newTableIds := make([]*TableMeta, len(newTables))
	for i, t := range newTables {
		newTableIds[i] = &TableMeta{Tid: t.Name(), Size: 0, InSstCache: t.InSstCache()} // using 0, to open the CSSF at its current size.
		t.CloseTable()
	}
	return newTableIds, nil
}

func (c *Compactor) Run(j *CompactionJob, errChn chan<- error) {
	defer c.reset()
	if newTableIds, err := c.RunDirect(j); err != nil {
		errChn <- err
		j.complete(CompactionJobReply{Success: false})
	} else {
		// prepare reply
		j.complete(CompactionJobReply{Success: true, NewTables: newTableIds})
	}
}

type CompactorPool struct {
	numCompactor int
	jobQ         chan *CompactionJob
	lc           *z.Closer
	allocPool    *z.AllocatorPool
	guidAlloc    func() string
	cssConnector func() io.CSSCli // each compactor uses this supplier to create its own connection to CSS
	errChn       chan error

	// sst cache
	sc io.SstCache
}

func NewCompactorPool(numCompactor int, guidAlloc func() string, cssConnector func() io.CSSCli, sc io.SstCache) *CompactorPool {
	return &CompactorPool{
		numCompactor: numCompactor,
		jobQ:         make(chan *CompactionJob, numCompactor),
		lc:           z.NewCloser(1),
		allocPool:    z.NewAllocatorPool(8),
		guidAlloc:    guidAlloc,
		cssConnector: cssConnector,
		errChn:       make(chan error, numCompactor),
		sc:           sc,
	}
}

func (p *CompactorPool) runCompactor(id int, lc *z.Closer) {
	defer lc.Done()
	c := NewCompactor(p.cssConnector(), p.allocPool, p.guidAlloc, p.sc)
	for {
		select {
		case job := <-p.jobQ:
			// process the job
			c.Run(job, p.errChn)
		case <-lc.HasBeenClosed():
			return
		}
	}
}

func (p *CompactorPool) Launch() {
	p.lc.AddRunning(p.numCompactor)
	for i := 0; i < p.numCompactor; i++ {
		go p.runCompactor(i, p.lc)
	}

	// the launch calling goroutine will become a monitor on the error channel
	defer p.lc.Done()
	for {
		select {
		case err := <-p.errChn:
			log.Printf("[CompactorPool] failed compaction job: %v", err)
		case <-p.lc.HasBeenClosed():
			return
		}
	}
}

func (p *CompactorPool) Schedule(j *CompactionJob) {
	j.reg()
	p.jobQ <- j
}

func (p *CompactorPool) Stop() {
	p.lc.SignalAndWait()
}
