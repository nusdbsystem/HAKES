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

package table

import (
	"errors"
	"fmt"
	"log"
	"sync"
)

/*
 * Prefetch buffer implements a prefetching layer on top of CSS.
 * It is used now in compaction. It supposes single thread access.
 */

func PREFETCHLOG(msg string) {
	// log.Printf("[%d]: %v", time.Now().UnixMicro(), msg)
}

type PrefetchOptions struct {
	Type         CompactionPrefetchType
	PrefetchSize int
}

type CompactionPrefetchType uint32

const (
	// None mode indicates that a block is not compressed.
	NoPrefetch CompactionPrefetchType = 0
	// Snappy mode indicates that a block is compressed using Snappy algorithm.
	SyncPrefetch CompactionPrefetchType = 1
	// ZSTD mode indicates that a block is compressed using ZSTD algorithm.
	AsyncPrefetch CompactionPrefetchType = 2
)

func NoPrefetchOption() PrefetchOptions {
	return PrefetchOptions{Type: NoPrefetch, PrefetchSize: 0}
}

var (
	errInvalidPrefetchRange = errors.New("invalid prefetch range")
)

type PrefetchSource interface {
	Bytes(off, sz int) ([]byte, error)
}

type PrefetchBuffer interface {
	PrefetchSource
	// should be called before Bytes(); can be called independently from ByteS()
	UpdateAccess(off, sz int)
}

type PrefetchPolicy interface {
	UpdatePattern(off, sz int)
	GetPrefetchSize() int
}

type FixedPrefetchPolicy struct {
	prefetchSize int
}

func NewFixedPrefetchPolicy(size int) PrefetchPolicy {
	return &FixedPrefetchPolicy{size}
}

func (p *FixedPrefetchPolicy) UpdatePattern(off, sz int) {
	PREFETCHLOG(fmt.Sprintf("pattern record: %d, %d", off, sz))
}

// return 0 if not suggesting prefetch
func (p *FixedPrefetchPolicy) GetPrefetchSize() int {
	return p.prefetchSize
}

type bufInfo struct {
	curOff int
	buf    []byte
}

func (i *bufInfo) in(off, sz int) bool {
	return off >= i.curOff && off+sz <= i.curOff+len(i.buf)
}

// should be called after in() check
func (i *bufInfo) get(off, sz int) []byte {
	startOff := off - i.curOff
	return i.buf[startOff : startOff+sz]
}

// assumes off is inside buf
func (i *bufInfo) getTillEnd(off int) []byte {
	startOff := off - i.curOff
	return i.buf[startOff:]
}

func (i *bufInfo) update(off int, buf []byte) {
	i.curOff = off
	i.buf = buf
}

func (i *bufInfo) getNextOff() int {
	return i.curOff + len(i.buf)
}

func (i *bufInfo) clear() {
	i.buf = i.buf[:0]
}

type SyncPrefetchBuffer struct {
	src    PrefetchSource
	srclen int
	policy PrefetchPolicy
	buf    bufInfo
}

func NewSyncPrefetchBuffer(src PrefetchSource, srclen int, policy PrefetchPolicy) PrefetchBuffer {
	return &SyncPrefetchBuffer{
		src:    src,
		srclen: srclen,
		policy: policy,
	}
}

func (p *SyncPrefetchBuffer) UpdateAccess(off, sz int) {
	p.policy.UpdatePattern(off, sz)
}

func (p *SyncPrefetchBuffer) Bytes(off, sz int) ([]byte, error) {
	p.UpdateAccess(off, sz)
	if off+sz > p.srclen {
		PREFETCHLOG(fmt.Sprintf("exceeds %d: %d, %d", p.srclen, off, sz))
		return nil, errInvalidPrefetchRange
	}
	// fast path: the requested range is buffered
	if p.buf.in(off, sz) {
		PREFETCHLOG(fmt.Sprintf("hit: %d, %d", off, sz))
		return p.buf.get(off, sz), nil
	}
	fetch := p.policy.GetPrefetchSize()
	// if not suggesting prefetch, return
	if fetch == 0 {
		return nil, nil
	}
	// adjust the size to avoid reading beyong valid range
	if off+fetch > p.srclen {
		fetch = p.srclen - off
	}
	fetched, err := p.src.Bytes(off, fetch)
	if err != nil {
		return nil, err
	}
	PREFETCHLOG(fmt.Sprintf("fetch: %d, %d", off, fetch))
	p.buf.update(off, fetched)
	PREFETCHLOG(fmt.Sprintf("fetch then use: %d, %d", off, sz))
	return p.buf.get(off, sz), nil
}

type AsyncTwoPrefetchBuffer struct {
	src       PrefetchSource
	srclen    int
	policy    PrefetchPolicy
	curBuf    int
	bufs      [3]bufInfo
	nextfetch sync.WaitGroup
}

func NewAsyncTwoPrefetchBuffer(src PrefetchSource, srclen int, policy PrefetchPolicy) PrefetchBuffer {
	return &AsyncTwoPrefetchBuffer{
		src:    src,
		srclen: srclen,
		policy: policy,
		curBuf: 0,
	}
}

// only called by Byte()
func (p *AsyncTwoPrefetchBuffer) scheduleNextFetch(bufIdx, off, fetch int) {
	if off > p.srclen || fetch == 0 {
		// at the end, noop
		return
	}
	if off+fetch > p.srclen {
		fetch = p.srclen - off
	}
	p.nextfetch.Add(1)
	go func() {
		defer p.nextfetch.Done()
		fetched, err := p.src.Bytes(off, fetch)
		if err == nil {
			p.bufs[bufIdx].update(off, fetched)
		}
		PREFETCHLOG(fmt.Sprintf("async fetched: %d, %d", off, fetch))
	}()
}

func (p *AsyncTwoPrefetchBuffer) UpdateAccess(off, sz int) {
	p.policy.UpdatePattern(off, sz)
}

// only called by Bytes, we already know that the requested range is not in current buffer.
// does not modify bufs[0] and bufs[1], only bufs[2]
func (p *AsyncTwoPrefetchBuffer) overlapWithTwoBufs(curBuf, off, sz int) bool {
	nextBuf := curBuf ^ 1
	if len(p.bufs[curBuf].buf) == 0 || len(p.bufs[nextBuf].buf) == 0 {
		return false
	}
	if !p.bufs[curBuf].in(off, 1) {
		// the first byte in current buffer
		return false
	}
	if !p.bufs[nextBuf].in(off+sz-1, 1) {
		// the last byte in next buffer
		return false
	}
	return p.bufs[curBuf].getNextOff() == p.bufs[nextBuf].curOff
}

// only called in Bytes after overlapWithTwoBufs passes.
func (p *AsyncTwoPrefetchBuffer) handleOverlapWithTwoBufs(curBuf, off, sz int) {
	nextBuf := curBuf ^ 1
	if cap(p.bufs[2].buf) < sz {
		// enlarge the buffer
		p.bufs[2].update(off, make([]byte, 0, sz))
	} else {
		// clear
		p.bufs[2].clear()
	}
	curPart := p.bufs[curBuf].getTillEnd(off)
	p.bufs[2].buf = append(p.bufs[2].buf, curPart...)
	p.bufs[2].buf = append(p.bufs[2].buf, p.bufs[nextBuf].buf[:sz-len(curPart)]...)
}

func adjustFetchSize(off, intended, totalLen int) int {
	if off > totalLen {
		return 0
	} else if off+intended <= totalLen {
		return intended
	} else {
		return totalLen - off
	}
}

func (p *AsyncTwoPrefetchBuffer) Bytes(off, sz int) ([]byte, error) {
	p.UpdateAccess(off, sz)
	if off+sz > p.srclen {
		PREFETCHLOG(fmt.Sprintf("exceeds %d: %d, %d", p.srclen, off, sz))
		return nil, errInvalidPrefetchRange
	}
	// fast path: served by the current buffer
	if p.bufs[p.curBuf].in(off, sz) {
		PREFETCHLOG(fmt.Sprintf("case 1 curr: %d, %d", off, sz))
		return p.bufs[p.curBuf].get(off, sz), nil
	}
	// case 2: served by the next buffer
	p.nextfetch.Wait()
	if p.bufs[p.curBuf^1].in(off, sz) {
		curBuf := p.curBuf ^ 1
		// switching buffer, schedule async fetch
		p.scheduleNextFetch(p.curBuf, p.bufs[curBuf].getNextOff(), p.policy.GetPrefetchSize())
		p.curBuf = curBuf
		PREFETCHLOG(fmt.Sprintf("case 2 next: %d, %d", off, sz))
		return p.bufs[curBuf].get(off, sz), nil
	}
	// case 3: not in both buffer, not suggesting prefetch: return
	fetch := p.policy.GetPrefetchSize()
	if fetch == 0 {
		PREFETCHLOG(fmt.Sprintf("case 3 fall back: %d, %d", off, sz))
		return nil, nil
	}
	// case 4: overlap the two buffers
	if p.overlapWithTwoBufs(p.curBuf, off, sz) {
		p.handleOverlapWithTwoBufs(p.curBuf, off, sz)
		// switch the current buffer
		curBuf := p.curBuf ^ 1
		// schedule fetch on the next
		fetchOff := p.bufs[curBuf].getNextOff()
		p.scheduleNextFetch(curBuf^1, fetchOff, adjustFetchSize(fetchOff, fetch, p.srclen))
		p.curBuf = curBuf
		PREFETCHLOG(fmt.Sprintf("case 4 stitch: %d, %d", off, sz))
		return p.bufs[2].buf, nil
	}

	// case 5: not in both buffer, suggesting prefetch
	fetch = adjustFetchSize(off, fetch, p.srclen)
	asyncfetch := adjustFetchSize(off+fetch, fetch, p.srclen)
	// point the curBuf on which we do sync fetch
	curBuf := p.curBuf
	// schedule async fetch to the next buffer
	p.scheduleNextFetch(curBuf^1, off+fetch, asyncfetch)
	// sync fetch in this buffer
	buf, err := p.src.Bytes(off, fetch)
	if err != nil {
		return nil, err
	}
	PREFETCHLOG(fmt.Sprintf("case 5 sync fetch: %d, %d", off, fetch))
	p.bufs[curBuf].update(off, buf)
	PREFETCHLOG(fmt.Sprintf("case 5 sync serve: %d, %d", off, sz))
	return p.bufs[curBuf].get(off, sz), nil
}

// factory method
func BuildPrefetchBuffer(opts PrefetchOptions, src PrefetchSource, srclen int) PrefetchBuffer {
	switch opts.Type {
	case NoPrefetch:
		return nil
	case SyncPrefetch:
		return NewSyncPrefetchBuffer(src, srclen, NewFixedPrefetchPolicy(opts.PrefetchSize))
	case AsyncPrefetch:
		return NewAsyncTwoPrefetchBuffer(src, srclen, NewFixedPrefetchPolicy(opts.PrefetchSize))
	default:
		log.Fatal("unknown prefetch option")
		return nil
	}
}
