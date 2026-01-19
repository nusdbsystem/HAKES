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

package io

import (
	"context"
	"log"
	"sync"
	"sync/atomic"
)

type SstCacheMonitor interface {
	GetCapacity() uint64
	Add(charge int) bool
	Drop(charge int)
}

var _ SstCacheMonitor = (*sstCacheMonitor)(nil)

type sstCacheMonitor struct {
	used     atomic.Uint64 // in B
	capacity uint64        // in B
}

func NewSstCacheMonitor(capacity uint64) SstCacheMonitor {
	return &sstCacheMonitor{capacity: capacity}
}

func (scm *sstCacheMonitor) GetCapacity() uint64 {
	return scm.capacity
}

func (scm *sstCacheMonitor) Add(charge int) bool {
	for {
		used := scm.used.Load()
		target := used + uint64(charge)
		if target > scm.capacity {
			return false
		}
		if scm.used.CompareAndSwap(used, target) {
			return true
		}
	}
}

func (scm *sstCacheMonitor) Drop(charge int) {
	used := scm.used.Load()
	if used < uint64(charge) {
		return
	}
	scm.used.Add(-uint64(charge))
}

type SstCache interface {
	Init() error
	Reserve(charge int) bool
	Release(charge int)
	Add(path string, charge int) CSSF
	Get(path string) (CSSF, error)
	Drop(path string, charge int)
	Close()
}

var _ SstCache = (*LocalSstCache)(nil)
var _ SstCache = (*RemoteSstCache)(nil)

type SstCacheCli interface {
	SstCacheReserve(ctx context.Context, charge int) bool
	SstCacheRelease(ctx context.Context, charge int)
	SstCacheAdd(ctx context.Context, path string, charge int) error
	SstCacheDrop(ctx context.Context, path string, charge int)
	Close()
}

type LocalSstCache struct {
	cssCli  CSSCli
	monitor SstCacheMonitor
}

func NewLocalSstCache(capacity uint64, cssCli CSSCli) SstCache {
	return &LocalSstCache{
		cssCli:  cssCli,
		monitor: NewSstCacheMonitor(capacity),
	}
}

func (sc *LocalSstCache) Init() error {
	// clearing the cache
	var hasError atomic.Bool
	fl, err := sc.cssCli.List()
	if err == nil {
		// parallel delete all files
		var wg sync.WaitGroup
		for _, f := range fl {
			wg.Add(1)
			go func(f string) {
				if de := sc.cssCli.Delete(f); de != nil {
					if hasError.CompareAndSwap(false, true) {
						err = de // assign the error for display
					}
				}
				wg.Done()
			}(f)
		}
		wg.Wait()
	}

	if err != nil {
		// given empty capacity, thus forbid adding to this cache
		log.Printf("Warning: failed to clear the SSTCache, will not admit any contents: %v", err)
		sc.monitor = NewSstCacheMonitor(0)
		return err
	}
	log.Printf("SSTCache initialized with capacity: %d", sc.monitor.GetCapacity())
	return nil
}

func (sc *LocalSstCache) Reserve(charge int) bool {
	return sc.monitor.Add(charge)
}

func (sc *LocalSstCache) Release(charge int) {
	sc.monitor.Drop(charge)
}

func (sc *LocalSstCache) Add(path string, charge int) CSSF {
	if !sc.monitor.Add(charge) {
		return nil
	}

	if f, err := sc.cssCli.OpenNewFile(path, charge); err != nil {
		sc.monitor.Drop(charge)
		return nil
	} else {
		return f
	}
}

func (sc *LocalSstCache) Get(path string) (CSSF, error) {
	return sc.cssCli.OpenFile(path, 0)
}

// pass 0 if need the sst cache to do one IO to figure out the space it takes
func (sc *LocalSstCache) Drop(path string, charge int) {
	c := charge
	if charge == 0 {
		if f, err := sc.cssCli.OpenFile(path, 0); err != nil {
			return
		} else {
			c = int(f.Size())
		}
	}
	if sc.cssCli.Delete(path) == nil {
		sc.monitor.Drop(c)
	} else {
		log.Printf("Local cache failed to delete %v", path)
	}
}

func (sc *LocalSstCache) Close() {
	sc.cssCli.Disconnect()
}

type RemoteSstCache struct {
	cssCli CSSCli
	scCli  SstCacheCli
}

// cssCli should have connected to the cache directory.
// scCli should have connected to a SstCache.
func NewRemoteSstCache(cssCli CSSCli, scCli SstCacheCli) SstCache {
	return &RemoteSstCache{
		cssCli: cssCli,
		scCli:  scCli,
	}
}

func (sc *RemoteSstCache) Init() error {
	return nil
}

func (sc *RemoteSstCache) Reserve(charge int) bool {
	return sc.scCli.SstCacheReserve(context.TODO(), charge)
}

func (sc *RemoteSstCache) Release(charge int) {
	sc.scCli.SstCacheRelease(context.TODO(), charge)
}

func (sc *RemoteSstCache) Add(path string, charge int) CSSF {
	// reserve space from SstCache first
	if !sc.scCli.SstCacheReserve(context.TODO(), charge) {
		return nil
	}
	if f, err := sc.cssCli.OpenNewFile(path, charge); err != nil {
		sc.scCli.SstCacheDrop(context.TODO(), "", charge) // empty path since we have not added the content to cache
		return nil
	} else {
		return f
	}
}

// get should only call on data already cached
// for non-existent data return error
func (sc *RemoteSstCache) Get(path string) (CSSF, error) {
	if f, err := sc.cssCli.OpenFile(path, 0); err != nil {
		return nil, err
	} else {
		return f, nil
	}
}

func (sc *RemoteSstCache) Drop(path string, charge int) {
	// async drop
	go func() {
		sc.cssCli.Delete(path)
		sc.scCli.SstCacheRelease(context.TODO(), charge)
	}()
}

func (sc *RemoteSstCache) Close() {
	sc.cssCli.Disconnect()
	sc.scCli.Close()
}
