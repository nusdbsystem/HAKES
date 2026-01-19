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

package storedaemon

import (
	"sync"
	"sync/atomic"
	"time"

	"github.com/mackerelio/go-osstat/cpu"
	"github.com/mackerelio/go-osstat/memory"
	"github.com/mackerelio/go-osstat/network"
)

type cpuInfo struct {
	lastTotal uint64
	lastIdle  uint64
	lastAvail float32
	mu        sync.RWMutex
}

func (ci *cpuInfo) update(stats *cpu.Stats) {
	ci.mu.Lock()
	ci.lastAvail = float32(stats.Idle-ci.lastIdle) / float32(stats.Total-ci.lastTotal)
	ci.lastIdle = stats.Idle
	ci.lastTotal = stats.Total
	ci.mu.Unlock()
}

// intermediate data owned by the server to update and report
type networkInfo struct {
	estUtil     float32
	lastRxbytes uint64
	lastTxBytes uint64
	lastTs      int64
	mu          sync.RWMutex
}

func (ni *networkInfo) update(stats *network.Stats) {
	ni.mu.Lock()
	curTs := time.Now().UnixMicro()
	ni.estUtil = float32((stats.RxBytes-ni.lastRxbytes)+(stats.TxBytes-ni.lastTxBytes)) / float32(curTs-ni.lastTs)
	ni.lastRxbytes = stats.RxBytes
	ni.lastTxBytes = stats.TxBytes
	ni.lastTs = curTs
	ni.mu.Unlock()
}

type resources struct {
	cpuInfo  cpuInfo
	memAvail atomic.Uint64
	netInfo  networkInfo
}

func (r *resources) updateCPUStats(stats *cpu.Stats) {
	r.cpuInfo.update(stats)
}

func (r *resources) updateMemStats(stats *memory.Stats) {
	r.memAvail.Store(stats.Total - stats.Used)
}

func (r *resources) updateNetInfo(stats *network.Stats) {
	r.netInfo.update(stats)
}

func (r *resources) getCPUInfo() float32 {
	r.cpuInfo.mu.RLock()
	defer r.cpuInfo.mu.RUnlock()
	return r.cpuInfo.lastAvail
}

func (r *resources) getMemInfo() uint64 {
	return r.memAvail.Load() >> 20
}

// only reports the kb/s
func (r *resources) getNetInfo() float32 {
	r.netInfo.mu.RLock()
	defer r.netInfo.mu.RUnlock()
	return r.netInfo.estUtil / 1024
}
