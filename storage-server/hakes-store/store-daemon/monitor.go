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
	"fmt"
	"log"
	"sync/atomic"
	"time"

	"github.com/mackerelio/go-osstat/cpu"
	"github.com/mackerelio/go-osstat/memory"
	"github.com/mackerelio/go-osstat/network"
)

type monitor struct {
	r            resources
	cpuTotal     uint64 // cpu measure unit (jiffy)
	cpuCount     int    // core count
	memTotal     uint64 // RAM in KB
	netName      string // network interface name
	netBandwidth uint64 // bandwidth in KB/s
	cpuUnAvail   atomic.Bool
	memUnAvail   atomic.Bool
	netUnAvail   atomic.Bool
}

func newMonitor(netName string, netBandwidth uint64) *monitor {
	return &monitor{
		netName:      netName,
		netBandwidth: netBandwidth,
	}
}

func (m *monitor) init() bool {
	if mem, err := memory.Get(); err != nil {
		return false
	} else {
		log.Printf("mem total: %d used: %d", mem.Total, mem.Used)
		m.memTotal = mem.Total >> 20
		m.r.updateMemStats(mem)
		m.memUnAvail.Store(false)
	}
	if cpu, err := cpu.Get(); err != nil {
		return false
	} else {
		log.Printf("cpu count: %d", cpu.CPUCount)
		m.cpuTotal = cpu.Total
		m.cpuCount = cpu.CPUCount
		m.r.updateCPUStats(cpu)
		m.cpuUnAvail.Store(false)
	}
	netUpdate := false
	if net, err := network.Get(); err == nil {
		for _, n := range net {
			if n.Name == m.netName {
				m.r.updateNetInfo(&n)
				netUpdate = true
				m.netUnAvail.Store(false)
				break
			}
		}
	}
	if !netUpdate {
		m.netUnAvail.Store(false)
		return false
	}
	return true
}

func (m *monitor) update() {
	if mem, err := memory.Get(); err != nil {
		m.memUnAvail.Store(true)
	} else {
		m.r.updateMemStats(mem)
		m.memUnAvail.Store(false)
	}
	if cpu, err := cpu.Get(); err != nil {
		m.cpuUnAvail.Store(true)
	} else {
		m.r.updateCPUStats(cpu)
		m.cpuUnAvail.Store(false)
	}
	netUpdate := false
	if net, err := network.Get(); err == nil {
		for _, n := range net {
			if n.Name == m.netName {
				m.r.updateNetInfo(&n)
				netUpdate = true
				m.netUnAvail.Store(false)
				break
			}
		}
	}
	if !netUpdate {
		m.netUnAvail.Store(false)
	}
}

func (m *monitor) start() bool {
	if !m.init() {
		return false
	}

	// should not be too small, otherwise not enough count to calculate cpu idle fraction
	ticker := time.NewTicker(100 * time.Millisecond)

	go func() {
		defer ticker.Stop()
		for {
			<-ticker.C
			m.update()
		}
	}()
	return true
}

// report in core count
func (m *monitor) getCPUInfo() (uint64, float32) {
	if m.cpuUnAvail.Load() {
		return 0, 0
	}
	return uint64(m.cpuCount), m.r.getCPUInfo()
}

// report in kb
func (m *monitor) getMemInfo() (uint64, float32) {
	if m.memUnAvail.Load() {
		return 0, 0
	}
	return m.memTotal, float32(m.r.getMemInfo()) / float32(m.memTotal)
}

// only reports the kb/s
func (m *monitor) getNetInfo() (uint64, float32) {
	if m.netUnAvail.Load() {
		return m.netBandwidth, 0
	}
	return m.netBandwidth, 1 - (m.r.getNetInfo() / float32(m.netBandwidth))
}

func (m *monitor) getAvailPercent() (float32, float32, float32) {
	return float32(m.r.getCPUInfo()), float32(m.r.getMemInfo()) / float32(m.memTotal), 1 - (m.r.getNetInfo() / float32(m.netBandwidth))
}

func (m *monitor) info() string {
	cpuTotal, cpuAvail := m.getCPUInfo()
	memTotal, memAvail := m.getMemInfo()
	netTotal, netAvail := m.getNetInfo()
	return fmt.Sprintf("cpu (%d core) %.2f; mem %.2f(%d); net %.2f(%d)", cpuTotal, cpuAvail, memAvail, memTotal, netAvail, netTotal)
}
