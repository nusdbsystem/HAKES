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

package hakesstoreutil

import (
	"errors"
	"fmt"
	"log"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	ReplicaNodeName = "replica"
)

var (
	ErrExhaustedRetry = errors.New("exhaused retry")
	ErrNoConn         = errors.New("no connection")
	ErrConnNotReady   = errors.New("failed to get a ready connection")
)

func GetReplicaUidStr(uid int) string {
	return fmt.Sprintf("%v%d", ReplicaNodeName, uid)
}

func ExtractUidFromStr(in string) (int, error) {
	return strconv.Atoi(in[len(ReplicaNodeName):])
}

type ReplicaGroupView map[string]struct{}

func (r ReplicaGroupView) Contain(nodeAddr string) bool {
	_, ok := r[nodeAddr]
	return ok
}

func (r ReplicaGroupView) CheckEqual(o ReplicaGroupView) bool {
	if len(r) != len(o) {
		return false
	}
	for k := range r {
		if _, ok := o[k]; !ok {
			return false
		}
	}
	return true
}

type ReplicaRSConfig struct {
	ModeChange   atomic.Int32
	ModeMu       sync.RWMutex
	Isleader     atomic.Bool // mode
	ReplicaCap   int         // num of replicas other than the current one
	PersistCount int         // num of reply needed for persistence
	Rp           RetryPolicy // retry policy for operations
}

type Pauser interface {
	Pause() bool
	Reset()
}

// retry policy provides a
type RetryPolicy interface {
	NewPauser() Pauser
}

type FixedIntervalPaucer struct {
	ms    time.Duration
	limit int
	count int
}

func (p *FixedIntervalPaucer) Pause() bool {
	if p.count >= p.limit {
		return false
	}
	time.Sleep(p.ms)
	p.count++
	return true
}

func (p *FixedIntervalPaucer) Reset() {
	p.count = 0
}

type FixedIntervalRetry struct {
	ms    int
	limit int
}

func NewFixedIntervalRetry(ms int, retryCount int) *FixedIntervalRetry {
	return &FixedIntervalRetry{ms, retryCount}
}

func (rp *FixedIntervalRetry) NewPauser() Pauser {
	return &FixedIntervalPaucer{time.Duration(rp.ms) * time.Millisecond, rp.limit, 0}
}

/*
 * ReplicaConnPool manages a pool of connection to the replicas in a given replica group.
 * It provides a safe mechnism to retrieve the connection references to handle requests
 * And it allows safe upgrade given a new view of the replica group
 * Note that we allow more replica to be added to zookeeper as znodes but not included in the active view. So they will be launched but their rcp only becomes active once they are part of the active view.
 */

func RCPLOG(msg string) {
	log.Printf("%v [RCP]: %v\n", time.Now().UnixNano(), msg)
}

// compared to CliConnHandler, CliConnHolder does not handle reconnection/redirection. Instead it provides Conn safely such that it can be updated by another goroutine.
type CliConnHolder struct {
	connReset atomic.Int32 // there won't be more than one update in queue. bool should be sufficient
	resetMu   sync.RWMutex
	ordered   sync.Mutex // an ordered mutex for requests that require one at a time delivery
	conn      *grpc.ClientConn
}

func (c *CliConnHolder) ReserveOrdered() {
	c.ordered.Lock()
}

func (c *CliConnHolder) ReleaseOrdered() {
	c.ordered.Unlock()
}

var (
	ErrConnReconnect = errors.New("grpc connection reconnecting")
)

// get connection returns a connection reference, when it is not undergoing reconnection
// returns ErrConnReconnect and the caller may retry later.
func (c *CliConnHolder) GetConn() (*grpc.ClientConn, error) {
	if c.connReset.Load() > 1 {
		time.Sleep(time.Second) // by default wait 1 second first
		if c.connReset.Load() > 1 {
			return nil, ErrConnReconnect
		}
	}
	// return a reference, making the caller a shared pointer holder
	c.resetMu.RLock()
	defer c.resetMu.RUnlock()
	if c.conn == nil {
		time.Sleep(1 * time.Second)
		return nil, ErrNoConn
	}
	return c.conn, nil
}

type ReplicaConnPool struct {
	// bootstrap
	active     atomic.Bool     // can only be turned on but not off. (via Start)
	boostrapCh chan<- struct{} // channel is closed once boostrap is done

	// setting
	ownAddr    string
	replicaCap int // number of replica
	// state
	metas  []string   // replica address of the view in use
	metaMu sync.Mutex // mutex to protect against replicaIds

	// connection management
	conns   []*CliConnHolder // after setup it is no longer changed
	watchCh chan ReplicaGroupView
}

func NewReplicaConnPool(ownAddr string, capacity int) *ReplicaConnPool {
	return &ReplicaConnPool{
		boostrapCh: make(chan struct{}),
		ownAddr:    ownAddr,
		replicaCap: capacity,
		metas:      make([]string, capacity),
		conns:      make([]*CliConnHolder, capacity),
		watchCh:    make(chan ReplicaGroupView),
	}
}

func (rcp *ReplicaConnPool) IsActive() bool {
	return rcp.active.Load()
}

// not thread-safe, should only be used by updateWatcher
func (rcp *ReplicaConnPool) updateConns(view ReplicaGroupView) {
	RCPLOG(fmt.Sprintf("update connection with view %v", view))
	// this replica is not part of the active replica group yet.
	if !view.Contain(rcp.ownAddr) {
		return
	}

	// remove the current nodes entry
	delete(view, rcp.ownAddr)

	// this lock only locks the replica meta
	// only accessed by update watcher, so actually the lock is not needed
	rcp.metaMu.Lock()
	defer rcp.metaMu.Unlock()

	// skip those connection that are still in view
	// set reset signal for those not in current view to change
	for i, meta := range rcp.metas {
		if view.Contain(meta) {
			delete(view, meta)
			continue
		}
		rcp.conns[i].connReset.Add(1)
	}
	RCPLOG(fmt.Sprintf("current conns %v", rcp.metas))

	// prepare the list of new candidate metas
	cands := make([]string, 0, len(view))
	for addr := range view {
		cands = append(cands, addr)
	}
	RCPLOG(fmt.Sprintf("replacement candidates %v", cands))

	// reset problematic connections
	nextCandIdx := 0
	for i, h := range rcp.conns {
		// preserved connection
		if h.connReset.Load() == 0 {
			continue
		}
		reset := false
		// there is a cand connection as replacement
		h.resetMu.Lock()
		// close the connection
		if h.conn != nil {
			h.conn.Close()
		}
		for nextCandIdx < len(cands) {
			curCand := cands[nextCandIdx]
			nextCandIdx++
			conn, err := grpc.Dial(curCand, grpc.WithTransportCredentials(insecure.NewCredentials()))
			// if we cannot connect to this one, skip to get the next one
			if err != nil {
				RCPLOG(fmt.Sprintf("failed to connect to replica at %v", curCand))
				continue
			}
			rcp.metas[i] = curCand
			h.conn = conn
			reset = true
			RCPLOG(fmt.Sprintf("conn-%d reset with replica at %v", i, curCand))
			break
		}
		h.connReset.Add(-1)
		if !reset {
			// reset the connection to nil, we have no candidate for replacement
			rcp.metas[i] = ""
			h.conn = nil
		}
		h.resetMu.Unlock()
	}

	// if the rcp is never boostraped, indicate bootstrapping done
	if !rcp.active.Load() {
		rcp.active.Store(true)
		close(rcp.boostrapCh) // unblock all goroutines waiting for active signal.
	}
}

// this replica's own uid is passed such that it is not counted as one replica connection
func (rcp *ReplicaConnPool) Start() {
	// initialize the structure first
	for i := 0; i < rcp.replicaCap; i++ {
		rcp.metas[i] = ""
		rcp.conns[i] = &CliConnHolder{}
	}
	// launch the view update watcher; there should be only one
	updateWatcher := func() {
		for {
			view := <-rcp.watchCh
			rcp.updateConns(view)
		}
	}

	go updateWatcher()
}

func (rcp *ReplicaConnPool) Stop() {
	// the stop logic terminates all the connections
	// we do this by sending an empty view to the update watcher
	// this would effectively set all connection to nil.
	rcp.watchCh <- ReplicaGroupView{}
}

func (rcp *ReplicaConnPool) GetConns() []*CliConnHolder {
	return rcp.conns
}

func (rcp *ReplicaConnPool) Update(view ReplicaGroupView) {
	rcp.watchCh <- view
}
