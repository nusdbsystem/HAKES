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
	"context"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	pb "hakes-store/grpc-api"
	dcli "hakes-store/store-daemon-cli"

	"github.com/dgraph-io/ristretto/z"
)

type workerState struct {
	cpuAvail float32 // core fraction
	memAvail float32 // MB
	netAvail float32
}

// pick function for a given resource state
func pick(s workerState, reserved float32) bool {
	log.Printf("reserved: %f, %v", reserved, s)
	return s.cpuAvail-reserved > 0.2 && s.memAvail-reserved > 0.2
}

type worker struct {
	started atomic.Bool
	state   workerState
	mu      sync.RWMutex
	conn    *dcli.StoreDaemonCli
}

func (w *worker) String() string {
	return fmt.Sprintf("worker: conn: %v", w.conn)
}

func newWorker(addr string) *worker {
	return &worker{conn: dcli.NewStoreDaemonCli(addr)}
}

// periodically update the state of worker
func (w *worker) stateUpdate(interval int, lc *z.Closer) {
	defer lc.Done()
	t := time.NewTicker(time.Duration(interval) * time.Millisecond)
	for {
		select {
		case <-t.C:
			ca, ma, na := w.conn.GetStats(context.TODO(), "")
			w.mu.Lock()
			w.state.cpuAvail, w.state.memAvail, w.state.netAvail = ca, ma, na
			w.mu.Unlock()
		case <-lc.HasBeenClosed():
			return
		}
	}
}

func (w *worker) getState() workerState {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.state
}

func (w *worker) start(lc *z.Closer) {
	if w.started.Load() {
		return
	}
	for {
		if w.conn.Connect() != nil {
			log.Printf("failed to connect to store-daemon worker (%v)", w.conn)
		} else {
			log.Printf("connected to store-daemon worker (%v)", w.conn)
			break
		}
		time.Sleep(time.Second)
	}

	lc.AddRunning(1)
	go w.stateUpdate(10, lc)
	w.started.Store(true)
}

func (w *worker) scheduleJob(ctx context.Context, req *pb.JobRequest) (bool, []byte) {
	if !w.started.Load() {
		return false, nil
	}
	return w.conn.ScheduleJob(ctx, pb.JobType_SCHEDULED, req.JobId, req.Payload)
}

func (w *worker) close() {
	if w.started.Load() {
		w.conn.Close()
	}
}

type JobScheduler struct {
	workers  []*worker
	mu       sync.RWMutex // guard the update of worker
	started  atomic.Bool
	lc       *z.Closer
	lastUsed atomic.Uint32
}

func NewJobScheduler(peers []string) *JobScheduler {
	workers := make([]*worker, len(peers))
	for i, addr := range peers {
		workers[i] = newWorker(addr)
	}
	return &JobScheduler{workers: workers, lc: z.NewCloser(0)}
}

func (sc *JobScheduler) start() {
	if sc.started.Load() {
		return
	}
	log.Printf("launching job scheduler to connect to peers: %v", sc.workers)
	for _, w := range sc.workers {
		w.start(sc.lc)
	}
	sc.started.Store(true)
}

// send a job request via the scheduler. scheduler returns
// * whether the job is scheduled
// * reply of the job if scheduled
// * error
func (sc *JobScheduler) schedule(req *pb.JobRequest) (bool, *pb.JobReply, error) {
	// avoid view change
	sc.mu.RLock()
	view := make([]*worker, len(sc.workers))
	copy(view, sc.workers)
	sc.mu.RUnlock()
	retried := false
	for {
		w := view[sc.lastUsed.Add(1)%uint32(len(view))]
		if pick(w.getState(), 0) {
			success, replyPayload := w.scheduleJob(context.TODO(), req)
			if success {
				return true, &pb.JobReply{Success: true, Payload: replyPayload}, nil
			}
			if retried {
				log.Printf("retried one more workers but no success, Job (%v) will fall back to local exec", req.JobId)
				return false, nil, nil
			}
			retried = true
		}
	}
}

func (sc *JobScheduler) close() {
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	sc.lc.SignalAndWait()          // terminate the stats listeners
	for _, w := range sc.workers { // close all connections
		w.close()
	}
}
