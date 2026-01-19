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

package hakesstore

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"log"
	"sync"
	"sync/atomic"
	"time"

	util "hakes-store/hakes-store-util"
	kvio "hakes-store/hakes-store/io"

	"golang.org/x/net/context"
	"google.golang.org/grpc/status"

	pb "hakes-store/grpc-api"
)

func RLHLOG(msg string) {
	log.Printf("%v [RLH]: %v\n", time.Now().UnixNano(), msg)
}

var _ kvio.DLog = (*RDLog)(nil)
var _ kvio.DLogHandler = (*RDLogHandler)(nil)

// RDLog implements DLog API to replicate the log across multiple replica for durability
type RDLog struct {
	name           string // name of the log
	buf            []byte // buffer to hold the log data
	cap            int
	tailOffset     atomic.Uint32 // single writer multi-reader
	lastSyncOffset uint32

	// config
	rscfg *util.ReplicaRSConfig

	// external communication
	rcp *util.ReplicaConnPool // if leader, connections to followers. nil for follower.
}

func NewRDLog(name string, maxSz int, rscfg *util.ReplicaRSConfig, rcp *util.ReplicaConnPool) *RDLog {
	return &RDLog{
		name:           name,
		buf:            make([]byte, maxSz), // allocate the size upfront, data will be copied in.
		cap:            maxSz,
		lastSyncOffset: 0,
		rscfg:          rscfg, // take reference from replica server
		rcp:            rcp,   // take reference from replica server
	}
}

func (l *RDLog) Name() string {
	return l.name
}

func (c *RDLog) replicatedAppend(name string, offset uint32, data []byte) bool {
	req := &pb.InAppendLogRequest{LogName: name, Offset: offset, Data: data}
	dummyfailedReply := &pb.InAppendLogReply{Success: false, HaveReq: false}
	desc := fmt.Sprintf("append at %v, offset %v", name, offset)

	completeChan := make(chan *pb.InAppendLogReply, len(c.rcp.GetConns()))
	replyChan := make(chan *pb.InAppendLogReply, len(c.rcp.GetConns()))

	collector := func() {
		success := 0
		var ret *pb.InAppendLogReply
		for i := 0; i < c.rscfg.ReplicaCap; i++ {
			reply, ok := <-replyChan
			if !ok {
				RLHLOG("unexpected closure of reply channel")
				close(completeChan)
				return
			}
			if reply == nil || (!reply.Success) {
				continue
			}
			success++
			if ret == nil {
				ret = reply
			}
			if success >= c.rscfg.PersistCount {
				completeChan <- ret
			}
		}
		if success < c.rscfg.PersistCount {
			RLHLOG("not enough successful replies")
			completeChan <- nil
		}
	}

	dispatcher := func(cch *util.CliConnHolder, rp util.RetryPolicy) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		pauser := rp.NewPauser()
		cch.ReserveOrdered()
		defer cch.ReleaseOrdered()
		for {
			conn, _ := cch.GetConn()
			if conn != nil {
				cli := pb.NewHakesStoreKVRClient(conn)
				r, err := cli.InAppendLog(ctx, req)
				// send more data over if a failed reply contain request for missing data in between
				for {
					if err != nil {
						if status, ok := status.FromError(err); ok {
							if status.Message() == ErrRDLogOpSkipped.Error() {
								replyChan <- nil
								return
							}
						}
						break // break to the outerloop for retry
					}
					if r.Success {
						replyChan <- r
						return
					}
					if !r.HaveReq {
						replyChan <- r
						return
					}
					dr := r.Req
					if dr.LogName != c.name {
						break // break as failed request
					}
					if dr.EndOff-dr.StartOff > 4<<20 {
						req = &pb.InAppendLogRequest{LogName: c.name, Offset: dr.StartOff, Data: c.buf[dr.StartOff : dr.StartOff+2<<20]}
					} else {
						req = &pb.InAppendLogRequest{LogName: c.name, Offset: dr.StartOff, Data: c.buf[dr.StartOff:dr.EndOff]}
					}
					r, err = cli.InAppendLog(ctx, req)
				}
				RLHLOG(fmt.Sprintf("%v request received error: %v", desc, err))
			} else {
				RLHLOG("got empty connection, entering retry to wait for replica")
			}
			if !pauser.Pause() {
				RLHLOG(fmt.Sprintf("exhausted retries in %v", desc))
				replyChan <- dummyfailedReply
				return
			}
			RLHLOG(fmt.Sprintf("retry on %v", desc))
		}
	}

	for _, cch := range c.rcp.GetConns() {
		go dispatcher(cch, c.rscfg.Rp)
	}
	go collector()
	pickedReply := <-completeChan
	if pickedReply == nil || !pickedReply.Success {
		return false
	}
	return true
}

// propagated to followers
// leader will have sequential append by HAKESKV
// follower can have multiple append, it is guarded one level above at the client interface impl to have sequential append
func (l *RDLog) Append(b []byte) (int, error) {
	if !l.rscfg.Isleader.Load() {
		// follower should never call this function
		RLHLOG("follower calling append log (possibly an error)")
		return 0, nil
	}
	curTail := l.tailOffset.Load()
	if len(b) > l.cap-int(curTail) {
		RLHLOG(fmt.Sprintf("input: %d to remaining space: cap: %d, tail: %v", len(b), l.cap, curTail))
		return 0, ErrRDLogNoSpace
	}
	written := copy(l.buf[curTail:], b)
	l.tailOffset.Store(curTail + uint32(written))

	return written, nil
}

func (l *RDLog) Close() error {
	return nil
}

func (l *RDLog) Size() uint32 {
	return l.tailOffset.Load()
}

func (l *RDLog) Sync() error {
	curTail := l.tailOffset.Load()
	if !l.replicatedAppend(l.name, curTail, l.buf[l.lastSyncOffset:curTail]) {
		l.tailOffset.Store(curTail)
		return ErrReplicatedAppend
	}
	l.lastSyncOffset = curTail
	return nil
}

func (l *RDLog) NewReader(offset int) io.Reader {
	curTail := l.tailOffset.Load()
	if curTail < uint32(offset) {
		return nil
	}
	return bytes.NewReader(l.buf[offset:curTail])
}

var (
	ErrRDLogNotFound     = errors.New("RDLog not found")
	ErrRDLogNoSpace      = errors.New("RDLog not enough space")
	ErrRDLogOpSkipped    = errors.New("RDLog op skipped")
	ErrReplicatedAppend  = errors.New("RDLog append failed to be replicated")
	ErrReplicatedOpenNew = errors.New("RDLog reopen failed to be replicated")
)

// only setup during start and mode/view change
type RDLogHandler struct {
	// internal state
	dir                  map[string]*RDLog
	appendReadyThreshold string // for slow follower to set, they will not respond to request below this from leader.
	lastCreated          string // for manual checking purpose, the follower should always create logs with monotonic increasing log id.
	dirMu                sync.RWMutex
	// config
	busyWaitInterval int // follower operation wait time for mode change, avoid busy waiting consuming too much resources
	rscfg            *util.ReplicaRSConfig
	// external connection
	rcp *util.ReplicaConnPool
}

func NewRDLogHandler(rscfg *util.ReplicaRSConfig, rcp *util.ReplicaConnPool, busyWaitInterval int) *RDLogHandler {
	return &RDLogHandler{
		dir:              make(map[string]*RDLog),
		busyWaitInterval: busyWaitInterval,
		rscfg:            rscfg, // take reference from replica server
		rcp:              rcp,   // take reference from replica server
	}
}

func (c *RDLogHandler) GetStats() string {
	// replicated log incurs no storage kvio.
	return ""
}

func (c *RDLogHandler) setReadyThreshold(logName string) {
	c.dirMu.Lock()
	defer c.dirMu.Unlock()
	c.appendReadyThreshold = logName
}

// Connect is used for both initialization and switch mode/view
func (c *RDLogHandler) Connect(config string) error {
	// no-op
	return nil
}

// in normal operation, Disconnect is never invoked
func (c *RDLogHandler) Disconnect() error {
	// no-op
	return nil
}

// for new leaader own use (reopen log and populate memtable during leader promotion)
func (c *RDLogHandler) OpenLog(name string, maxSz int) (kvio.DLog, error) {
	c.dirMu.RLock()
	defer c.dirMu.RUnlock()
	l, ok := c.dir[name]
	if !ok {
		return nil, ErrRDLogNotFound
	}
	return l, nil
}

func (c *RDLogHandler) replicatedOpenNewLog(name string, maxSz int) bool {
	req := &pb.InOpenLogRequest{LogName: name, MaxSz: uint32(maxSz)}
	dummyfailedReply := &pb.InOpenLogReply{Success: false, Msg: "exhausted retry"}
	desc := fmt.Sprintf("open new log: %v", name)

	completeChan := make(chan *pb.InOpenLogReply, len(c.rcp.GetConns()))
	replyChan := make(chan *pb.InOpenLogReply, len(c.rcp.GetConns()))

	collector := func() {
		success := 0
		var ret *pb.InOpenLogReply
		for i := 0; i < c.rscfg.ReplicaCap; i++ {
			reply, ok := <-replyChan
			if !ok {
				RLHLOG("unexpected closure of reply channel")
				close(completeChan)
				return
			}
			if reply == nil || (!reply.Success) {
				RLHLOG(fmt.Sprintf("received one failed %v reply", desc))
				continue
			}
			RLHLOG(fmt.Sprintf("received one successful %v reply", desc))
			success++
			if ret == nil {
				ret = reply
			}
			if success >= c.rscfg.PersistCount {
				completeChan <- ret
			}
		}
		if success < c.rscfg.PersistCount {
			RLHLOG("not enough successful replies")
			completeChan <- nil
		}
		RLHLOG(fmt.Sprintf("%v received all reply", desc))
	}

	dispatcher := func(cch *util.CliConnHolder, rp util.RetryPolicy) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		pauser := rp.NewPauser()
		for {
			conn, _ := cch.GetConn()
			if conn != nil {
				RLHLOG(conn.GetState().String())
				cli := pb.NewHakesStoreKVRClient(conn)
				r, err := cli.InOpenLog(ctx, req)
				if status, ok := status.FromError(err); ok {
					if status.Message() == ErrRDLogOpSkipped.Error() {
						replyChan <- nil
						return
					}
				}
				if err == nil && r.Success {
					replyChan <- r
					return
				}
				RLHLOG(fmt.Sprintf("%v request received error: %v", desc, err))
			} else {
				RLHLOG("got empty connection, entering retry to wait for replica")
			}
			if !pauser.Pause() {
				RLHLOG(fmt.Sprintf("exhausted retries in %v", desc))
				replyChan <- dummyfailedReply
				return
			}
			RLHLOG(fmt.Sprintf("retry on %v", desc))
		}
	}

	for _, cch := range c.rcp.GetConns() {
		go dispatcher(cch, c.rscfg.Rp)
	}
	go collector()
	pickedReply := <-completeChan
	if pickedReply == nil || !pickedReply.Success {
		return false
	}
	return true
}

// propagated to followers
func (c *RDLogHandler) OpenNewLog(name string, maxSz int) (kvio.DLog, error) {
	c.dirMu.Lock()
	defer c.dirMu.Unlock()
	if l, ok := c.dir[name]; ok {
		return l, nil
	}
	c.dir[name] = NewRDLog(name, maxSz, c.rscfg, c.rcp)
	if !c.rscfg.Isleader.Load() {
		// follower should never call this function
		RLHLOG("follower calling open new log (possibly an error)")
		return nil, nil
	}
	// replicate to followers
	if !c.replicatedOpenNewLog(name, maxSz) {
		RLHLOG("error encountered when replicating open new log to followers ")
		return nil, ErrReplicatedOpenNew
	}
	c.lastCreated = name
	// HAKESKV requires the new file error for checking.
	return c.dir[name], nil
}

// local function
func (c *RDLogHandler) List() ([]string, error) {
	c.dirMu.RLock()
	defer c.dirMu.RUnlock()
	ret := make([]string, 0, len(c.dir))
	for k := range c.dir {
		ret = append(ret, k)
	}
	RLHLOG(fmt.Sprintf("returning log list: %v", ret))
	return ret, nil
}

func (c *RDLogHandler) replicatedDropLog(name string) bool {
	req := &pb.InDropLogRequest{LogName: name}
	dummyfailedReply := &pb.InDropLogReply{Success: false, Msg: "exhausted retry"}
	desc := fmt.Sprintf("drop log: %v", name)

	completeChan := make(chan *pb.InDropLogReply, len(c.rcp.GetConns()))
	replyChan := make(chan *pb.InDropLogReply, len(c.rcp.GetConns()))

	collector := func() {
		success := 0
		var ret *pb.InDropLogReply
		for i := 0; i < c.rscfg.ReplicaCap; i++ {
			reply, ok := <-replyChan
			if !ok {
				RLHLOG("unexpected closure of reply channel")
				close(completeChan)
				return
			}
			if !reply.Success {
				RLHLOG(fmt.Sprintf("received one failed %v reply", desc))
				continue
			}
			RLHLOG(fmt.Sprintf("received one successful %v reply", desc))
			success++
			if ret == nil {
				ret = reply
			}
			if success >= c.rscfg.PersistCount {
				completeChan <- ret
			}
		}
		if success < c.rscfg.PersistCount {
			RLHLOG("not enough successful replies")
			completeChan <- nil
		}
		RLHLOG(fmt.Sprintf("%v received all reply", desc))
	}

	dispatcher := func(cch *util.CliConnHolder, rp util.RetryPolicy) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		pauser := rp.NewPauser()
		for {
			conn, _ := cch.GetConn()
			if conn != nil {
				RLHLOG(conn.GetState().String())
				cli := pb.NewHakesStoreKVRClient(conn)
				r, err := cli.InDropLog(ctx, req)
				if err == nil && r.Success {
					replyChan <- r
					return
				}
				RLHLOG(fmt.Sprintf("%v request received error: %v", desc, err))
			} else {
				RLHLOG("got empty connection, entering retry to wait for replica")
			}
			if !pauser.Pause() {
				RLHLOG(fmt.Sprintf("exhausted retries in %v", desc))
				replyChan <- dummyfailedReply
				return
			}
			RLHLOG(fmt.Sprintf("retry on %v", desc))
		}
	}

	for _, cch := range c.rcp.GetConns() {
		go dispatcher(cch, c.rscfg.Rp)
	}
	go collector()
	pickedReply := <-completeChan
	if pickedReply == nil || !pickedReply.Success {
		return false
	}
	return true
}

// propagated to followers
func (c *RDLogHandler) Drop(name string) error {
	// todo
	c.dirMu.Lock()
	defer c.dirMu.Unlock()
	delete(c.dir, name)
	if !c.rscfg.Isleader.Load() {
		return nil
	}
	// replicate to followers
	c.replicatedDropLog(name)
	return nil
}

// call only when dirMu is locked
func (c *RDLogHandler) printLogNames() {
	summary := make([]string, 0, len(c.dir))
	for k := range c.dir {
		summary = append(summary, " "+k)
	}
	RLHLOG(fmt.Sprintf("current log entries: %v", summary))
}

// implement the three log service API.
func (c *RDLogHandler) InOpenLog(ctx context.Context, in *pb.InOpenLogRequest) (*pb.InOpenLogReply, error) {
	c.dirMu.Lock()
	defer c.dirMu.Unlock()
	c.printLogNames() // for manual checking
	name := in.LogName
	if name <= c.lastCreated {
		RLHLOG(fmt.Sprintf("created log name is not increasing (last created: %v, target %v)", c.lastCreated, in.LogName))
	} else {
		name = in.LogName
	}
	if name <= c.appendReadyThreshold {
		RLHLOG("request below the ready threshold")
		return nil, ErrRDLogOpSkipped
	}
	if _, ok := c.dir[name]; !ok {
		// create if non-exist
		RLHLOG(fmt.Sprintf("created new log %v", name))
		c.dir[name] = NewRDLog(name, int(in.MaxSz), c.rscfg, c.rcp)
		c.lastCreated = name
	}
	c.printLogNames()
	return &pb.InOpenLogReply{Success: true}, nil
}

func (c *RDLogHandler) InAppendLog(ctx context.Context, in *pb.InAppendLogRequest) (*pb.InAppendLogReply, error) {
	if in.LogName <= c.appendReadyThreshold {
		RLHLOG("request below the ready threshold")
		return &pb.InAppendLogReply{Success: false, HaveReq: false}, nil
	}
	c.dirMu.Lock()
	l, ok := c.dir[in.LogName] // get a reference
	defer c.dirMu.Unlock()     // cannot quickly release the append, concurrent appending the DLog buffer still needs to be protected
	if !ok {
		return nil, ErrRDLogOpSkipped
	} else {
		// append to the log
		curTail := l.tailOffset.Load()
		if curTail == in.Offset {
			// should be the common case, follower append as the leader pushes append requests synchronously
			if len(in.Data) > l.cap-int(curTail) {
				RLHLOG("append failed due to lack of space (possibly an error)")
				return nil, ErrRDLogOpSkipped
			}
			written := copy(l.buf[curTail:], in.Data)
			l.tailOffset.Add(uint32(written))
			return &pb.InAppendLogReply{Success: true, HaveReq: false}, nil
		} else if curTail >= in.Offset+uint32(len(in.Data)) {
			// already appended
			return &pb.InAppendLogReply{Success: true, HaveReq: false}, nil
		} else {
			return &pb.InAppendLogReply{
				Success: false,
				HaveReq: true,
				Req:     &pb.LogDataRequest{LogName: in.LogName, StartOff: curTail, EndOff: in.Offset + uint32(len(in.Data))},
			}, nil
		}
	}
}

func (c *RDLogHandler) InDropLog(ctx context.Context, in *pb.InDropLogRequest) (*pb.InDropLogReply, error) {
	c.dirMu.Lock()
	defer c.dirMu.Unlock()
	if in.LogName <= c.appendReadyThreshold {
		RLHLOG("request below the ready threshold")
		return &pb.InDropLogReply{Success: true, Msg: "skipped"}, nil
	}
	name := in.LogName
	delete(c.dir, name)
	return &pb.InDropLogReply{Success: true}, nil
}

// return the new tail log and the tail offset
func (c *RDLogHandler) instsallLogData(logName string, data []byte, startOff uint32, endOff uint32, cap uint32) {
	c.dirMu.RLock()
	RLHLOG(fmt.Sprintf("installing log data %v, startoff: %d, endoff: %d, cap %d", logName, startOff, endOff, cap))
	// some checking for manual debugging
	if logName < c.lastCreated {
		RLHLOG(fmt.Sprintf("old log received (possibly an error): curr %v, received: %v", c.lastCreated, logName))
		c.dirMu.RUnlock()
		return
	}
	if logName > c.lastCreated {
		if startOff != 0 {
			RLHLOG(fmt.Sprintf("new log received but data is incomplete (possibly an error): received start offset: %v", startOff))
		}
		RLHLOG(fmt.Sprintf("installing a full log: %v", logName))
		c.dirMu.RUnlock()
		// promote to write lock
		c.dirMu.Lock()
		// create the new log
		c.dir[logName] = NewRDLog(logName, int(cap), c.rscfg, c.rcp)
		l := c.dir[logName]
		copy(l.buf, data)
		l.tailOffset.Store(endOff)
		c.dirMu.Unlock()
	} else {
		//  logName == c.lastCreated
		l := c.dir[logName]
		curTail := l.tailOffset.Load()
		if curTail != startOff {
			RLHLOG(fmt.Sprintf("mismatch in the partial current log offset (possibly an error): cur: %v, received %v", curTail, startOff))
		}
		RLHLOG(fmt.Sprintf("updating the tail log: %v, from %d to %d", logName, startOff, endOff))
		// copy over the partial log
		written := copy(l.buf[curTail:], data)
		l.tailOffset.Store(endOff)
		if written != int(endOff-startOff) {
			RLHLOG(fmt.Sprintf("not all received data are applied (possibly an error): written: %v, recieved: endoff: %v, startoff %v", written, endOff, startOff))
		}
		c.dirMu.RUnlock()
	}
}

func (c *RDLogHandler) GetSnapshot() (string, bool, uint32) {
	c.dirMu.RLock()
	defer c.dirMu.RUnlock()
	lastLogName := c.lastCreated
	if l, ok := c.dir[c.lastCreated]; !ok {
		// last log dropped
		return lastLogName, true, 0
	} else {
		// get the last log tail
		return lastLogName, false, l.tailOffset.Load()
	}
}

// serving leaders sync request
func (c *RDLogHandler) GetSnapshotDiff(inLogName string, inLogDropped bool, inTailOffset uint32, requireDiff bool) (string, bool, uint32, []*pb.LogDiff) {
	c.dirMu.RLock()
	defer c.dirMu.RUnlock()
	RLHLOG(fmt.Sprintf("getting snapshot for log >= %v, (dropped: %v) at offset (%v)", inLogName, inLogDropped, inTailOffset))
	diffs := make([]*pb.LogDiff, 0)
	if requireDiff {
		for name, l := range c.dir {
			if name > c.lastCreated {
				continue
			}
			if name < inLogName {
				// the caller already have the data
				RLHLOG("getting diff skipped, caller already have this")
				continue
			} else if name > inLogName {
				curTail := l.tailOffset.Load()
				diffs = append(diffs, &pb.LogDiff{
					LogName:  name,
					Dropped:  false,
					StartOff: 0,
					EndOff:   curTail,
					Capacity: uint32(l.cap),
					Data:     l.buf[:curTail],
				})
				RLHLOG(fmt.Sprintf("shipped the log: %v to caller", name))
			} else {
				if inLogDropped || inTailOffset > l.tailOffset.Load() {
					RLHLOG("skipping tail log as it has been dropped or the caller has a newer tail offset")
					// the caller already have the data
					continue
				}
				// partial log data needed
				curTail := l.tailOffset.Load()
				diffs = append(diffs, &pb.LogDiff{
					LogName:  name,
					Dropped:  false,
					StartOff: inTailOffset,
					EndOff:   curTail,
					Capacity: uint32(l.cap),
					Data:     l.buf[inTailOffset:curTail],
				})
				RLHLOG(fmt.Sprintf("shipped the log: %v from input offset to caller", name))
			}
		}
	}

	lastLogName := c.lastCreated
	if l, ok := c.dir[c.lastCreated]; !ok {
		// last log dropped
		return lastLogName, true, 0, diffs
	} else {
		// get the last log tail
		return lastLogName, false, l.tailOffset.Load(), diffs
	}
}

func (c *RDLogHandler) SetFlushPending(logNameThrehold string) {
	c.dirMu.RLock()
	if logNameThrehold >= c.lastCreated {
		c.dirMu.RUnlock()
		c.dirMu.Lock()
		if logNameThrehold >= c.lastCreated {
			c.appendReadyThreshold = logNameThrehold
		}
		c.dirMu.Unlock()
		return
	}
	c.dirMu.RUnlock()
}

func (c *RDLogHandler) IsSwitchLogRequired() bool {
	c.dirMu.RLock()
	defer c.dirMu.RUnlock()
	return c.appendReadyThreshold >= c.lastCreated
}
