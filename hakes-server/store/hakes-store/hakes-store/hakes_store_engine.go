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
	"context"
	"errors"
	"fmt"
	"time"

	pb "hakes-store/grpc-api"
	io "hakes-store/hakes-store/io"
	kv "hakes-store/hakes-store/kv"

	util "hakes-store/hakes-store-util"
)

type Engine interface {
	InitLeader() bool
	InitFollower() bool
	Get(context.Context, *pb.HakesStoreGetRequest) (*pb.HakesStoreGetReply, error)
	Put(context.Context, *pb.HakesStorePutRequest) (*pb.HakesStorePutReply, error)
	Del(context.Context, *pb.HakesStoreDelRequest) (*pb.HakesStoreDelReply, error)
	Scan(context.Context, *pb.HakesStoreScanRequest) (*pb.HakesStoreScanReply, error)
	InOpenLog(context.Context, *pb.InOpenLogRequest) (*pb.InOpenLogReply, error)
	InAppendLog(context.Context, *pb.InAppendLogRequest) (*pb.InAppendLogReply, error)
	InDropLog(context.Context, *pb.InDropLogRequest) (*pb.InDropLogReply, error)
	InUpdateManifest(context.Context, *pb.InUpdateManifestRequest) (*pb.InUpdateManifestReply, error)
	InGetSnapshot(context.Context, *pb.InSnapshotRequest) (*pb.InSnapshotReply, error)
	InFlushSync(context.Context, *pb.InFlushRequest) (*pb.InFlushReply, error)
}

var _ Engine = (*HakesStoreEngine)(nil)

type HakesStoreEngine struct {
	// internal state
	rlh *RDLogHandler

	// leader lsm
	opts kv.Options
	kv   *kv.HakesKV

	// configs
	busyWaitInterval int // busy waiting in ms
	rscfg            *util.ReplicaRSConfig
	// external connection
	rcp *util.ReplicaConnPool
}

func NewHakesStoreEngine(rscfg *util.ReplicaRSConfig, rcp *util.ReplicaConnPool, busyWaitInterval int, optsIn *kv.Options) *HakesStoreEngine {

	var opts kv.Options
	if optsIn == nil {
		cssCli := &io.FSCli{}
		cssCli.Connect("kv-test")
		opts = kv.DefaultOptions().WithCSSCli(cssCli)
	} else {
		opts = *optsIn
	}

	return &HakesStoreEngine{
		rlh: NewRDLogHandler(rscfg, rcp, busyWaitInterval),

		opts: opts,
		kv:   nil,

		busyWaitInterval: busyWaitInterval,
		rscfg:            rscfg,
		rcp:              rcp,
	}
}

// return the large
func pickLargerSnapshotReply(r1 *pb.InSnapshotReply, r2 *pb.InSnapshotReply) *pb.InSnapshotReply {
	// compare the latest log first
	if r1.LastLogName > r2.LastLogName {
		return r1
	} else if r1.LastLogName < r2.LastLogName {
		return r2
	}
	// then log tail
	if (!r2.LastLogDropped) && (r1.LastLogDropped || r1.LastLogTail > r2.LastLogTail) {
		return r1
	} else if !r1.LastLogDropped && (r2.LastLogDropped || r1.LastLogTail < r2.LastLogTail) {
		return r2
	}
	if r1.ManifestId > r2.ManifestId {
		return r1
	}
	return r2
}

// for leader
func (n *HakesStoreEngine) installSnapshotAsLeader(own *HakesStoreEngineSnapshot, in *pb.InSnapshotReply) {
	// install log
	if in.LastLogName < own.lastLogName {
		INFOLOG("local last log name is larger, skip log sync")
		return // holding the latest log view
	}
	if in.LastLogDropped {
		n.rlh.dirMu.Lock()
		n.rlh.dir = make(map[string]*RDLog)
		n.rlh.lastCreated = in.LastLogName
		n.rlh.dirMu.Unlock()
		return
	}
	// no diff
	if len(in.Logdiffs) == 0 {
		return
	}
	newTailLogName := ""
	activeLogName := make(map[string]struct{})
	for _, d := range in.Logdiffs {
		activeLogName[d.LogName] = struct{}{}
		if d.LogName >= newTailLogName {
			newTailLogName = d.LogName
		}
		n.rlh.instsallLogData(d.LogName, d.Data, d.StartOff, d.EndOff, d.Capacity)
	}
	n.rlh.dirMu.Lock()
	// update the latest created
	n.rlh.lastCreated = newTailLogName
	for name := range n.rlh.dir {
		if _, ok := activeLogName[name]; !ok {
			delete(n.rlh.dir, name)
		}
	}
	n.rlh.printLogNames()
	n.rlh.dirMu.Unlock()
}

func (n *HakesStoreEngine) sendFlushTrigger(LogNameThreshold string) {
	req := &pb.InFlushRequest{LogName: LogNameThreshold}
	desc := "send flush trigger"

	dispatcher := func(cch *util.CliConnHolder, rp util.RetryPolicy) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		conn, _ := cch.GetConn()
		if conn != nil {
			INFOLOG(conn.GetState().String())
			cli := pb.NewHakesStoreKVRClient(conn)
			if _, err := cli.InFlushSync(ctx, req); err != nil {
				INFOLOG(fmt.Sprintf("%v request received error: %v", desc, err))
			}
		} else {
			INFOLOG("got empty connection, entering retry to wait for replica")
		}
	}
	// main logic
	for _, cch := range n.rcp.GetConns() {
		go dispatcher(cch, n.rscfg.Rp)
	}
}

func (n *HakesStoreEngine) installSnapshotAsFollower(own *HakesStoreEngineSnapshot, in *pb.InSnapshotReply) {
	INFOLOG(fmt.Sprintf("last log: %v, last tail off: %d", in.LastLogName, in.LastLogTail))
	if in.LastLogName == "" || kv.IsKVBootstrap(in.LastLogName, in.LastLogTail) {
		INFOLOG("cluster setup: do not send flush trigger")
		return
	}
	n.rlh.setReadyThreshold(in.LastLogName)
	INFOLOG(fmt.Sprintf("send flush trigger on log: %v", in.LastLogName))
	n.sendFlushTrigger(in.LastLogName)
}

// common codes for syncLeaderState and syncFollowerState
func (n *HakesStoreEngine) syncStateImpl(isleader bool, snapshot *HakesStoreEngineSnapshot) *pb.InSnapshotReply {
	INFOLOG("syncing state...")
	req := &pb.InSnapshotRequest{
		ManifestId:     snapshot.manifestId,
		RequireData:    isleader,
		LastLogName:    snapshot.lastLogName,
		LastLogDropped: snapshot.lastLogDropped,
		LastLogTail:    snapshot.lastLogTail,
	}
	dummyfailedReply := &pb.InSnapshotReply{Success: false}
	desc := "get snapshot"

	// getSnapshot from other replica and the op count.
	completeChan := make(chan *pb.InSnapshotReply, len(n.rcp.GetConns())) // pass back the picked snapshot reply
	replyChan := make(chan *pb.InSnapshotReply, len(n.rcp.GetConns()))    // reply buffer of snapshot reply

	// only called by toLeader
	collector := func() {
		success := 0
		var ret *pb.InSnapshotReply
		for i := 0; i < n.rscfg.ReplicaCap; i++ {
			reply, ok := <-replyChan
			if !ok {
				INFOLOG("unexpected closure of reply channel")
				close(completeChan)
				return
			}
			if !reply.Success {
				INFOLOG(fmt.Sprintf("received one failed %v reply", desc))
				continue
			}
			INFOLOG(fmt.Sprintf("received one successful %v reply", desc))
			success++
			// pick the one that is more upto date
			if ret == nil {
				ret = reply
			} else {
				// get a more up to date one
				ret = pickLargerSnapshotReply(ret, reply)
			}
			if success >= n.rscfg.PersistCount {
				completeChan <- ret
			}
		}
		if success < n.rscfg.PersistCount {
			INFOLOG("not enough successful replies")
			completeChan <- nil
		}
		INFOLOG(fmt.Sprintf("%v received all reply", desc))
	}

	// only called by toLeader
	dispatcher := func(cch *util.CliConnHolder, rp util.RetryPolicy) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		pauser := rp.NewPauser()
		for {
			conn, _ := cch.GetConn()
			if conn != nil {
				INFOLOG(conn.GetState().String())
				cli := pb.NewHakesStoreKVRClient(conn)
				r, err := cli.InGetSnapshot(ctx, req)
				if err == nil && r.Success {
					replyChan <- r
					return
				}
				INFOLOG(fmt.Sprintf("%v request received error: %v", desc, err))
			} else {
				INFOLOG("got empty connection, entering retry to wait for replica")
			}
			if !pauser.Pause() {
				INFOLOG(fmt.Sprintf("exhaused retries in %v", desc))
				replyChan <- dummyfailedReply
				return
			}
			INFOLOG(fmt.Sprintf("retry on %v", desc))
		}
	}

	// main logic
	for _, cch := range n.rcp.GetConns() {
		go dispatcher(cch, n.rscfg.Rp)
	}
	go collector()

	pickedReply := <-completeChan
	return pickedReply
}

func (n *HakesStoreEngine) syncLeaderState(snapshot *HakesStoreEngineSnapshot) bool {
	pickedReply := n.syncStateImpl(true, snapshot)
	if pickedReply == nil || !pickedReply.Success {
		// it could happen during initial setup when the replica group is not stablized within retry time.
		INFOLOG("did not receive valid snapshot, this should never happen with less than 1/2 failure")
		return false
	}
	// install the snapshot
	n.installSnapshotAsLeader(snapshot, pickedReply)
	INFOLOG(fmt.Sprintf("new snapshot installed (manifestid: %v, lastlog: %v (dropped: %v, off %v)", pickedReply.ManifestId, pickedReply.LastLogName, pickedReply.LastLogDropped, pickedReply.LastLogTail))
	return true
}

func (n *HakesStoreEngine) syncFollowerState(snapshot *HakesStoreEngineSnapshot) bool {
	pickedReply := n.syncStateImpl(false, snapshot)
	if pickedReply == nil || !pickedReply.Success {
		// it could happen during initial setup when the replica group is not stablized within retry time.
		INFOLOG("did not receive valid snapshot, this should never happen with less than 1/2 failure")
		return false
	}
	// install the snapshot
	n.installSnapshotAsFollower(snapshot, pickedReply)
	INFOLOG(fmt.Sprintf("new snapshot installed (manifestid: %v, lastlog: %v (dropped: %v, off %v)", pickedReply.ManifestId, pickedReply.LastLogName, pickedReply.LastLogDropped, pickedReply.LastLogTail))
	return true
}

type HakesStoreEngineSnapshot struct {
	manifestId     uint32
	lastLogName    string
	lastLogDropped bool
	lastLogTail    uint32
}

func (n *HakesStoreEngine) getSnapshot() *HakesStoreEngineSnapshot {
	// get log first
	lastLogName, lastLogDropped, lastLogTail := n.rlh.GetSnapshot()
	// get then manifest.
	// manifestId, manifestData := n.rmh.GetSnapshot()
	INFOLOG("constructed local snapshot")
	return &HakesStoreEngineSnapshot{
		lastLogName:    lastLogName,
		lastLogDropped: lastLogDropped,
		lastLogTail:    lastLogTail,
	}
}

func (n *HakesStoreEngine) InitLeader() bool {
	INFOLOG("HakesStoreEngine leader init started ...")
	// synchronization with other snapshot
	// get snapshot for log first then manifest
	if !n.syncLeaderState(n.getSnapshot()) {
		// wait for further view change
		return false
	}

	n.opts = n.opts.WithDLogHandler(n.rlh)
	n.kv = kv.NewHakeSKVWithOpts(n.opts)
	// open the kv on the synchronized snapshot
	INFOLOG("Opening HAKESKV...")
	err := n.kv.Open() // the open operation will uses the the manifest and durable log handler to instantiate the memtable and LSM tree.
	if err != nil {
		INFOLOG(fmt.Sprintf("failed to open kv: %v", err))
		return false
	}
	INFOLOG("HakesStoreEngine leader init finished ...")
	return true
}

func (n *HakesStoreEngine) InitFollower() bool {
	// synchronization with other snapshot
	// get snapshot for log first then manifest
	if !n.syncFollowerState(n.getSnapshot()) {
		// wait for further view change
		return false
	}

	if n.kv != nil {
		INFOLOG("HAKES kv closed during demotion to follower (possibly an error)")
		// for manual checking
		n.kv.Close()
	}
	return true
}

// implement the HakesKVServer

var (
	ErrFollowerReceivingExternalReq = errors.New("follower received external request")
)

// leader
func (s *HakesStoreEngine) Put(ctx context.Context, r *pb.HakesStorePutRequest) (*pb.HakesStorePutReply, error) {
	waitCount := 0
	for s.rscfg.ModeChange.Load() > 0 {
		waitCount++
		time.Sleep(time.Duration(s.busyWaitInterval) * time.Millisecond)
	}
	s.rscfg.ModeMu.RLock()
	defer s.rscfg.ModeMu.RUnlock()
	if !s.rscfg.Isleader.Load() {
		return &pb.HakesStorePutReply{Success: false, ErrMsg: "follower do not handle external request"}, nil
	}
	if err := s.kv.Put(r.GetKey(), r.GetVal()); err != nil {
		return &pb.HakesStorePutReply{Success: false, ErrMsg: err.Error()}, nil
	} else {
		return &pb.HakesStorePutReply{Success: true}, nil
	}
}

// leader
func (s *HakesStoreEngine) Get(ctx context.Context, r *pb.HakesStoreGetRequest) (*pb.HakesStoreGetReply, error) {
	waitCount := 0
	for s.rscfg.ModeChange.Load() > 0 {
		waitCount++
		time.Sleep(time.Duration(s.busyWaitInterval) * time.Millisecond)
	}
	s.rscfg.ModeMu.RLock()
	defer s.rscfg.ModeMu.RUnlock()

	if !s.rscfg.Isleader.Load() {
		return &pb.HakesStoreGetReply{Success: false, ErrMsg: "follower do not handle external request"}, nil
	}
	if val, err := s.kv.Get(r.GetKey()); err == kv.ErrKeyNotFound {
		return &pb.HakesStoreGetReply{Success: true, Found: false}, nil
	} else if err != nil {
		INFOLOG(fmt.Sprintf("HAKES-Store engine error %v", err))
		return &pb.HakesStoreGetReply{Success: false, ErrMsg: err.Error()}, nil
	} else {
		return &pb.HakesStoreGetReply{Success: true, Found: true, Val: val}, nil
	}
}

func (s *HakesStoreEngine) Del(ctx context.Context, r *pb.HakesStoreDelRequest) (*pb.HakesStoreDelReply, error) {
	waitCount := 0
	for s.rscfg.ModeChange.Load() > 0 {
		waitCount++
		time.Sleep(time.Duration(s.busyWaitInterval) * time.Millisecond)
	}
	s.rscfg.ModeMu.RLock()
	defer s.rscfg.ModeMu.RUnlock()
	if !s.rscfg.Isleader.Load() {
		return &pb.HakesStoreDelReply{Success: false, ErrMsg: "follower do not handle external request"}, nil
	}
	if err := s.kv.Delete(r.GetKey()); err != nil {
		return &pb.HakesStoreDelReply{Success: false, ErrMsg: err.Error()}, nil
	} else {
		return &pb.HakesStoreDelReply{Success: true}, nil
	}
}

func (s *HakesStoreEngine) Scan(ctx context.Context, r *pb.HakesStoreScanRequest) (*pb.HakesStoreScanReply, error) {
	waitCount := 0
	for s.rscfg.ModeChange.Load() > 0 {
		waitCount++
		time.Sleep(time.Duration(s.busyWaitInterval) * time.Millisecond)
	}
	s.rscfg.ModeMu.RLock()
	defer s.rscfg.ModeMu.RUnlock()
	if !s.rscfg.Isleader.Load() {
		return &pb.HakesStoreScanReply{Success: false, ErrMsg: "follower do not handle external request"}, nil
	}

	if r.GetType() != pb.HakesStoreScanType_FIXEDCOUNTSCAN && r.GetType() != pb.HakesStoreScanType_ENDKEYSCAN && r.GetType() != pb.HakesStoreScanType_FULLSCAN {
		return &pb.HakesStoreScanReply{Success: false, ErrMsg: "unknown scan type"}, nil

	}

	var keySet, valSet [][]byte
	count := 0
	collect := func(item *kv.Item) error {
		if value, err := item.ValueCopy(nil); err != nil {
			return err
		} else {
			keySet = append(keySet, item.KeyCopy(nil))
			valSet = append(valSet, value)
			count++
			return nil
		}
	}

	it := s.kv.NewIterator(kv.DefaultIteratorOptions)
	defer it.Close()

	if r.GetType() == pb.HakesStoreScanType_FIXEDCOUNTSCAN {
		i := 0
		for it.Seek(r.GetStartKey()); it.Valid() && i < int(r.GetCount()); it.Next() {
			if err := collect(it.Item()); err != nil {
				return &pb.HakesStoreScanReply{Success: false, ErrMsg: err.Error()}, nil
			}
			i++
		}
	} else if r.GetType() == pb.HakesStoreScanType_ENDKEYSCAN {
		endKey := r.GetEndKey()
		for it.Seek(r.GetStartKey()); it.Valid(); it.Next() {
			item := it.Item()
			if bytes.Compare(item.Key(), endKey) > 0 {
				break
			}
			if err := collect(item); err != nil {
				return &pb.HakesStoreScanReply{Success: false, ErrMsg: err.Error()}, nil
			}
		}
	} else if r.GetType() == pb.HakesStoreScanType_FULLSCAN {
		for it.Seek(r.GetStartKey()); it.Valid(); it.Next() {
			if err := collect(it.Item()); err != nil {
				return &pb.HakesStoreScanReply{Success: false, ErrMsg: err.Error()}, nil
			}
		}
	}
	return &pb.HakesStoreScanReply{Success: true, Count: uint32(count), KeySet: keySet, ValSet: valSet}, nil
}

// follower
func (s *HakesStoreEngine) InOpenLog(ctx context.Context, in *pb.InOpenLogRequest) (*pb.InOpenLogReply, error) {
	waitCount := 0
	for s.rscfg.ModeChange.Load() > 0 {
		waitCount++
		time.Sleep(time.Duration(s.busyWaitInterval) * time.Millisecond)
	}
	s.rscfg.ModeMu.RLock()
	defer s.rscfg.ModeMu.RUnlock()
	if s.rscfg.Isleader.Load() {
		return nil, ErrRDLogOpSkipped
	}
	return s.rlh.InOpenLog(ctx, in)
}

// follower
func (s *HakesStoreEngine) InAppendLog(ctx context.Context, in *pb.InAppendLogRequest) (*pb.InAppendLogReply, error) {
	waitCount := 0
	for s.rscfg.ModeChange.Load() > 0 {
		waitCount++
		time.Sleep(time.Duration(s.busyWaitInterval) * time.Millisecond)
	}
	s.rscfg.ModeMu.RLock()
	defer s.rscfg.ModeMu.RUnlock()

	if s.rscfg.Isleader.Load() {
		return nil, ErrRDLogOpSkipped
	}
	return s.rlh.InAppendLog(ctx, in)
}

// follower
func (s *HakesStoreEngine) InDropLog(ctx context.Context, in *pb.InDropLogRequest) (*pb.InDropLogReply, error) {
	waitCount := 0
	for s.rscfg.ModeChange.Load() > 0 {
		waitCount++
		time.Sleep(time.Duration(s.busyWaitInterval) * time.Millisecond)
	}
	s.rscfg.ModeMu.RLock()
	defer s.rscfg.ModeMu.RUnlock()
	if s.rscfg.Isleader.Load() {
		return nil, ErrRDLogOpSkipped
	}
	return s.rlh.InDropLog(ctx, in)
}

// follower
func (s *HakesStoreEngine) InUpdateManifest(ctx context.Context, in *pb.InUpdateManifestRequest) (*pb.InUpdateManifestReply, error) {
	return &pb.InUpdateManifestReply{Success: true}, nil // this function is no longer used
}

// both
func (s *HakesStoreEngine) InGetSnapshot(ctx context.Context, in *pb.InSnapshotRequest) (*pb.InSnapshotReply, error) {
	// get log first
	lastLogName, lastLogDropped, lastLogTail, diff := s.rlh.GetSnapshotDiff(in.LastLogName, in.LastLogDropped, in.LastLogTail, in.RequireData)
	// get then manifest.
	return &pb.InSnapshotReply{
		Success:        true,
		LastLogName:    lastLogName,
		LastLogDropped: lastLogDropped,
		LastLogTail:    lastLogTail,
		Logdiffs:       diff,
	}, nil
}

// leader
func (s *HakesStoreEngine) InFlushSync(ctx context.Context, in *pb.InFlushRequest) (*pb.InFlushReply, error) {
	if s.rscfg.Isleader.Load() {
		// set to rlh handler to be handled in next append
		s.rlh.SetFlushPending(in.LogName)
		return &pb.InFlushReply{Success: true, Msg: "leader set"}, nil
	}
	return &pb.InFlushReply{Success: true, Msg: "follower skipped"}, nil
}
