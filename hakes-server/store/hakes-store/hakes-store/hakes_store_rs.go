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
	"context"
	"fmt"
	"net"
	"sync/atomic"

	pb "hakes-store/grpc-api"
	util "hakes-store/hakes-store-util"
	kv "hakes-store/hakes-store/kv"

	"google.golang.org/grpc"
)

var _ ReplicaServer = (*HakesStoreReplicaServer)(nil)

type HakesStoreReplicaServer struct {
	// server definition
	pb.UnimplementedHakesStoreKVRServer

	// serving context
	server *grpc.Server
	port   int
	msgChn chan error

	// internal state
	ne Engine

	// general state
	initialized atomic.Bool // denote if the mode change is the first; different handling

	// mode switch impl
	modeChn chan ModeMsg
	viewChn chan util.ReplicaGroupView

	rscfg       *util.ReplicaRSConfig
	bootstrapRp util.RetryPolicy

	// external connection
	rcp *util.ReplicaConnPool
}

func NewHakesStoreReplicaServer(port, replicaCap, persistCount int, ownAddr string, retryPolicy util.RetryPolicy, opts *kv.Options) (*HakesStoreReplicaServer, <-chan error) {
	msgChn := make(chan error)
	modeChn := make(chan ModeMsg, 10)
	viewChn := make(chan util.ReplicaGroupView, 10)
	if retryPolicy == nil {
		retryPolicy = util.NewFixedIntervalRetry(1000, 30)
	}
	rscfg := &util.ReplicaRSConfig{
		ReplicaCap:   replicaCap - 1,
		PersistCount: persistCount - 1,
		Rp:           retryPolicy,
	}
	busyWaitInterval := 1000
	rcp := util.NewReplicaConnPool(ownAddr, replicaCap-1)
	return &HakesStoreReplicaServer{
		server: grpc.NewServer(),
		port:   port,
		msgChn: msgChn,

		ne: NewHakesStoreEngine(rscfg, rcp, busyWaitInterval, opts),

		modeChn: modeChn,
		viewChn: viewChn,

		rscfg:       rscfg,
		bootstrapRp: retryPolicy,

		rcp: rcp,
	}, msgChn
}

func (s *HakesStoreReplicaServer) modeSwitcher() {
	INFOLOG("mode switcher launched")
	for {
		msg := <-s.modeChn
		switch msg.t {
		case PromoteToleaderMsg:
			INFOLOG("server will promote to leader mode")
			if s.rscfg.Isleader.Load() {
				// already initialized as leader
				return
			}
			// halt for serving external requests
			s.rscfg.ModeChange.Add(1)
			s.rscfg.ModeMu.Lock()
			s.rscfg.Isleader.Store(true)
			if s.ne.InitLeader() {
				s.initialized.Store(true)
				INFOLOG("leader state ready")
			} else {
				s.initialized.Store(false)
				INFOLOG("leader state not ready (will be retried with new view)")
			}
			s.rscfg.ModeMu.Unlock()
			s.rscfg.ModeChange.Add(-1)
			INFOLOG("server promoted to leader mode")
		case DemoteToFollowerMsg:
			INFOLOG("server will set to follower mode")
			if (s.initialized.Load()) && (!s.rscfg.Isleader.Load()) {
				// initialized and is a follower
				return
			}
			s.rscfg.ModeChange.Add(1)
			s.rscfg.ModeMu.Lock()
			s.rscfg.Isleader.Store(false)
			if s.ne.InitFollower() {
				s.initialized.Store(true)
				INFOLOG("follower state ready")
			} else {
				s.initialized.Store(false)
				INFOLOG("follower state not ready (will be retried with new view)")
			}
			s.rscfg.ModeMu.Unlock()
			s.rscfg.ModeChange.Add(-1)
			INFOLOG("server set to follower mode (to follower should only happend during setup)")
		default:
			INFOLOG("unknown mode message skipped")
		}
	}
}

func (s *HakesStoreReplicaServer) viewSwitcher() {
	INFOLOG("view switcher launched")
	for {
		view := <-s.viewChn
		INFOLOG("server will process view change")
		s.rcp.Update(view)
		if !s.rcp.IsActive() {
			continue
		}
		if !s.initialized.Load() {
			s.rscfg.ModeChange.Add(1)
			s.rscfg.ModeMu.Lock()
			// check again after being exclusive holder of mode mu.
			if !s.initialized.Load() {
				isleader := s.rscfg.Isleader.Load()
				ready := false
				if isleader {
					ready = s.ne.InitLeader()
				} else {
					ready = s.ne.InitFollower()
				}
				if ready {
					s.initialized.Store(true)
					INFOLOG(fmt.Sprintf("view triggered state change (is leader %v): ready", ready))
				} else {
					s.initialized.Store(false)
					INFOLOG(fmt.Sprintf("view triggered state change (is leader: %v): not ready (will be retried with new view)", ready))
				}
			}
			s.rscfg.ModeMu.Unlock()
			s.rscfg.ModeChange.Add(-1)
		}
	}
}

func (s *HakesStoreReplicaServer) Start() {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", s.port))
	if err != nil {
		s.msgChn <- fmt.Errorf("failed to launch server")
		return
	}

	// state init is done during mode change
	pb.RegisterHakesStoreKVRServer(s.server, s)

	// launch rcp first before receiving any mode/view switch event from ft replica manager
	s.rcp.Start()
	go s.viewSwitcher()
	go s.modeSwitcher()
	INFOLOG("launching HAKES-Store server")
	if err = s.server.Serve(lis); err != nil {
		s.msgChn <- fmt.Errorf("failed to serve: %v", err)
	}
}

// implement the ReplicaServer interface

func (s *HakesStoreReplicaServer) GetErrChan() <-chan error {
	return s.msgChn
}

func (s *HakesStoreReplicaServer) PromoteToLeader() {
	s.modeChn <- ModeMsg{PromoteToleaderMsg, ""}
}

func (s *HakesStoreReplicaServer) DemoteToFollower() {
	s.modeChn <- ModeMsg{DemoteToFollowerMsg, ""}
}

func (s *HakesStoreReplicaServer) ViewChange(vc util.ReplicaGroupView) {
	s.viewChn <- vc
}

func (s *HakesStoreReplicaServer) Stop() {
	s.server.Stop()
	s.rcp.Stop()
}

// implement HakesKVServer
func (s *HakesStoreReplicaServer) Get(ctx context.Context, in *pb.HakesStoreGetRequest) (*pb.HakesStoreGetReply, error) {
	return s.ne.Get(ctx, in)
}

func (s *HakesStoreReplicaServer) Put(ctx context.Context, in *pb.HakesStorePutRequest) (*pb.HakesStorePutReply, error) {
	return s.ne.Put(ctx, in)
}

func (s *HakesStoreReplicaServer) Del(ctx context.Context, in *pb.HakesStoreDelRequest) (*pb.HakesStoreDelReply, error) {
	return s.ne.Del(ctx, in)
}

func (s *HakesStoreReplicaServer) Scan(ctx context.Context, in *pb.HakesStoreScanRequest) (*pb.HakesStoreScanReply, error) {
	return s.ne.Scan(ctx, in)
}

func (s *HakesStoreReplicaServer) InOpenLog(ctx context.Context, in *pb.InOpenLogRequest) (*pb.InOpenLogReply, error) {
	return s.ne.InOpenLog(ctx, in)
}
func (s *HakesStoreReplicaServer) InAppendLog(ctx context.Context, in *pb.InAppendLogRequest) (*pb.InAppendLogReply, error) {
	return s.ne.InAppendLog(ctx, in)
}
func (s *HakesStoreReplicaServer) InDropLog(ctx context.Context, in *pb.InDropLogRequest) (*pb.InDropLogReply, error) {
	return s.ne.InDropLog(ctx, in)
}
func (s *HakesStoreReplicaServer) InUpdateManifest(ctx context.Context, in *pb.InUpdateManifestRequest) (*pb.InUpdateManifestReply, error) {
	return s.ne.InUpdateManifest(ctx, in)
}
func (s *HakesStoreReplicaServer) InGetSnapshot(ctx context.Context, in *pb.InSnapshotRequest) (*pb.InSnapshotReply, error) {
	return s.ne.InGetSnapshot(ctx, in)
}
func (s *HakesStoreReplicaServer) InFlushSync(ctx context.Context, in *pb.InFlushRequest) (*pb.InFlushReply, error) {
	return s.ne.InFlushSync(ctx, in)
}
