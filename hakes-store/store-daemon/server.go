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
	"net"
	"sync/atomic"
	"time"

	"hakes-store/cloud"
	pb "hakes-store/grpc-api"
	util "hakes-store/hakes-store-util"
	"hakes-store/hakes-store/io"
	"hakes-store/hakes-store/kv"
	"hakes-store/hakes-store/table"
	lambda "hakes-store/lambda"

	"github.com/dgraph-io/badger/v3/y"
	"google.golang.org/grpc"
)

// Currently, the store-daemon allocate a fixed capacity compactor pool to handle compaction jobs. This token based admission should be replaced with resource based later.

type StoreDaemon struct {
	pb.UnimplementedStoreDaemonServer
	pb.UnimplementedSstCacheSvcServer
	server         *grpc.Server
	port           int
	uid            string
	cssType        string
	idAlloc        atomic.Uint32
	m              *monitor
	p              *kv.CompactorPool
	prefetchOptGen func() table.PrefetchOptions
	errChn         chan<- error
	loggerChn      chan<- struct{}
	sc             io.SstCache
	js             *JobScheduler
	reserved       float32 // reserved capacity
	localThreshold float32
	lambdaCompCli  *lambda.LambdaCli
	numCompactor   int
}

func (s *StoreDaemon) String() string {
	return fmt.Sprintf("store-daemon %s: port: %d, enable js: %v, js reserved: %.2f js local threshold: %.2f, lambda: %s, numCompactor: %d", s.uid, s.port, s.js != nil, s.reserved, s.localThreshold, s.lambdaCompCli, s.numCompactor)
}

func NewStoreDaemon(port int, uid, netName string, netBandwidth uint64, cssType string, prefetchMode string, prefetchSize int) (*StoreDaemon, <-chan error) {
	errChn := make(chan error)
	return &StoreDaemon{
		server:         grpc.NewServer(),
		port:           port,
		m:              newMonitor(netName, netBandwidth),
		errChn:         errChn,
		uid:            uid,
		cssType:        cssType,
		prefetchOptGen: buildPrefetchOptGen(prefetchMode, prefetchSize),
	}, errChn
}

func NewStoreDaemonFromConfig(port int, cfg util.StoreDaemonConfig) (*StoreDaemon, <-chan error) {
	errChn := make(chan error)
	numCompactor := cfg.NumCompactor
	if cfg.NumCompactor == 0 {
		numCompactor = 8
	}
	return &StoreDaemon{
		server:         grpc.NewServer(),
		port:           port,
		m:              newMonitor(cfg.NetName, uint64(cfg.NetBandwidth)),
		errChn:         errChn,
		uid:            cfg.ID,
		cssType:        cfg.Css.Type,
		prefetchOptGen: buildPrefetchOptGen(cfg.PrefetchMode, cfg.PrefetchSize<<10),
		sc:             util.NewSstCacheFromConfig(cfg.Sc),
		js:             NewJobScheduler(cfg.Peers),
		reserved:       cfg.Reserved,
		localThreshold: cfg.LocalThreshold,
		lambdaCompCli:  lambda.NewLambdaCli(cfg.LambdaComp),
		numCompactor:   numCompactor,
	}, errChn
}

func buildPrefetchOptGen(prefetchMode string, prefetchSize int) func() table.PrefetchOptions {
	mode := table.NoPrefetch
	switch prefetchMode {
	case "Sync":
		mode = table.SyncPrefetch
	case "Async":
		mode = table.AsyncPrefetch
	}
	return func() table.PrefetchOptions {
		return table.PrefetchOptions{
			Type:         mode,
			PrefetchSize: prefetchSize,
		}
	}
}

func buildCSSConnector(cssType string) func() io.CSSCli {
	switch cssType {
	case "s3":
		return func() io.CSSCli { return cloud.NewS3CCS() }
	case "local":
		log.Println("using local FS as CSS")
		return func() io.CSSCli { return &io.FSCli{} }
	default:
		log.Println("Unknown CSS type: using local FS as CSS")
		return func() io.CSSCli { return &io.FSCli{} }
	}
}

func (s *StoreDaemon) Start() {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", s.port))
	if err != nil {
		s.errChn <- fmt.Errorf("failed to launch server")
		return
	}

	// init the stats
	if !s.m.start() {
		s.errChn <- fmt.Errorf("failed to initialize a resource monitor")
		return
	}

	logger := func(stopC <-chan struct{}) {
		logTicker := time.NewTicker(10 * time.Second)
		defer logTicker.Stop()
		for {
			select {
			case <-logTicker.C:
				log.Println(s.m.info())
			case <-stopC:
				return
			}
		}
	}
	loggerChn := make(chan struct{})
	s.loggerChn = loggerChn
	go logger(loggerChn)

	// setup and launch compactor pool
	cssConnector := buildCSSConnector(s.cssType)
	guidAlloc := func() string {
		return fmt.Sprintf("%v%06d", s.uid, s.idAlloc.Add(1))
	}

	if err := s.sc.Init(); err != nil {
		log.Println("failed to initalize sst cache at store-daemon, request will always be rejected")
		s.sc.Close()
		s.sc = nil
	}

	s.p = kv.NewCompactorPool(s.numCompactor, guidAlloc, cssConnector, s.sc)

	go s.p.Launch()

	// connect to lambda
	if s.lambdaCompCli != nil {
		log.Printf("using lambda compactor (%v)", s.lambdaCompCli)
		if s.lambdaCompCli.Connect() != nil {
			log.Printf("failed to connect to lambda (%v): disabled", s.lambdaCompCli)
			s.lambdaCompCli = nil
		} else {
			log.Printf("connected to lambda (%v)", s.lambdaCompCli)
		}
	} else {
		log.Printf("lambda compactor not supplied")
	}

	go s.js.start()

	pb.RegisterStoreDaemonServer(s.server, s)
	pb.RegisterSstCacheSvcServer(s.server, s)
	log.Printf("store-daemon launched: (%v).", s)
	if err := s.server.Serve(lis); err != nil {
		s.errChn <- fmt.Errorf("failed to serve: %v", err)
	}

}

func (s *StoreDaemon) ScheduleJob(ctx context.Context, req *pb.JobRequest) (*pb.JobReply, error) {
	start := time.Now()
	// prepare the compaction job
	var jobDef kv.CompactionJobDef
	if err := jobDef.Decode(req.Payload); err != nil {
		log.Printf("Job (%v) failed to decode job payload", req.JobId)
		return &pb.JobReply{Success: false}, nil
	}

	// check resources and decide whether do it local or remote
	lca, lma, lna := s.m.getAvailPercent()
	localState := workerState{lca, lma, lna}
	disableSCLoad := false
	if req.Type == pb.JobType_SCHEDULED {
		// if it is a rescheduled task, we can reject it under resource pressure.
		if !pick(localState, s.reserved) {
			return &pb.JobReply{Success: false}, nil
		}
		// for rescheduled task we do not add the build table to local sst cache
		jobDef.UseSstCache = false
		disableSCLoad = true
	} else {
		// received from local kv
		acceptLocal := pick(localState, s.localThreshold)
		log.Printf("first check on local accept: %v", acceptLocal)
		if !acceptLocal && s.js != nil {
			// we can think about reschedule it.
			scheduled, reply, err := s.js.schedule(req)
			if scheduled && err == nil {
				log.Printf("success in rescheduled job (%v) (took: %v), %v -> result: %v", req.JobId, time.Since(start).Milliseconds(), string(req.Payload), string(reply.Payload))
				return reply, nil
			} else {
				log.Println("remote workers busy (will invoke lambda if specified)")
				if s.lambdaCompCli != nil {
					if replyPayload, err := s.lambdaCompCli.Invoke(req.Payload); err == nil {
						log.Printf("success in lambda execution job (%v) (took: %v), %v -> result: %v", req.JobId, time.Since(start).Milliseconds(), string(req.Payload), string(replyPayload))
						return &pb.JobReply{Success: true, Payload: replyPayload}, nil
					}
				}
			}
		}
	}

	log.Printf("job (%v) scheduled: %v", req.JobId, string(req.Payload))
	job := kv.NewCompactionJob(&jobDef, s.prefetchOptGen(), disableSCLoad)
	s.p.Schedule(job)

	// will block and wait for job completion
	success, ret := job.GetResult()
	if !success {
		log.Printf("job (%v) failed (took: %v)", req.JobId, time.Since(start).Milliseconds())
		return &pb.JobReply{Success: false}, nil
	} else {
		payload, err := ret.Encode()
		y.AssertTrue(err == nil)
		log.Printf("job (%v) finished (took: %v): %v", req.JobId, time.Since(start).Milliseconds(), string(payload))
		return &pb.JobReply{Success: true, Payload: payload}, nil
	}
}

func (s *StoreDaemon) GetStats(ctx context.Context, req *pb.StatsRequest) (*pb.StatsReply, error) {
	cpuTotal, cpuAvail := s.m.getCPUInfo()
	memTotal, memAvail := s.m.getMemInfo()
	netTotal, netAvail := s.m.getNetInfo()
	return &pb.StatsReply{
		Cpu: &pb.StatsReply_StatsEntry{Avail: cpuAvail, Total: cpuTotal},
		Mem: &pb.StatsReply_StatsEntry{Avail: memAvail, Total: memTotal},
		Net: &pb.StatsReply_StatsEntry{Avail: netAvail, Total: netTotal},
	}, nil
}

func (s *StoreDaemon) Stop() {
	s.loggerChn <- struct{}{}
	s.js.close()
	s.p.Stop()
	s.server.Stop()
}

// implement sstcache
func (s *StoreDaemon) SstCacheReserve(ctx context.Context, req *pb.SstCacheReserveRequest) (*pb.SstCacheReserveReply, error) {
	if s.sc == nil {
		return &pb.SstCacheReserveReply{Success: false, Msg: "No SSTCache available"}, nil
	}
	if s.sc.Reserve(int(req.Charge)) {
		return &pb.SstCacheReserveReply{Success: true}, nil
	} else {
		return &pb.SstCacheReserveReply{Success: false, Msg: "failed to reserve"}, nil
	}
}

func (s *StoreDaemon) SstCacheRelease(ctx context.Context, req *pb.SstCacheReleaseRequest) (*pb.SstCacheReleaseReply, error) {
	if s.sc == nil {
		return &pb.SstCacheReleaseReply{Success: false, Msg: "No SSTCache available"}, nil
	}
	s.sc.Release(int(req.Charge))
	return &pb.SstCacheReleaseReply{Success: true}, nil
}

func (s *StoreDaemon) SstCacheAdd(ctx context.Context, req *pb.SstCacheAddRequest) (*pb.SstCacheAddReply, error) {
	if s.sc == nil {
		// no sst cache
		return &pb.SstCacheAddReply{Success: false, Msg: "No SSTCache available"}, nil
	}
	if f := s.sc.Add(req.Path, int(req.Charge)); f == nil {
		return &pb.SstCacheAddReply{Success: false, Msg: "failed to add"}, nil
	} else {
		f.Close(-1)
		return &pb.SstCacheAddReply{Success: false, Msg: "failed to add"}, nil
	}
}
func (s *StoreDaemon) SstCacheDrop(ctx context.Context, req *pb.SstCacheDropRequest) (*pb.SstCacheDropReply, error) {
	if s.sc == nil {
		return &pb.SstCacheDropReply{Success: false, Msg: "No SSTCache available"}, nil
	}
	s.sc.Drop(req.Path, int(req.Charge))
	return &pb.SstCacheDropReply{Success: true}, nil
}
