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

package StoreDaemonCli

import (
	"context"
	"fmt"
	"log"
	"time"

	pb "hakes-store/grpc-api"
	"hakes-store/hakes-store/io"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

var _ io.SstCacheCli = (*StoreDaemonCli)(nil)

type StoreDaemonCli struct {
	svrAddr string
	conn    *grpc.ClientConn
}

func NewStoreDaemonCli(svrAddr string) *StoreDaemonCli {
	return &StoreDaemonCli{
		svrAddr: svrAddr,
	}
}

func (c *StoreDaemonCli) String() string {
	return fmt.Sprintf("connection (%s)", c.svrAddr)
}

func (c *StoreDaemonCli) Connect() error {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*3)
	defer cancel()
	conn, err := grpc.DialContext(ctx, c.svrAddr, grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock())
	if err != nil {
		return err
	}
	c.conn = conn
	return nil
}

func (c *StoreDaemonCli) ScheduleJob(ctx context.Context, jobType pb.JobType, jobId string, payload []byte) (bool, []byte) {
	cli := pb.NewStoreDaemonClient(c.conn)
	reply, err := cli.ScheduleJob(ctx, &pb.JobRequest{Type: jobType, JobId: jobId, Payload: payload})
	if err != nil {
		log.Printf("Job (%v) failed to decode job payload", jobId)
		return false, nil
	}
	return reply.Success, reply.Payload
}

func (c *StoreDaemonCli) GetStats(ctx context.Context, payload string) (float32, float32, float32) {
	cli := pb.NewStoreDaemonClient(c.conn)
	reply, err := cli.GetStats(ctx, &pb.StatsRequest{Payload: payload})
	if err != nil {
		return 0, 0, 0
	}
	return reply.Cpu.Avail, reply.Mem.Avail, reply.Net.Avail
}

func (c *StoreDaemonCli) Close() {
	if c.conn != nil {
		c.conn.Close()
	}
}

// SstCacheCli API implementation

func (c *StoreDaemonCli) SstCacheReserve(ctx context.Context, charge int) bool {
	cli := pb.NewSstCacheSvcClient(c.conn)
	reply, err := cli.SstCacheReserve(ctx, &pb.SstCacheReserveRequest{Charge: uint32(charge)})
	if err != nil {
		log.Printf("remote sstcache client: failed to reserve %d (error %v)", charge, err)
		return false
	}
	if !reply.Success {
		log.Printf("remote sstcache client: failed to reserve %d (rejected)", charge)
		return false
	}
	return true
}

func (c *StoreDaemonCli) SstCacheRelease(ctx context.Context, charge int) {
	cli := pb.NewSstCacheSvcClient(c.conn)
	reply, err := cli.SstCacheRelease(ctx, &pb.SstCacheReleaseRequest{Charge: uint32(charge)})
	if err != nil {
		log.Printf("remote sstcache client: failed to reserve %d (error %v)", charge, err)
	}
	if !reply.Success {
		log.Printf("remote sstcache client: failed to reserve %d (rejected)", charge)
	}
}

func (c *StoreDaemonCli) SstCacheAdd(ctx context.Context, path string, charge int) error {
	cli := pb.NewSstCacheSvcClient(c.conn)
	reply, err := cli.SstCacheAdd(ctx, &pb.SstCacheAddRequest{Path: path, Charge: uint32(charge)})
	if err != nil {
		return fmt.Errorf("remote sstcache client: failed to add %s (error %v)", path, err)
	}
	if !reply.Success {
		return fmt.Errorf("remote sstcache client: failed to add %v (rejected)", path)
	}
	return nil
}

func (c *StoreDaemonCli) SstCacheDrop(ctx context.Context, path string, charge int) {
	cli := pb.NewSstCacheSvcClient(c.conn)
	reply, err := cli.SstCacheDrop(ctx, &pb.SstCacheDropRequest{Path: path, Charge: uint32(charge)})
	if err != nil {
		log.Printf("remote sstcache client: failed to drop %s (error %v)", path, err)
	}
	if !reply.Success {
		log.Printf("remote sstcache client: failed to drop %v (rejected)", path)
	}
}
