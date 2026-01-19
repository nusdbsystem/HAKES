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

package kv

import (
	"context"
	"fmt"
	"log"
	"sync/atomic"

	proto "hakes-store/grpc-api"
	"hakes-store/hakes-store/io"
	"hakes-store/hakes-store/table"
	dcli "hakes-store/store-daemon-cli"

	"github.com/dgraph-io/badger/v3/y"
	"github.com/pkg/errors"
)

var (
	errFailedRemoteCompaction = errors.New("failed remote compaction")
	errFailedDecodeRCReply    = errors.New("failed to decode remote compaction reply")
	errFailedRCReply          = errors.New("failure reported remote compaction reply")
	errFailedLoadTable        = errors.New("failure to load remote compaction table")
)

// input to compaction scheduler to determine if it is to be scheduled
// and perform scheduling
type RemoteCompactionContext struct {
	kv *DB
	cd *compactDef
}

type CompactionScheduler interface {
	// needs to be thread safe
	// returns: if the compaction is executed remotely, generated new tables, error
	scheduleRemote(RemoteCompactionContext) (bool, []*table.Table, error)
	close() error
}

// common logic for packaging the compaction and schedule to a remote store-daemon.
func scheduleCompact(rcc *RemoteCompactionContext, invoke func([]byte) (bool, []byte)) ([]*table.Table, error) {
	job := prepareCompactionJobDef(*rcc.cd, rcc.kv)
	payload, err := json.Marshal(job)
	y.AssertTrue(err == nil)

	var ret CompactionJobReply
	if success, reply := invoke(payload); !success {
		return nil, errFailedRemoteCompaction
	} else if err := ret.Decode(reply); err != nil {
		return nil, errFailedRCReply
	} else if !ret.Success {
		return nil, errFailedDecodeRCReply
	}

	// load tables
	topts := job.getTableOptions()
	topts.CssCli = rcc.kv.opt.cssCli
	var sc io.SstCache
	if rcc.cd.useSstCache {
		sc = rcc.kv.opt.sstCache
	}
	if newTables, err := parallelLoadTablesWithRetry(ret.NewTables, topts, 10, sc); err != nil {
		return nil, err
	} else {
		return newTables, nil
	}
}

// AlwaysCheckRemoteScheduler decouples resource management and remote scheduling to store-daemon and always talk to local store-daemon service to schedule compaction.
type AlwaysCheckRemoteScheduler struct {
	ndc   *dcli.StoreDaemonCli
	jobId atomic.Uint32
}

func NewAlwaysCheckRemoteScheduler(ndc *dcli.StoreDaemonCli) CompactionScheduler {
	return &AlwaysCheckRemoteScheduler{
		ndc: ndc,
	}
}

func (cs *AlwaysCheckRemoteScheduler) scheduleRemote(rcc RemoteCompactionContext) (bool, []*table.Table, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if cs.ndc == nil {
		// return false remote scheduling, which forces local execution
		return false, nil, nil
	}
	jobId := cs.jobId.Add(1) - 1
	newTables, err := scheduleCompact(&rcc,
		func(payload []byte) (bool, []byte) {
			return cs.ndc.ScheduleJob(ctx, proto.JobType_COMPACTION, fmt.Sprintf("job-%d", jobId), payload)
		})
	if err != nil {
		log.Printf("remote scheduler error: %v", err)
		return false, nil, err // fall back to local compaction
	}
	return true, newTables, nil
}

func (cs *AlwaysCheckRemoteScheduler) close() error {
	cs.ndc.Close()
	return nil
}
