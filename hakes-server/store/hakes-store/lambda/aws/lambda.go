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

package main

import (
	"fmt"
	"log"
	"sync/atomic"

	"hakes-store/cloud"

	"github.com/aws/aws-lambda-go/lambda"

	"hakes-store/hakes-store/io"
	"hakes-store/hakes-store/kv"
	"hakes-store/hakes-store/table"
)

var cssCli io.CSSCli
var idAlloc atomic.Uint32

func buildS3CSS() io.CSSCli {
	return cloud.NewS3CCS()
}

func handler(input kv.CompactionJobDef) (string, error) {
	var jobDef = input
	prefetchOpts := table.PrefetchOptions{
		Type:         table.SyncPrefetch,
		PrefetchSize: 2 << 20,
	}
	if cssCli == nil {
		cssCli = buildS3CSS()
	}

	job := kv.NewCompactionJob(&jobDef, prefetchOpts, true)
	job.UseSstCache = false
	guidAlloc := func() string {
		return fmt.Sprintf("%v%06d", job.OutTablePrefix, idAlloc.Add(1))
	}

	var jobReply kv.CompactionJobReply
	c := kv.NewCompactor(cssCli, nil, guidAlloc, nil)
	if newTableMetas, err := c.RunDirect(job); err != nil {
		log.Printf("compaction job %s failed: %v", job.OutTablePrefix, err)
		jobReply = kv.CompactionJobReply{Success: false}
		jobReplyByte, err := jobReply.Encode()
		return string(jobReplyByte), err
	} else {
		jobReply = kv.CompactionJobReply{Success: true, NewTables: newTableMetas}
		if response, err := jobReply.Encode(); err != nil {
			log.Printf("fail to encode compaction job reply: %v", err)
			jobReply = kv.CompactionJobReply{Success: false}
			jobReplyByte, err := jobReply.Encode()
			return string(jobReplyByte), err
		} else {
			return string(response), nil
		}
	}
}

func main() {
	lambda.Start(handler)
}
