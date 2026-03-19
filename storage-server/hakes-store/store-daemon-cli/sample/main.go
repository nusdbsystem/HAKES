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
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	pb "hakes-store/grpc-api"
	dcli "hakes-store/store-daemon-cli"
)

var (
	addr = flag.String("addr", "127.0.0.1", "The store-daemon address")
	port = flag.Int("port", 2191, "The store-daemon port")
)

func main() {
	flag.Parse()
	cli := dcli.NewStoreDaemonCli(fmt.Sprintf("%v:%d", *addr, *port))
	defer cli.Close()
	log.Printf("connecting to %v", fmt.Sprintf("%v:%d", *addr, *port))
	if err := cli.Connect(); err != nil {
		panic(err)
	}

	ctx1, cancel1 := context.WithTimeout(context.Background(), time.Second)
	defer cancel1()
	if status, reply := cli.ScheduleJob(ctx1, pb.JobType_COMPACTION, "jobid-1", []byte{}); status {
		fmt.Printf("schedule job response: %v\n", reply)
	} else {
		fmt.Println("schedule job failed.")
	}

	ctx2, cancel2 := context.WithTimeout(context.Background(), time.Second)
	defer cancel2()
	cpu, mem, net := cli.GetStats(ctx2, "")
	fmt.Printf("CPU available: %f core\nmem: %f KB\nnetwork bandwidth: %f KB/s\n", cpu, mem, net)
}
