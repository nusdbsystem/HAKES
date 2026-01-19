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
	"flag"
	"log"

	util "hakes-store/hakes-store-util"
	storedaemon "hakes-store/store-daemon"
)

var (
	port         = flag.Int("port", 2191, "The store-daemon port")
	netName      = flag.String("net", "eno1", "The network interface name")
	netBandwidth = flag.Uint64("bandwidth", 100<<10>>3, "The network bandwidth in KB/s")
	cssType      = flag.String("cssType", "local", "The css type: local(default)/s3")
	prefetchMode = flag.String("prefetchMode", "None", "The compaction prefetching mode: None(default)/Async/Sync")
	prefetchSize = flag.Int("prefetchSize", 8, "The prefetchSize (default 8KB)")
	config       = flag.String("config", "",
		"the store-daemon config yaml (override the command line settings when provided)")
)

func main() {
	flag.Parse()
	var s *storedaemon.StoreDaemon
	var errChn <-chan error
	if len(*config) > 0 {
		if cfg, err := util.ParseStoreDaemonConfig(*config); err != nil {
			log.Printf("failed to parse store-daemon config: %v (fall back to command line setting)", *config)
		} else {
			log.Printf("using config: %v", *config)
			s, errChn = storedaemon.NewStoreDaemonFromConfig(*port, *cfg)
		}
	}
	if s == nil {
		s, errChn = storedaemon.NewStoreDaemon(*port, "r0", *netName, *netBandwidth,
			*cssType, *prefetchMode, *prefetchSize<<10)
	}

	go s.Start()

	err := <-errChn
	log.Printf("server encountered error (terminating): %v", err)
	s.Stop()
}
