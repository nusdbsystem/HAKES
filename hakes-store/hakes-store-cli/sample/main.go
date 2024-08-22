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

// Package main implements a client.
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"path/filepath"
	"time"

	cli "hakes-store/hakes-store-cli"

	"google.golang.org/grpc/status"
)

var (
	cfgPath = flag.String("config", "", "the replica server config")
)

func test_hakes_store(cfg cli.Config) {
	// Set up a connection to the server.
	cli := cli.NewHakesStoreClient(filepath.Join(cfg.Root, cfg.Region[0]), cfg.Zk, nil)
	defer cli.Close()
	if err := cli.Setup(); err != nil {
		log.Printf("failed to setup service client: %v", err)
		os.Exit(1)
	}

	// Contact the server and print out its response.
	{
		ctx, cancel := context.WithTimeout(context.Background(), time.Second*20)
		defer cancel()
		log.Printf("Put sent")
		err := cli.Put(ctx, []byte("k1"), []byte("v1"))
		if err != nil {
			log.Fatalf("could not Put: %v, %v", status.Convert(err).Code().String(), err)
		}
		log.Println("Put done for k1")
	}

	<-time.After(time.Second * 5)

	{

		ctx, cancel := context.WithTimeout(context.Background(), time.Second*20)
		defer cancel()
		log.Printf("Get sent")
		r, err := cli.Get(ctx, []byte("k1"))
		if err != nil {
			log.Fatalf("could not get: %v: %v", status.Convert(err).Code().String(), err)
		}
		log.Printf("Get: %s", r)
	}
}

// client should handle error:
//
//	DeadlineExceeded (client set deadline failed), we can retry with a backoff (need to reset the context though, not implemented for now)/ extend the dealine for exp
//	Unavailable (connection to server lost), leader is probably down, fetch new view from zookeeper.
func main() {
	flag.Parse()
	if len(*cfgPath) == 0 {
		log.Fatal("no config provided")
	}
	cfg, err := cli.ParseConfigFile(*cfgPath)
	if err != nil {
		log.Fatalf("failed to parse replica server client config: %v", *cfgPath)
	}
	test_hakes_store(*cfg)
}
