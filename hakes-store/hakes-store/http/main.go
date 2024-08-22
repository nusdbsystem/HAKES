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

// Package implements a replica.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"path/filepath"
	"time"

	hakesstore "hakes-store/hakes-store"
	util "hakes-store/hakes-store-util"
	"hakes-store/hakes-store/kv"

	pb "hakes-store/grpc-api"

	"github.com/gin-gonic/gin"
)

func INFOLOG(msg string) {
	log.Printf("%v [replica-%d]: %v\n", time.Now().UnixNano(), *uid, msg)
}

var (
	// replica setting
	uid            = flag.Int("replicaUid", 0, "the assigned replica uid") // only for logging
	cfgPath        = flag.String("config", "", "the replica server config")
	hakeskvCfgPath = flag.String("hakeskvConfig", "", "the hakeskv config")
)

func main() {
	flag.Parse()

	// for hakesstore, we enforce requirement for a config file
	if len(*cfgPath) == 0 {
		log.Fatal("no config provided")
	}
	cfg, err := hakesstore.ParseConfigFile(*cfgPath)
	if err != nil {
		log.Fatalf("failed to parse replica server config: %v", *cfgPath)
	}

	// parse the hakeskv options
	var opts *kv.Options
	var resourceGC func()
	if len(*hakeskvCfgPath) > 0 {
		if hakeskvCfg, err := util.ParseHakesKVConfig(*hakeskvCfgPath); err != nil {
			log.Printf("failed to parse hakeskv config: %v (fall back to default)", *hakeskvCfg)
		} else {
			log.Printf("using hakeskv config: %v", *hakeskvCfg)
			opts, resourceGC = util.PrepareHakesKVOptFromConfig(hakeskvCfg)
		}
	}

	// service should start by default in follower mode
	ownAddr := fmt.Sprintf("%s:%d", cfg.Addr, cfg.Port)
	rs, kvec := hakesstore.NewHakesStoreReplicaServer(cfg.Port, 3, 2, ownAddr, nil, opts)
	go rs.Start()
	defer resourceGC()

	http_handle := func(c *gin.Context) {
		var req HttpRequest
		if err := c.BindJSON(&req); err != nil {
			c.IndentedJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		var ret HttpResponse
		switch req.Type {
		case "get":
			for _, key := range req.Keys {
				reply, err := rs.Get(context.Background(), &pb.HakesStoreGetRequest{Key: []byte(key)})
				if err != nil || !reply.Success {
					ret.Values = append(ret.Values, "")
				} else {
					ret.Values = append(ret.Values, HttpPayloadEncode(reply.Val))
				}
			}
			ret.Status = true
		case "put":
			for i, key := range req.Keys {
				key_bytes := []byte(key)
				val_bytes, err := HttpPayloadDecode(req.Values[i])
				log.Printf("put: %v, %v\n", key, val_bytes)
				if err != nil {
					ret.Values = append(ret.Values, "")
				} else if reply, err := rs.Put(context.Background(), &pb.HakesStorePutRequest{Key: key_bytes, Val: val_bytes}); err != nil || !reply.Success {
					ret.Values = append(ret.Values, "")
				} else {
					ret.Values = append(ret.Values, key)
				}
			}
			ret.Status = true
		case "del":
			for _, key := range req.Keys {
				reply, err := rs.Del(context.Background(), &pb.HakesStoreDelRequest{Key: []byte(key)})
				if err != nil || !reply.Success {
					ret.Values = append(ret.Values, "")
				} else {
					ret.Values = append(ret.Values, key)
				}
			}
			ret.Status = true
		default:
			c.IndentedJSON(http.StatusBadRequest, gin.H{"error": "invalid request type"})
		}
		c.IndentedJSON(http.StatusOK, ret)
	}

	// start http server
	router := gin.Default()
	router.POST("/kv", http_handle)
	go router.Run(fmt.Sprintf(":%d", cfg.HttpPort))

	// setup ft replica manager
	ftr, mec := hakesstore.NewFTReplica(filepath.Join(cfg.Root, cfg.Region), fmt.Sprintf("%s:%d", cfg.Addr, cfg.Port), cfg.Zk, rs)
	go ftr.Start()
	// main thread watch for error and perform cleanup
	for {
		select {
		case err := <-kvec:
			log.Fatalf("kv error: %v", err)
			panic(err)
		case err := <-mec:
			log.Fatalf("ft replica error: %v", err)
			// tear down
			rs.Stop()
			return
		case <-time.After(time.Second):
			// pass
		}
	}
}
