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
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"

	"fnpacker/fnpack"
)

var (
	port = flag.Int("port", 7310, "The port to listen on for HTTP requests")
)

func main() {
	flag.Parse()

	fmt.Println("Hello")
	var err error

	// set controller address
	ow_address, exist := os.LookupEnv("OW_SERVICE_ADDRESS")
	if !exist {
		ow_address = "localhost"
		log.Print("controller addr not set, default to localhost")
	} else {
		log.Println("proxy address: " + ow_address)
	}

	// set controller port
	ow_port, exist := os.LookupEnv("OW_SERVICE_PORT")
	if !exist {
		ow_port = "7312"
		log.Print("controller port not set, default to 7312")
	} else {
		log.Printf("proxy address: %v:%v\n", ow_address, ow_port)
	}

	// set controller port
	ow_auth, exist := os.LookupEnv("OW_SERVICE_AUTH")
	if !exist {
		ow_auth = "23bc46b1-71f6-4ed5-8c54-816aa4f8c502:123zO3xZCLrMN6v2BKK1dXYFpXlPkccOFqm12CdAsMgRU4VrNZ9lyGVCGuMDGIwP"
	} else {
		log.Println("OW AUTH: " + ow_auth)
	}

	pool_size := fnpack.DEFAULT_POOL_SIZE
	pool_size_str, exist := os.LookupEnv("OW_FUNCTION_MANAGER_POOL_SIZE")

	// set pool size
	if !exist {
		log.Printf("function manager pool size set to default %v", pool_size)
	} else {
		pool_size, err = strconv.Atoi(pool_size_str)
		if err != nil {
			log.Printf("invalid pool size environment variable: %e", err)
			pool_size = fnpack.DEFAULT_POOL_SIZE
		} else {
			log.Printf("function manager pool size: %v\n", pool_size)
		}
	}

	// launch function manager
	fp, err := fnpack.NewOwFunctionPacker()
	mux := http.NewServeMux()
	mux.HandleFunc("/create", fp.HandleCreate)
	mux.HandleFunc("/delete", fp.HandleDelete)
	mux.HandleFunc("/list", fp.HandleList)
	mux.HandleFunc("/invoke", fp.HandleInvoke)

	if err != nil {
		log.Fatal("Failed to create function manager: ", err.Error())
	} else {
		log.Printf("Launching function manager on port %d", *port)
		log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", *port), mux))
	}
}
