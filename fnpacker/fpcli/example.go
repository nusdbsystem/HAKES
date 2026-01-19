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
	"net/http"
	"os"

	"fnpacker/req"
)

const fpaddr = "localhost"
const fpport = 7310

// const fnname = "crtw-tvm-mb-4"
const fnname = "crtw-tvm-rs-switch"

// const fnimage = "crtw-tvm-mb-4:v4"
const fnimage = "crtw-tvm-rs-1:v5"

// const fnconcurrency = 4
const fnconcurrency = 1

const fnmembudget = 384

func main() {
	url := fmt.Sprintf("http://%v:%d", fpaddr, fpport)

	// --- basics ---
	fpcli := &FPClient{cli: http.DefaultClient, url: url}
	// create
	fpcli.Create(fnname, "blackbox", fnimage, fnconcurrency, fnmembudget)

	// list
	fpcli.List()

	// invoke
	ireq := req.InvokeRequest{}
	ireqbytes, _ := os.ReadFile("data/tvm_mb_req.json")
	ireq.Load(ireqbytes)
	ireq.KeyServiceAddr = "localhost"
	fpcli.Invoke(fnname, &ireq)

	// delete
	fpcli.Delete(fnname)
	// --- basics ---
}
