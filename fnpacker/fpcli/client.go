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
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync/atomic"

	fpreq "fnpacker/req"
)

const (
	fplistpath   = "/list"
	fpcreatepath = "/create"
	fpdeletepath = "/delete"
	fpinvokepath = "/invoke"
)

type FPClient struct {
	cli  *http.Client
	url  string
	crid uint32
}

func (c *FPClient) List() error {
	crid := atomic.AddUint32(&c.crid, 1)
	req, err := http.NewRequest(http.MethodGet, c.url+fplistpath, nil)
	if err != nil {
		log.Printf("[CRID-%d] cannot create get request: %v", crid, err)
		return err
	}

	res, err := c.cli.Do(req)
	if err != nil {
		log.Printf("[CRID-%d] error in sending http request: %v", crid, err)
		return err
	}

	body, err := io.ReadAll(res.Body)
	if err != nil {
		log.Printf("[CRID-%d] cannot read response body: %v", crid, err)
		return err
	}
	fmt.Printf("[CRID-%d] Response: %s", crid, body)
	return nil
}

func (c *FPClient) Create(name, kind, image string, concurrency, budget int) error {
	crid := atomic.AddUint32(&c.crid, 1)

	// construct request
	req_body := fpreq.FPCreateRequest{Name: name, Kind: kind, Image: image, Concurrency: concurrency, Budget: budget}
	req_bytes, err := json.Marshal(req_body)
	if err != nil {
		log.Printf("[CRID-%d] failed to construct create request: %v", crid, err)
		return err
	}
	bodyReader := bytes.NewReader(req_bytes)

	req, err := http.NewRequest(http.MethodPost, c.url+fpcreatepath, bodyReader)
	if err != nil {
		log.Printf("[CRID-%d] cannot create get request: %v", crid, err)
		return err
	}

	res, err := c.cli.Do(req)
	if err != nil {
		log.Printf("[CRID-%d] error in sending http request: %v", crid, err)
		return err
	}

	body, err := io.ReadAll(res.Body)
	if err != nil {
		log.Printf("[CRID-%d] cannot read response body: %v", crid, err)
		return err
	}
	fmt.Printf("[CRID-%d] Response: %s", crid, body)
	return nil
}

func (c *FPClient) Delete(name string) error {
	crid := atomic.AddUint32(&c.crid, 1)

	// construct request
	req_body := fpreq.FPDeleteRequest{Name: name}
	req_bytes, err := json.Marshal(req_body)
	if err != nil {
		log.Printf("[CRID-%d] failed to construct delete request: %v", crid, err)
		return err
	}
	bodyReader := bytes.NewReader(req_bytes)

	req, err := http.NewRequest(http.MethodPost, c.url+fpdeletepath, bodyReader)
	if err != nil {
		log.Printf("[CRID-%d] cannot create get request: %v", crid, err)
		return err
	}

	res, err := c.cli.Do(req)
	if err != nil {
		log.Printf("[CRID-%d] error in sending http request: %v", crid, err)
		return err
	}

	body, err := io.ReadAll(res.Body)
	if err != nil {
		log.Printf("[CRID-%d] cannot read response body: %v", crid, err)
		return err
	}
	fmt.Printf("[CRID-%d] Response: %s", crid, body)
	return nil
}

func (c *FPClient) Invoke(name string, ireq *fpreq.InvokeRequest) error {
	crid := atomic.AddUint32(&c.crid, 1)

	// construct request
	req_body := fpreq.FPInvokeRequest{Name: name, Req: *ireq}
	req_bytes, err := json.Marshal(req_body)

	if err != nil {
		log.Printf("[CRID-%d] failed to construct invoke request: %v", crid, err)
		return err
	}
	bodyReader := bytes.NewReader(req_bytes)

	req, err := http.NewRequest(http.MethodPost, c.url+fpinvokepath, bodyReader)
	if err != nil {
		log.Printf("[CRID-%d] cannot create get request: %v", crid, err)
		return err
	}

	res, err := c.cli.Do(req)
	if err != nil {
		log.Printf("[CRID-%d] error in sending http request: %v", crid, err)
		return err
	}

	body, err := io.ReadAll(res.Body)
	if err != nil {
		log.Printf("[CRID-%d] cannot read response body: %v", crid, err)
		return err
	}
	log.Printf("[CRID-%d] [%v] Response: %s", crid, ireq.ModelType, body)

	if res.StatusCode != http.StatusOK {
		return errors.New("fn packer returned error")
	}
	return nil
}
