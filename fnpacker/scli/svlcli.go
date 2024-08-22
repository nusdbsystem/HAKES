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

package scli

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"fnpacker/req"

	"github.com/apache/openwhisk-client-go/whisk"
)

// Serverless Client interface
type InvokeResponse struct {
	ExecTime uint64
	Msg      []byte
}

type ServerlessClient interface {
	Init() error
	CreateAction(name, kind, image string, concurrency, budget int) error
	InvokeAction(name string, req *req.InvokeRequest, rid uint64) (string, string, error)
	GetResult(rid uint64, resp_id string) (InvokeResponse, error)
	DeleteAction(name string) error
}

// Openwhisk client
type OwClient struct {
	cli *whisk.Client
}

func OwBodySerialize(r *req.InvokeRequest) (body map[string]interface{}) {
	return map[string]interface{}{
		"model_type":          r.ModelType,
		"user_id":             r.UserId,
		"key_service_address": r.KeyServiceAddr,
		"key_service_port":    r.KeyServicePort,
		"encrypted_sample":    r.EncryptedSample,
	}
}

func (oc *OwClient) Init() error {
	var err error
	oc.cli, err = whisk.NewClient(http.DefaultClient, nil)
	return err
}

func (oc *OwClient) InitWithConfig(ow_address, ow_port, ow_auth string) error {
	var err error
	c := whisk.Config{Host: ow_address + ":" + ow_port, AuthToken: ow_auth}
	oc.cli, err = whisk.NewClient(http.DefaultClient, &c)
	return err
}

// thread safe
func (oc *OwClient) CreateAction(n, k, im string, c, b int) error {
	timeout := 300000
	action := whisk.Action{
		Name:      n,
		Namespace: "_",
		Limits:    &whisk.Limits{Concurrency: &c, Memory: &b, Timeout: &timeout},
		Exec:      &whisk.Exec{Kind: k, Image: im},
	}
	_, resp, err := oc.cli.Actions.Insert(&action, false)
	log.Printf("create response (at %v): %v", time.Now().UnixMicro(), resp)
	return err
}

// thread safe
func (oc *OwClient) InvokeAction(n string, r *req.InvokeRequest, rid uint64) (string, string, error) {
	log.Printf("invoke (%v at %v) for %v", rid, time.Now().UnixMicro(), r.ModelType)
	wskresp, resp, err := oc.cli.Actions.Invoke(n, OwBodySerialize(r), false, false)
	log.Printf("invoke response (%v at %v): %v", rid, time.Now().UnixMicro(), resp)
	wskresp_map := wskresp.(map[string]interface{})
	resp_id := fmt.Sprintf("%v", wskresp_map["activationId"])
	return resp_id, fmt.Sprintf("invoke response: %v", resp), err
}

func (oc *OwClient) GetResult(rid uint64, resp_id string) (InvokeResponse, error) {
	wskresp, resp, err := oc.cli.Activations.Get(resp_id)
	retry_count := 60
	back_off := 2
	for err != nil {
		retry_count--
		if resp.Status != "404 Not Found" {
			break
		}
		wskresp, resp, err = oc.cli.Activations.Get(resp_id)
		// loop to fetch. The activation will return not found before it is finished.
		if retry_count <= 0 {
			log.Printf("invoke response (%v at %v): exhausted retries", rid, time.Now().UnixMicro())
			break
		}
		time.Sleep(time.Duration(back_off) * time.Millisecond)
		back_off = back_off * 2
	}
	log.Printf("invoke result (%v at %v)", rid, time.Now().UnixMicro())
	// return InvokeResponse{ExecTime: uint64(wskresp.Duration * 1000), Msg: fmt.Sprintf("ow-response: %v", resp)}, err
	if err != nil {
		return InvokeResponse{}, err
	} else if json_res, err := json.Marshal(wskresp.Result); err != nil {
		return InvokeResponse{}, err
	} else {
		return InvokeResponse{ExecTime: uint64(wskresp.Duration * 1000), Msg: json_res}, err
	}
}

// thread safe
func (oc *OwClient) DeleteAction(n string) error {
	resp, err := oc.cli.Actions.Delete(n)
	log.Printf("delete response (at %v): %v", time.Now().UnixMicro(), resp)
	return err
}
