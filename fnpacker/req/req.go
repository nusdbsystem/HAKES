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

package req

import (
	"encoding/json"
)

type InvokeRequest struct {
	ModelType       string `json:"model_type"`
	UserId          string `json:"user_id"`
	KeyServiceAddr  string `json:"key_service_address"`
	KeyServicePort  int    `json:"key_service_port"`
	EncryptedSample string `json:"encrypted_sample"`
}

func (r *InvokeRequest) Load(content []byte) (err error) {
	return json.Unmarshal(content, r)
}

type FPInvokeRequest struct {
	Name string        `json:"name"`
	Req  InvokeRequest `json:"request"`
}

func (r *FPInvokeRequest) Load(content []byte) (err error) {
	return json.Unmarshal(content, r)
}

type FPCreateRequest struct {
	Name        string `json:"name"`
	Kind        string `json:"kind"`
	Image       string `json:"image"`
	Concurrency int    `json:"concurrency"`
	Budget      int    `json:"budget"`
}

func (r *FPCreateRequest) Load(content []byte) (err error) {
	return json.Unmarshal(content, r)
}

type FPDeleteRequest struct {
	Name string `json:"name"`
}

func (r *FPDeleteRequest) Load(content []byte) (err error) {
	return json.Unmarshal(content, r)
}
