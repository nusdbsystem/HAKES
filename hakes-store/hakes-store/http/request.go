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
	"encoding/hex"
)

type HttpRequest struct {
	Type   string   `json:"type"`
	Keys   []string `json:"keys"`
	Values []string `json:"values"`
}

type HttpResponse struct {
	Status bool     `json:"status"`
	Values []string `json:"values"`
}

func HttpPayloadEncode(payload []byte) string {
	return hex.EncodeToString(payload)
}

func HttpPayloadDecode(payload string) ([]byte, error) {
	return hex.DecodeString(payload)
}
