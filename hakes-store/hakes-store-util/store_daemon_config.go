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

package hakesstoreutil

import (
	"os"

	"gopkg.in/yaml.v2"
)

type StoreDaemonConfig struct {
	ID             string         `yaml:"id"`
	NetName        string         `yaml:"net-name"`
	NetBandwidth   int            `yaml:"net-bandwidth"`
	PrefetchMode   string         `yaml:"prefetch-mode"`
	PrefetchSize   int            `yaml:"prefetch-size"`
	Css            CSSConfig      `yaml:"css-config"`
	Sc             SstCacheConfig `yaml:"sstcache-config"`
	Peers          []string       `yaml:"store-daemon-peers"`
	Reserved       float32        `yaml:"reserved"`         // resource percentage reserved for local processing
	LocalThreshold float32        `yaml:"local-threshold"`  // local threshold above which favors trying remote scheduling
	LambdaComp     string         `yaml:"lambda-compactor"` // lambda function name
	NumCompactor   int            `yaml:"num-compactor"`    // number of concurrent compaction job scheduling
}

func ParseStoreDaemonConfig(filename string) (*StoreDaemonConfig, error) {
	configData, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var conf StoreDaemonConfig
	err = yaml.Unmarshal(configData, &conf)
	if err != nil {
		return nil, err
	}
	return &conf, nil
}
