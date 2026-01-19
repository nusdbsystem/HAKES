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

type CSSConfig struct {
	Type string `yaml:"type"`
	Path string `yaml:"path"`
}

type ManifestHandlerConfig struct {
	Type string `yaml:"type"`
	Path string `yaml:"path"`
}

type DLogHandlerConfig struct {
	Type string `yaml:"type"`
	Path string `yaml:"path"`
}

type RemoteCompactionConfig struct {
	Use             bool   `yaml:"use"`
	Type            string `yaml:"type"`
	StoreDaemonIP   string `yaml:"store-daemon-ip"`
	StoreDaemonPort int    `yaml:"store-daemon-port"`
}

type SstCacheConfig struct {
	Type       string    `yaml:"type"`        // local or remote
	RemoteIP   string    `yaml:"remote-ip"`   // GB
	RemotePort int       `yaml:"remote-port"` // GB
	Capacity   int       `yaml:"capacity"`    // GB, only for local
	Css        CSSConfig `yaml:"css-config"`
}
type LSMConfig struct {
	SyncWrites          bool    `yaml:"sync-writes"`
	LogType             string  `yaml:"log-type"`
	LogPath             string  `yaml:"log-path"`
	CompressionType     string  `yaml:"compression-type"`
	MemtableSize        int     `yaml:"memtable-size"`
	BasetableSize       int     `yaml:"basetable-size"`
	BaselevelSize       int     `yaml:"baselevel-size"`
	LevelsizeMultiplier int     `yaml:"levelsize-multiplier"`
	TablesizeMultiplier int     `yaml:"tablesize-multiplier"`
	Maxlevel            int     `yaml:"maxlevel"`
	ValueThreshold      int     `yaml:"value-threshold"`
	NumMemtable         int     `yaml:"num-memtable"`
	BlockSize           int     `yaml:"block-size"`
	BloomFPR            float64 `yaml:"bloom-fpr"`
	BlockCacheSize      int     `yaml:"block-cache-size"`
	IndexCacheSize      int     `yaml:"index-cache-size"`
	NumLevelZeroTable   int     `yaml:"num-level-zero-table"`
	NumLZTStall         int     `yaml:"num-level-zero-table-stall"`
	NumCompactor        int     `yaml:"num-compactor"`
	LMaxCompaction      bool    `yaml:"lmax-compaction"`
	DisableWAL          bool    `yaml:"disableWAL"`
	PrefetchMode        string  `yaml:"prefetch-mode"`
	PrefetchSize        int     `yaml:"prefetch-size"`
	LogCompactionStats  bool    `yaml:"log-compaction-stats"`
	LogFlushStats       bool    `yaml:"log-flush-stats"`
}

type HakesKVConfig struct {
	Name string                 `yaml:"name"`
	Css  CSSConfig              `yaml:"css-config"`
	Mh   ManifestHandlerConfig  `yaml:"manifest-handler"`
	Dlh  DLogHandlerConfig      `yaml:"dlog-handler"`
	Rc   RemoteCompactionConfig `yaml:"compaction-scheduler"`
	Sc   SstCacheConfig         `yaml:"sstcache-config"`
	Lsm  LSMConfig              `yaml:"lsm-config"`
}

func ParseHakesKVConfig(filename string) (*HakesKVConfig, error) {
	// Read the configuration file contents into a byte slice.
	configData, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	// Parse the YAML-encoded configuration data into a Config struct.
	var conf HakesKVConfig
	err = yaml.Unmarshal(configData, &conf)
	if err != nil {
		return nil, err
	}

	return &conf, nil
}
