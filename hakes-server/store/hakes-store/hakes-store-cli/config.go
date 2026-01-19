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

package hakesstorecli

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v2"
)

type Config struct {
	Root   string   `yaml:"root"`
	Region []string `yaml:"region"`
	Zk     []string `yaml:"zk"`
}

func ParseConfigFile(filename string) (*Config, error) {
	// Read the YAML file
	cfgData, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read rs client config %s: %w", filename, err)
	}

	// Create a new instance of Config struct
	config := &Config{}

	// Unmarshal the YAML into the Config struct
	err = yaml.Unmarshal(cfgData, config)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal rs client config %s: %w", filename, err)
	}

	return config, nil
}
