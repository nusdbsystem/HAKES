/*
 * Copyright 2024 The HAKES Authors
 * Copyright 2017 Dgraph Labs, Inc. and Contributors
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

package kv

import (
	"encoding/hex"
	"math/rand"
	"time"

	"hakes-store/hakes-store/io"
	"hakes-store/hakes-store/table"

	"github.com/dgraph-io/badger/v3/y"
	"github.com/pkg/errors"
)

func (s *levelsController) validate() error {
	for _, l := range s.levels {
		if err := l.validate(); err != nil {
			return y.Wrap(err, "Levels Controller")
		}
	}
	return nil
}

// Check does some sanity check on one level of data or in-memory index.
func (s *levelHandler) validate() error {
	if s.level == 0 {
		return nil
	}

	s.RLock()
	defer s.RUnlock()
	numTables := len(s.tables)
	for j := 1; j < numTables; j++ {
		if j >= len(s.tables) {
			return errors.Errorf("Level %d, j=%d numTables=%d", s.level, j, numTables)
		}

		if y.CompareKeys(s.tables[j-1].Biggest(), s.tables[j].Smallest()) >= 0 {
			return errors.Errorf(
				"Inter: Biggest(j-1)[%v] \n%s\n vs Smallest(j)[%v]: \n%s\n: "+
					"level=%d j=%d numTables=%d",
				s.tables[j-1].ID(), hex.Dump(s.tables[j-1].Biggest()), s.tables[j].ID(),
				hex.Dump(s.tables[j].Smallest()), s.level, j, numTables)
		}

		if y.CompareKeys(s.tables[j].Smallest(), s.tables[j].Biggest()) > 0 {
			return errors.Errorf(
				"Intra: \n%s\n vs \n%s\n: level=%d j=%d numTables=%d",
				hex.Dump(s.tables[j].Smallest()), hex.Dump(s.tables[j].Biggest()), s.level, j, numTables)
		}
	}
	return nil
}

func (s *levelsController) reserveFileID() string {
	return s.kv.idAlloc.getNext()
}

func getIDMap(cssCli io.CSSCli) map[string]struct{} {
	idMap := make(map[string]struct{})
	cssIds, err := cssCli.List()
	y.Check(err)
	for _, cssId := range cssIds {
		fileID, ok := table.ParseFileID(cssId)
		if !ok {
			continue
		}
		idMap[fileID] = struct{}{}
	}
	return idMap
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
