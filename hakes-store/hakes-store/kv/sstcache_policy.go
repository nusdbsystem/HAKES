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

package kv

import "log"

type SstCachePolicy interface {
	UseSstCache(targetLevel, baseLevel int) bool
}

var _ SstCachePolicy = (*levelSstCachePolicy)(nil)

// the default SstCache policy: based on level
type levelSstCachePolicy struct {
	threadholdChecker func(targetLevel, baseLevel int) bool
}

func NewLevelSstCachePolicy(maxLevels, levelSizeMultiplier int) *levelSstCachePolicy {
	// only level holding 10% data cached
	var offsetFromLast int
	if levelSizeMultiplier >= 10 {
		offsetFromLast = 1
	} else if levelSizeMultiplier >= 3 {
		offsetFromLast = 2
	} else {
		offsetFromLast = 3
	}
	log.Printf("Level SSTCache Policy used %d of %d level will be cached", maxLevels-offsetFromLast, maxLevels)

	return &levelSstCachePolicy{
		threadholdChecker: func(targetLevel, baseLevel int) bool {
			// if the target level is current baselevel or a level that data from top to that level accounts to roughly 10% of total data
			return targetLevel == baseLevel || targetLevel < maxLevels-offsetFromLast
		},
	}
}

func (scp *levelSstCachePolicy) UseSstCache(targetLevel, baseLevel int) bool {
	return scp.threadholdChecker(targetLevel, baseLevel)
}
