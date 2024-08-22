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
	"errors"
	"fmt"
	"log"
	"math"
	"strconv"
)

var (
	ErrNotSeqZnode      = errors.New("not a sequence znode name")
	ErrPrevNodeNotFound = errors.New("prev sequence znode not found")
)

func splitEmphemeralNodeName(znode string) (string, int, error) {
	size := len(znode)
	if size < 10 {
		return "", -1, fmt.Errorf("not a emphermeral node")
	}
	seq, err := strconv.Atoi(znode[size-10:])
	return znode[:size-10], seq, err
}

func FindSmallestSeqNode(znodes []string, prefixFilter string) (int, string, error) {
	minSeq := math.MaxInt
	var minZnode string
	for _, r := range znodes {
		prefix, seq, err := splitEmphemeralNodeName(r)
		log.Printf("prefix %v, seq: %d", prefix, seq)
		if err != nil || prefix != prefixFilter {
			// skip the current node
			continue
		}
		if seq < minSeq {
			minSeq = seq
			minZnode = r
		}
	}
	if len(minZnode) == 0 {
		return -1, "", fmt.Errorf("smallest sequence znode not found")
	}
	return minSeq, minZnode, nil
}

func FindPrevSeqNode(znodes []string, cur string) (int, string, error) {

	curPrefix, curSeq, err := splitEmphemeralNodeName(cur)
	if err != nil {
		return -1, "", fmt.Errorf("input is not a sequence znode")
	}
	prevSeq := math.MinInt
	var prevZnode string
	for _, r := range znodes {
		prefix, seq, err := splitEmphemeralNodeName(r)
		if err != nil || prefix != curPrefix { // skip the current node
			continue
		}
		if seq < curSeq && seq > prevSeq {
			prevSeq = seq
			prevZnode = r
		}
	}
	if len(prevZnode) == 0 {
		return -1, "", ErrPrevNodeNotFound
	}
	return prevSeq, prevZnode, nil
}
