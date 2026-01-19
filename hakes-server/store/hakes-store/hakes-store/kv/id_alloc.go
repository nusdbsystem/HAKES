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

import (
	"fmt"
	"strconv"
	"strings"
	"sync/atomic"
)

// Abstraction for uid allocation

type IdAllocator interface {
	// allocate a new id.
	getNext() string
	// allocator will be forbidden to output allocated ids.
	updateWithAllocated(string)
	// get the prefix to be used by remote compactors.
	getPrefix() string
}

var _ IdAllocator = (*PrefixedIdAllocator)(nil)
var _ IdAllocator = (*EpochPrefixIdAllocator)(nil)

// when prefix id allocator is used, make sure different prefix is used when db is reopened
type PrefixedIdAllocator struct {
	prefix    string
	lastAlloc atomic.Uint64
}

func NewPrefixedIdAllocator(prefix string) *PrefixedIdAllocator {
	return &PrefixedIdAllocator{
		prefix: prefix,
	}
}

func (a *PrefixedIdAllocator) getPrefix() string {
	return a.prefix
}

func (a *PrefixedIdAllocator) getNext() string {
	allocated := a.lastAlloc.Add(1)
	return fmt.Sprintf("%v%06d", a.prefix, allocated)
}

func (a *PrefixedIdAllocator) updateWithAllocated(allocated string) {
	if !strings.HasPrefix(allocated, a.prefix) {
		return
	}
	in, err := strconv.Atoi(strings.TrimPrefix(allocated, a.prefix))
	if err != nil {
		return
	}
	in64 := uint64(in)
	for {
		cur := a.lastAlloc.Load()
		if cur > in64 || a.lastAlloc.CompareAndSwap(cur, in64) {
			break
		}
	}
}

// A special type of prefix id allocator.
// It assumes tables have a prefix in the form of "ep%d-".
// The epoch number will be automatically incremented during reopen
// Same epoch is used for the duration of a hakeskv running.
type EpochPrefixIdAllocator struct {
	epochNum  atomic.Uint64
	lastAlloc atomic.Uint64
}

func NewEpochPrefixIdAllocator() *EpochPrefixIdAllocator {
	return &EpochPrefixIdAllocator{}
}

func NewEpochPrefixIdAllocatorWithEpoch(e uint64) *EpochPrefixIdAllocator {
	ret := &EpochPrefixIdAllocator{}
	ret.epochNum.Store(e)
	return ret
}

func buildEpochPrefix(epochNum uint64) string {
	return fmt.Sprintf("ep%d-", epochNum)
}

func (a *EpochPrefixIdAllocator) getPrefix() string {
	return buildEpochPrefix(a.epochNum.Load())
}

func (a *EpochPrefixIdAllocator) getNext() string {
	allocated := a.lastAlloc.Add(1)
	return fmt.Sprintf("%v%06d", buildEpochPrefix(a.epochNum.Load()), allocated)
}

func (a *EpochPrefixIdAllocator) updateWithAllocated(allocated string) {
	if len(allocated) < 4 || allocated[:2] != "ep" {
		return
	}
	epochPrefixLen := strings.IndexRune(allocated, '-')
	if epochPrefixLen == -1 {
		return
	}
	epochNum, err := strconv.ParseUint(allocated[2:epochPrefixLen], 10, 64)
	if err != nil {
		return
	}
	nextEpoch := epochNum + 1
	for {
		cur := a.epochNum.Load()
		if cur >= nextEpoch || a.epochNum.CompareAndSwap(cur, nextEpoch) {
			break
		}
	}
}
