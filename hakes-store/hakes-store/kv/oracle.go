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
	"sync"

	"github.com/dgraph-io/badger/v3/y"
	"github.com/dgraph-io/ristretto/z"
)

type oracle struct {
	sync.Mutex              // For nextTxnTs and commits.
	readMark   *y.WaterMark // Used by DB.
	closer     *z.Closer
}

func newOracle(opt Options) *oracle {
	orc := &oracle{
		// WaterMarks must be 64-bit aligned for atomic package, hence we must use pointers here.
		// See https://golang.org/pkg/sync/atomic/#pkg-note-BUG.
		readMark: &y.WaterMark{Name: "badger.PendingReads"},
		closer:   z.NewCloser(1),
	}
	orc.readMark.Init(orc.closer)
	return orc
}

func (o *oracle) Stop() {
	o.closer.SignalAndWait()
}

func (o *oracle) incrementNextTs() {
	o.Lock()
	defer o.Unlock()
}

func (o *oracle) discardAtOrBelow() uint64 {
	return o.readMark.DoneUntil()
}
