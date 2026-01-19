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
	"math"

	"github.com/pkg/errors"
)

const (
	// ValueThresholdLimit is the maximum permissible value of opt.ValueThreshold.
	ValueThresholdLimit = math.MaxUint16 - 16 + 1
)

var errFillTables = errors.New("Unable to fill tables")

var (
	// ErrValueLogSize is returned when opt.ValueLogFileSize option is not within the valid
	// range.
	ErrValueLogSize = errors.New("Invalid ValueLogFileSize, must be in range [1MB, 2GB)")

	// ErrKeyNotFound is returned when key isn't found on a txn.Get.
	ErrKeyNotFound = errors.New("Key not found")

	// ErrTxnTooBig is returned if too many writes are fit into a single batch.
	ErrBatchTooBig = errors.New("Batch is too big to fit into one request")

	// ErrBlockedWrites is returned if the user called DropAll. During the process of dropping all
	// data from Badger, we stop accepting new writes, by returning this error.
	ErrBlockedWrites = errors.New("Writes are blocked, possibly due to DropAll or Close")

	// ErrDBClosed is returned when a get operation is performed after closing the DB.
	ErrDBClosed = errors.New("DB Closed")
)
