//go:build !windows && !plan9
// +build !windows,!plan9

/*
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

package io

import (
	"os"

	"github.com/dgraph-io/badger/v3/y"
)

// openDir opens a directory for syncing.
func openDir(path string) (*os.File, error) { return os.Open(path) }

// When you create or delete a file, you have to ensure the directory entry for the file is synced
// in order to guarantee the file is visible (if the system crashes). (See the man page for fsync,
// or see https://github.com/coreos/etcd/issues/6368 for an example.)
func syncDir(dir string) error {
	f, err := openDir(dir)
	if err != nil {
		return y.Wrapf(err, "While opening directory: %s.", dir)
	}

	err = f.Sync()
	closeErr := f.Close()
	if err != nil {
		return y.Wrapf(err, "While syncing directory: %s.", dir)
	}
	return y.Wrapf(closeErr, "While closing directory: %s.", dir)
}

func checkDirExists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return true, err
}

func createDirIfNotExist(path string) error {
	dirExists, err := checkDirExists(path)
	if err != nil {
		return y.Wrapf(err, "Invalid Dir: %q", path)
	}
	if !dirExists {
		err = os.MkdirAll(path, 0700)
		if err != nil {
			return y.Wrapf(err, "Error Creating Dir: %q", path)
		}
	}
	return nil
}
