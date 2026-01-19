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

package io

import (
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"

	"github.com/dgraph-io/ristretto/z"
)

var (
	ErrLogNonExist = errors.New("durable log non-exist")
)

type DLog interface {
	// return the id
	Name() string
	// append new contents to the log
	Append(b []byte) (int, error)
	// sync ensures all appends are persisted
	Sync() error
	// seal ensures no new appends will be accepted after return
	// Seal() error
	// close is an explicit signal to clear any resources DLog holds, before dlogHandler drop it.
	Close() error
	// Size() (int64, error) // length of the log
	Size() uint32 // length of the log

	// create an independent reader that does not modify the state of DLog
	NewReader(offset int) io.Reader
}

// Durable log service log client connects to a service that allocate durable log.
type DLogHandler interface {
	// setup dls client
	Connect(config string) error
	// open an existing log resource
	OpenLog(name string, maxSz int) (DLog, error)
	// open a new log resources potentially avoid fetching that existing CRL needs
	OpenNewLog(name string, maxSz int) (DLog, error)
	// Drop the log
	Drop(name string) error
	// list objects
	List() ([]string, error)
	// disconnect from dls
	Disconnect() error
	// from hakes-store replica trigger
	IsSwitchLogRequired() bool
	// report stats
	GetStats() string
}

var _ io.Reader = (*LFLogReader)(nil)

type LFLogReader struct {
	reader io.Reader
	stats  *FSStats
}

func (r *LFLogReader) Read(p []byte) (n int, err error) {
	n, err = r.reader.Read(p)
	r.stats.AddReadSize(len(p))
	return
}

type LFLog struct {
	*z.MmapFile
	name       string
	tailOffset uint32
	stats      *FSStats
}

func (l *LFLog) Name() string {
	return l.name
}

func (l *LFLog) Append(b []byte) (int, error) {
	l.stats.AddWriteSize(len(b))
	written := copy(l.Data[l.tailOffset:], b)
	l.tailOffset += uint32(written)
	return written, nil
}

func (l *LFLog) Close() error {
	// 0 means no truncation
	// it will be deleted anw.
	return l.MmapFile.Close(0)
}

func (l *LFLog) Size() uint32 {
	return l.tailOffset
}

func (l *LFLog) NewReader(offset int) io.Reader {
	return l.MmapFile.NewReader(offset)
}

var _ DLogHandler = (*FDLogHandler)(nil)

type FDLogHandler struct {
	// directory to store files
	dir   string
	stats FSStats
}

func (c *FDLogHandler) GetStats() string {
	return c.stats.Encode()
}

func (c *FDLogHandler) Connect(config string) error {
	c.dir = config
	return createDirIfNotExist(c.dir)
}

func (c *FDLogHandler) Disconnect() error {
	return nil
}

func (c *FDLogHandler) openFile(name string, flag, maxSz int) (DLog, error) {
	path := filepath.Join(c.dir, name)
	if fp, err := z.OpenMmapFile(path, flag, maxSz); fp == nil {
		return nil, err
	} else {
		// for new file a new file error will be returned
		return &LFLog{fp, name, 0, &c.stats}, err
	}
}

func (c *FDLogHandler) OpenLog(name string, maxSz int) (DLog, error) {
	return c.openFile(name, os.O_RDWR, maxSz)
}

func (c *FDLogHandler) OpenNewLog(name string, maxSz int) (DLog, error) {
	// return c.openFile(name, os.O_CREATE|os.O_RDWR, maxSz)
	if f, err := c.openFile(name, os.O_CREATE|os.O_RDWR, maxSz); err == z.NewFile {
		return f, nil
	}
	log.Printf("new file %v already exists (will be recreated)", name)
	// delete and recreated
	c.Drop(name)
	if f, err := c.openFile(name, os.O_CREATE|os.O_RDWR, maxSz); err != z.NewFile {
		return nil, fmt.Errorf("expect new file")
	} else {
		return f, nil
	}
}

func (c *FDLogHandler) List() ([]string, error) {
	fileInfos, err := os.ReadDir(c.dir)
	if err != nil {
		return nil, err
	}
	ids := make([]string, 0, len(fileInfos))
	for _, info := range fileInfos {
		if info.IsDir() {
			continue
		}
		ids = append(ids, info.Name())
	}
	return ids, nil
}

func (c *FDLogHandler) Drop(name string) error {
	path := filepath.Join(c.dir, name)
	return os.Remove(path)
}

func (c *FDLogHandler) IsSwitchLogRequired() bool {
	// always return false in local setting
	return false
}
