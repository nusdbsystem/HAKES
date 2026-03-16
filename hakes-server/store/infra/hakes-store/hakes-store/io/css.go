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

// cloud storage service abstraction that exposes an append-able blob APIs.

package io

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"

	"github.com/dgraph-io/ristretto/z"
)

type CSSF interface {
	// return the id
	Name() string
	// should be thread-safe
	Bytes(off, sz int) ([]byte, error)
	Sync() error
	Delete() error
	Close(maxSz int64) error // input -1 if truncation undesirable

	// stats used to manage sst
	// file size is used to read from the back
	Size() int64 // length in bytes for regular files; system-dependent for others
	// mod time is used for level controller such that newly created tables are not used for flushing/compaction
	ModTime() time.Time // modification time

	// try to remove this when cleaned upper logics
	// i dont think it is of any usage for cloud storage
	ReadAll() []byte
	Write([]byte) int
}

/*
 * The client maintain the connection to the storage service
 * 	and open the resources for applications to use
 */
type CSSCli interface {
	// connect to css
	Connect(config string) error
	// Open an existing file
	OpenFile(name string, maxSz int) (CSSF, error) // maxSz 0 for openning it at current size, > 0 for truncation
	// Open a new file for persistence
	OpenNewFile(name string, maxSz int) (CSSF, error)
	// sync any client local info to css
	Sync() error
	// disconnect from css
	Disconnect() error
	// estimate the storage size // only for user api to get db size.
	EstimateSize() (int64, int64, error)
	// list objects
	List() ([]string, error)
	// delete the object
	Delete(name string) error
	// export the config
	Config() string
	// report stats
	GetStats() string
}

type LFSF struct {
	*z.MmapFile
	name     string
	writeOff int
	stats    *FSStats
}

func (f *LFSF) Name() string {
	return f.name
}

func (f *LFSF) Size() int64 {
	fileInfo, err := f.Fd.Stat()
	if err != nil {
		return 0
	}
	return fileInfo.Size()
}

func (f *LFSF) ModTime() time.Time {
	fileInfo, err := f.Fd.Stat()
	if err != nil {
		return time.Now()
	}
	return fileInfo.ModTime()
}

func (f *LFSF) Bytes(off, sz int) ([]byte, error) {
	data, err := f.MmapFile.Bytes(off, sz)
	f.stats.AddReadSize(len(data))
	return data, err
}

func (f *LFSF) ReadAll() []byte {
	f.stats.AddReadSize(len(f.Data))
	return f.Data
}

func (f *LFSF) Write(data []byte) int {
	f.stats.AddWriteSize(len(data))
	written := copy(f.Data[f.writeOff:], data)
	f.writeOff += written
	return written
}

var _ CSSCli = (*FSCli)(nil)

type FSStats struct {
	WriteSize atomic.Uint64
	ReadSize  atomic.Uint64
}

func (s *FSStats) AddReadSize(sz int) {
	if s == nil {
		return
	}
	s.ReadSize.Add(uint64(sz))
}

func (s *FSStats) AddWriteSize(sz int) {
	if s == nil {
		return
	}
	s.WriteSize.Add(uint64(sz))
}

func (s *FSStats) Encode() string {
	if s == nil {
		return ""
	}
	return fmt.Sprintf("FS: Written: %d B, Read %d B", s.WriteSize.Load(), s.ReadSize.Load())
}

type FSCli struct {
	// directory to store files
	dir   string
	stats FSStats
}

func (c *FSCli) GetStats() string {
	return c.stats.Encode()
}

func (c *FSCli) Config() string {
	return c.dir
}

// config to FSCli is the directory to store data.
func (c *FSCli) Connect(config string) error {
	c.dir = config
	return createDirIfNotExist(c.dir)
}

func (c *FSCli) Disconnect() error {
	return nil
}

func (c *FSCli) openFile(name string, flag, maxSz int) (CSSF, error) {
	path := filepath.Join(c.dir, name)
	fp, err := z.OpenMmapFile(path, flag, maxSz)
	return &LFSF{fp, name, 0, &c.stats}, err
}

func (c *FSCli) OpenFile(name string, maxSz int) (CSSF, error) {
	return c.openFile(name, os.O_RDWR, maxSz)
}

func (c *FSCli) OpenNewFile(name string, maxSz int) (CSSF, error) {
	if f, err := c.openFile(name, os.O_CREATE|os.O_RDWR, maxSz); err == z.NewFile {
		return f, nil
	}
	log.Printf("new file %v already exists (will be recreated)", name)
	// delete and recreated
	c.Delete(name)
	if f, err := c.openFile(name, os.O_CREATE|os.O_RDWR, maxSz); err != z.NewFile {
		return nil, fmt.Errorf("expect new file")
	} else {
		return f, nil
	}
}

func (c *FSCli) Sync() error {
	return syncDir(c.dir)
}

func (c *FSCli) EstimateSize() (int64, int64, error) {
	var lsmSize, vlogSize int64
	err := filepath.Walk(c.dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		ext := filepath.Ext(path)
		switch ext {
		case ".sst":
			lsmSize += info.Size()
		case ".vlog":
			vlogSize += info.Size()
		}
		return nil
	})
	return lsmSize, vlogSize, err
}

func (c *FSCli) List() ([]string, error) {
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

func (c *FSCli) Delete(name string) error {
	path := filepath.Join(c.dir, name)
	return os.Remove(path)
}
