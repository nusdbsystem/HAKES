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

package cloud

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"path/filepath"
	"sync/atomic"
	"time"

	kvio "hakes-store/hakes-store/io"

	"github.com/aws/aws-sdk-go-v2/aws"
	awshttp "github.com/aws/aws-sdk-go-v2/aws/transport/http"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/dgraph-io/badger/v3/y"
)

var (
	errS3Connection = errors.New("failed to connect to S3")
)

type s3svc interface {
	GetObject(bucket, name string) ([]byte, error)
	GetObjectRange(bucket, name string, off, sz int) ([]byte, error)
	PutObject(bucket, name string, data []byte) error
	DeleteObject(bucket, name string) error
	HeadObject(bucket, name string) (S3ObjectMeta, error)
	ListObject(bucket string, consume func(types.Object)) error
}

type S3ObjectMeta struct {
	len          int64
	lastModified time.Time
}

type S3Stats struct {
	PutCount    atomic.Int32
	GetCount    atomic.Int32
	DeleteCount atomic.Int32
	HeadCount   atomic.Int32
	ListCount   atomic.Int32
}

func (s *S3Stats) Encode() string {
	if s == nil {
		return ""
	}
	return fmt.Sprintf("S3 client: Put: %d, Get: %d, Delete: %d, Head: %d, List %d", s.PutCount.Load(), s.GetCount.Load(), s.DeleteCount.Load(), s.HeadCount.Load(), s.ListCount.Load())
}

// wrapper class s3 client
type S3C struct {
	cli   *s3.Client
	stats S3Stats
}

func ConnectToS3() (*S3C, error) {
	customCli := awshttp.NewBuildableClient().WithTransportOptions(func(tr *http.Transport) {
		tr.MaxIdleConnsPerHost = 128
	})
	cfg, err := config.LoadDefaultConfig(context.TODO(), config.WithHTTPClient(customCli))
	if err != nil {
		log.Println(err)
		return nil, errS3Connection
	}
	cfg.Region = awsRegion
	defer log.Println("set up s3 client")
	return &S3C{cli: s3.NewFromConfig(cfg)}, nil
}

func (c *S3C) GetObject(bucket, name string) ([]byte, error) {
	c.stats.GetCount.Add(1)
	r, err := c.cli.GetObject(context.TODO(), &s3.GetObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(name),
	})
	if err != nil {
		log.Printf("s3 read all failed: %v\n", err)
		return nil, err
	}
	if content, err := io.ReadAll(r.Body); err != nil {
		log.Printf("s3 read all failed: %v\n", err)
		return nil, err
	} else {
		return content, nil
	}
}

func (c *S3C) GetObjectRange(bucket, name string, off, sz int) ([]byte, error) {
	c.stats.GetCount.Add(1)
	// fetch from s3
	r, err := c.cli.GetObject(context.TODO(), &s3.GetObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(name),
		Range:  aws.String(genS3Range(off, sz)),
	})
	if err != nil {
		log.Printf("s3 get object failed (%v: %d-%d): %v", name, off, sz, err)
		return nil, err
	}
	if content, err := io.ReadAll(r.Body); err != nil {
		log.Printf("s3 get object read response error (%v: %d-%d): %v", name, off, sz, err)
		return nil, err
	} else {
		return content, nil
	}
}

func (c *S3C) PutObject(bucket, name string, data []byte) error {
	c.stats.PutCount.Add(1)
	_, err := c.cli.PutObject(context.TODO(), &s3.PutObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(name),
		Body:   bytes.NewReader(data),
	})
	if err != nil {
		log.Printf("s3 put object failed (%v): %v", name, err)
		return err
	}
	return nil
}

func (c *S3C) DeleteObject(bucket, name string) error {
	c.stats.DeleteCount.Add(1)
	_, err := c.cli.DeleteObject(context.TODO(), &s3.DeleteObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(name),
	})
	if err != nil {
		log.Printf("failed to delete s3 object: %v", name)
		return err
	}
	return nil
}

func (c *S3C) HeadObject(bucket, name string) (S3ObjectMeta, error) {
	c.stats.HeadCount.Add(1)
	r, err := c.cli.HeadObject(context.TODO(), &s3.HeadObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(name),
	})
	if err != nil {
		log.Println("s3 size request failed: 0 is returned")
		return S3ObjectMeta{}, err
	}
	return S3ObjectMeta{len: r.ContentLength, lastModified: *r.LastModified}, nil
}

func (c *S3C) ListObject(bucket string, consume func(types.Object)) error {

	var token *string
	for {
		c.stats.ListCount.Add(1)
		output, err := c.cli.ListObjectsV2(context.TODO(), &s3.ListObjectsV2Input{
			Bucket:            aws.String(bucket),
			ContinuationToken: token,
		})
		if err != nil {
			log.Println("s3 list bucket to estimate size failed")
			return err
		}
		for _, o := range output.Contents {
			consume(o)
		}

		if !output.IsTruncated {
			// all objects have been scanned
			break
		}
		token = output.NextContinuationToken
	}
	return nil
}

type S3F struct {
	name     string
	bucket   string
	cli      s3svc
	data     []byte
	writeOff int
	size     int64
	modTime  time.Time
}

// return the id
func (f *S3F) Name() string {
	return f.name
}

func genS3Range(off, sz int) string {
	return fmt.Sprintf("bytes=%d-%d", off, off+sz-1)
}

func (f *S3F) Bytes(off, sz int) ([]byte, error) {
	if sz <= 0 {
		log.Println("requires bytes of length <= 0")
		return nil, nil
	}
	// fetch from s3
	return f.cli.GetObjectRange(f.bucket, f.name, off, sz)
}

func (f *S3F) Sync() error {
	return f.cli.PutObject(f.bucket, f.name, f.data)
}

func (f *S3F) Delete() error {
	return f.cli.DeleteObject(f.bucket, f.name)
}

// input -1 if truncation undesirable
func (f *S3F) Close(maxSz int64) error {
	return nil
}

// stats used to manage sst
// file size is used to read from the back
// length in bytes for regular files; system-dependent for others
func (f *S3F) Size() int64 {
	if f.size == 0 {
		meta, err := f.cli.HeadObject(f.bucket, f.name)
		if err != nil {
			return 0
		}
		f.size = meta.len
		f.modTime = meta.lastModified
	}
	return f.size
}

// mod time is used for level controller such that newly created tables are not used for flushing/compaction
// modification time
func (f *S3F) ModTime() time.Time {
	if f.modTime.Equal(time.Time{}) {
		meta, err := f.cli.HeadObject(f.bucket, f.name)
		if err != nil {
			return time.Now()
		}
		f.modTime = meta.lastModified
		f.size = meta.len
	}
	return f.modTime
}

func (f *S3F) ReadAll() []byte {
	content, err := f.cli.GetObject(f.bucket, f.name)
	if err != nil {
		return nil
	}
	return content
}

// write only write to a local buffer and need sync to package as an object and store on s3
func (f *S3F) Write(data []byte) int {
	written := copy(f.data[f.writeOff:], data)
	y.AssertTrue(len(data) == written)
	f.writeOff += written
	f.size = int64(f.writeOff)
	return written
}

var _ kvio.CSSCli = (*S3CSS)(nil)

type S3CSS struct {
	connected bool
	bucket    string
	cli       s3svc
	stats     *S3Stats
}

func NewS3CCS() *S3CSS {
	return &S3CSS{}
}

// connect to s3 the bucket
func (c *S3CSS) Connect(bucketName string) error {
	c.bucket = bucketName
	if !c.connected {
		cli, err := ConnectToS3()
		if err != nil {
			return err
		}
		c.cli = cli
		c.stats = &cli.stats
		c.connected = true
	}
	return nil
}

func (c *S3CSS) GetStats() string {
	return c.stats.Encode()
}

// Open an existing file
// no truncation via openning size
func (c *S3CSS) OpenFile(name string, _ int) (kvio.CSSF, error) {
	return &S3F{
		name:   name,
		bucket: c.bucket,
		cli:    c.cli,
	}, nil
}

// Open a new file for persistence
func (c *S3CSS) OpenNewFile(name string, maxSz int) (kvio.CSSF, error) {
	return &S3F{
		name:   name,
		bucket: c.bucket,
		cli:    c.cli,
		data:   make([]byte, maxSz),
	}, nil
}

// sync as objects are written
func (c *S3CSS) Sync() error {
	return nil
}

// disconnect from css
func (c *S3CSS) Disconnect() error {
	return nil
}

// estimate the storage size // only for user api to get db size.
func (c *S3CSS) EstimateSize() (int64, int64, error) {
	var lsmSize, vlogSize int64
	collectSize := func(o types.Object) {
		ext := filepath.Ext(*o.Key)
		switch ext {
		case ".sst":
			lsmSize += o.Size
		case ".vlog":
			vlogSize += o.Size
		}
	}
	if err := c.cli.ListObject(c.bucket, collectSize); err != nil {
		return 0, 0, err
	}
	return lsmSize, vlogSize, nil
}

// list objects
func (c *S3CSS) List() ([]string, error) {
	var ret []string
	collectNames := func(o types.Object) {
		ret = append(ret, *o.Key)
	}
	if err := c.cli.ListObject(c.bucket, collectNames); err != nil {
		return nil, err
	}
	return ret, nil
}

// delete the object
func (c *S3CSS) Delete(name string) error {
	return c.cli.DeleteObject(c.bucket, name)
}

// return the bucket name
func (c *S3CSS) Config() string {
	return c.bucket
}
