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
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	kvio "hakes-store/hakes-store/io"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/kinesis"
	"github.com/aws/aws-sdk-go-v2/service/kinesis/types"
)

var (
	errKinesisConnection  = errors.New("failed to connect to Kinesis")
	errKinesisLogNotFound = errors.New("failed to open recovered log")
)

// allow us to mock the service and test locally
type kinesisSvc interface {
	CreateStream(topic string) error
	PutRecord(topic string, part string, record []byte) error
	ScanAll(topic string, consume func(string, []byte) error) error
	ListStream(topic string) ([]string, error)
	DeleteStream(topic string)
}

type KinesisStats struct {
	PutSize atomic.Uint64
	GetSize atomic.Uint64
}

func (s *KinesisStats) Encode() string {
	if s == nil {
		return ""
	}
	return fmt.Sprintf("Kinesis client: Put: %d, Get: %d", s.PutSize.Load(), s.GetSize.Load())
}

type KIC struct {
	cli   *kinesis.Client
	stats KinesisStats
}

func (c *KIC) CreateStream(topic string) error {
	_, err := c.cli.CreateStream(context.TODO(), &kinesis.CreateStreamInput{
		StreamName:        aws.String(topic),
		StreamModeDetails: &types.StreamModeDetails{StreamMode: types.StreamModeOnDemand},
	})
	if err != nil {
		log.Printf("failed to create stream %v: %v", topic, err)
		return err
	}
	sew := kinesis.NewStreamExistsWaiter(c.cli)
	waitTime := time.Minute * 10
	_, err = sew.WaitForOutput(context.TODO(), &kinesis.DescribeStreamInput{
		StreamName: aws.String(topic),
	}, waitTime)
	if err != nil {
		log.Printf("failed to wait for stream %v to be ready: %v", topic, err)
		return err
	}
	return nil
}

func (c *KIC) PutRecord(topic string, part string, record []byte) error {
	c.stats.PutSize.Add(uint64(len(record)))
	_, err := c.cli.PutRecord(context.TODO(), &kinesis.PutRecordInput{
		StreamName:   aws.String(topic),
		Data:         record,
		PartitionKey: aws.String(part),
	})
	if err != nil {
		log.Printf("kinesis put record to stream %v (part: %v) failed: %v", topic, part, err)
		return err
	}
	return nil
}

func (c *KIC) ScanAll(topic string, consume func(string, []byte) error) error {
	// default max 1000 shard should finish the stream
	lso, err := c.cli.ListShards(context.TODO(), &kinesis.ListShardsInput{
		StreamName: aws.String(topic),
	})
	if err != nil {
		log.Printf("kinesis list shard in stream %v failed: %v", topic, err)
		return err
	}
	for _, shard := range lso.Shards {
		sio, err := c.cli.GetShardIterator(context.TODO(), &kinesis.GetShardIteratorInput{
			StreamName:        aws.String(topic),
			ShardId:           shard.ShardId,
			ShardIteratorType: types.ShardIteratorTypeTrimHorizon,
		})
		if err != nil {
			log.Fatal(err)
		}
		sharditer := sio.ShardIterator
		if sharditer != nil {
			for {
				gro, err := c.cli.GetRecords(context.TODO(), &kinesis.GetRecordsInput{
					ShardIterator: sharditer,
					Limit:         aws.Int32(2),
				})
				if err != nil {
					log.Fatal(err)
				}
				for _, r := range gro.Records {
					c.stats.GetSize.Add(uint64(len(r.Data)))
					if err := consume(*r.PartitionKey, r.Data); err != nil {
						log.Printf("failed to consume record from stream %v (part %v): %v", topic, r.PartitionKey, err)
						return err
					}
				}
				if gro.NextShardIterator == nil || *gro.MillisBehindLatest == 0 {
					break
				}
				sharditer = gro.NextShardIterator
			}
		}
	}
	log.Printf("finish scanning kinesis stream %v", topic)
	return nil
}

func (c *KIC) ListStream(topic string) ([]string, error) {
	var ret []string
	startStream := topic
	for {
		listStreamOut, err := c.cli.ListStreams(context.TODO(), &kinesis.ListStreamsInput{
			ExclusiveStartStreamName: aws.String(startStream),
		})
		if err != nil {
			log.Printf("failed to list stream in topic %v: %v", topic, err)
		}
		sns := listStreamOut.StreamNames
		for _, sn := range sns {
			if !strings.HasPrefix(sn, topic) {
				// do not list further.
				return ret, nil
			}
			ret = append(ret, sn)
		}
		if !*listStreamOut.HasMoreStreams {
			break
		}
		startStream = sns[len(sns)-1]
	}
	return ret, nil
}

// use with caution. stream can only be deleted when all wal contained data are persisted as SST.
func (c *KIC) DeleteStream(topic string) {
	_, err := c.cli.DeleteStream(context.TODO(), &kinesis.DeleteStreamInput{
		StreamName: aws.String(topic),
	})
	if err != nil {
		log.Printf("kinesis delete stream %v failed: %v", topic, err)
	}
}

func ConnectToKinesis() (*KIC, error) {
	cfg, err := config.LoadDefaultConfig(context.TODO())
	if err != nil {
		log.Println(err)
		return nil, err
	}
	cfg.Region = awsRegion
	return &KIC{cli: kinesis.NewFromConfig(cfg)}, nil
}

type KinesisLog struct {
	topic string
	name  string
	size  uint32
	cli   kinesisSvc // for logging
	data  []byte     // for recovery
}

// return the id
func (l *KinesisLog) Name() string {
	return l.name
}

// append new contents to the log
// write should only buffer the write
func (l *KinesisLog) Append(b []byte) (int, error) {
	requiredSize := len(l.data) + len(b)
	if l.data == nil {
		l.data = make([]byte, len(b), 2*len(b))
		copy(l.data, b)
	} else if cap(l.data) < requiredSize {
		targetCap := 2 * cap(l.data)
		for targetCap < requiredSize {
			targetCap = 2 * targetCap
		}
		newbuf := make([]byte, requiredSize, targetCap)
		copy(newbuf, l.data)
		copy(newbuf[len(l.data):], b)
		l.data = newbuf
	} else {
		l.data = append(l.data, b...)
	}
	return len(b), nil
}

// sync ensures all appends are persisted
func (l *KinesisLog) Sync() error {
	if len(l.data) == 0 {
		return nil // nothing to sync
	}
	err := l.cli.PutRecord(l.topic, l.name, l.data)
	l.data = l.data[:0]
	if err != nil {
		return err
	}
	return nil
}

func (l *KinesisLog) Close() error {
	return nil
}

// Size() (int64, error) // length of the log
func (l *KinesisLog) Size() uint32 {
	return l.size
}

// only used during recovery so we are certain it is cached
// create an independent reader that does not modify the state of DLog
func (l *KinesisLog) NewReader(offset int) io.Reader {
	return bytes.NewReader(l.data[offset:])
}

type stream2LogMeta struct {
	logIds map[string]struct{}
}

var _ kvio.DLogHandler = (*KinesisLogHandler)(nil)

type KinesisLogHandler struct {
	connected      bool
	topic          string
	cli            kinesisSvc
	estLogSize     uint64
	recoverCache   map[string][]byte
	stream2LogList map[string]stream2LogMeta // list in ascending topic versions, used to GC kinesis stream when log are persisted into SST. initialized during tailing.
	stats          *KinesisStats
}

// topic is versioned by handler. During open, the last version id is retrieved
func NewKinesisLogHandler(oldTopic string, estLogSize uint64) *KinesisLogHandler {
	return &KinesisLogHandler{topic: oldTopic, estLogSize: estLogSize}
}

func (h *KinesisLogHandler) GetStats() string {
	return h.stats.Encode()
}

// return the last string name which is topic plus the latest version. it is used to create the new stream name if same topic is to be used.
func (h *KinesisLogHandler) tailing() (string, error) {
	log.Println("start tailing")
	streams, err := h.cli.ListStream(h.topic)
	if err != nil {
		return "", err
	}

	if len(streams) == 0 {
		return "", nil
	}
	log.Printf("streams: %v", streams)

	h.stream2LogList = make(map[string]stream2LogMeta, len(streams))
	h.recoverCache = make(map[string][]byte, len(streams))

	for _, s := range streams {
		h.stream2LogList[s] = stream2LogMeta{
			logIds: make(map[string]struct{}),
		}

		appendLog := func(part string, record []byte) error {
			// add the stream log mapping to list
			if _, ok := h.stream2LogList[s].logIds[part]; !ok {
				h.stream2LogList[s].logIds[part] = struct{}{}
			}

			// prepare the log in mem for recovery
			if _, ok := h.recoverCache[part]; !ok {
				h.recoverCache[part] = make([]byte, 0, h.estLogSize)
			}
			h.recoverCache[part] = append(h.recoverCache[part], record...)
			return nil
		}
		if err := h.cli.ScanAll(s, appendLog); err != nil {
			return "", err
		}

		// delete empty stream
		if len(h.stream2LogList[s].logIds) == 0 {
			h.cli.DeleteStream(s)
			delete(h.stream2LogList, s)
		}
	}
	log.Printf("last stream: %v", streams[len(streams)-1])
	return streams[len(streams)-1], nil
}

func createVersionedTopicName(lastUsed, topic string) string {

	getName := func(version int) string {
		return fmt.Sprintf("%v-%05d", topic, version)
	}

	if len(lastUsed) < 6 || lastUsed[:len(lastUsed)-6] != topic {
		return getName(0)
	}
	if version, err := strconv.ParseInt(lastUsed[len(lastUsed)-5:], 10, 32); err != nil {
		return getName(0)
	} else {
		return getName(int(version) + 1)
	}
}

func (h *KinesisLogHandler) init() error {
	if lastStream, err := h.tailing(); err != nil {
		log.Printf("kinesis log handler failed in tailing: %v", err)
		return err
	} else {
		h.topic = createVersionedTopicName(lastStream, h.topic)
		if err := h.cli.CreateStream(h.topic); err != nil {
			log.Printf("failed to setup new topic during init: %v", err)
			return err
		}
		log.Printf("new versioned topic name: %v", h.topic)
		log.Println("set up kinesis client")
		return nil
	}
}

// setup dls client
// connect should perform tailing in kinesis
func (h *KinesisLogHandler) Connect(newTopic string) error {
	if !h.connected {
		cli, err := ConnectToKinesis()
		if err != nil {
			return err
		}
		if cli == nil {
			return errKinesisConnection
		}
		h.cli = cli
		h.connected = true
		h.stats = &cli.stats
	} else {
		log.Fatal("kinesis log handler forbids reconnection")
	}
	return h.init()
}

// only used during recovery, we are certain it is cached
// open an existing log resource
func (h *KinesisLogHandler) OpenLog(name string, _ int) (kvio.DLog, error) {
	if _, ok := h.recoverCache[name]; !ok {
		log.Printf("trying to open log %v, that is not rebuild from kinesis tailing", name)
		return nil, errKinesisLogNotFound
	}
	return &KinesisLog{topic: h.topic, name: name, size: uint32(len(h.recoverCache[name])), data: h.recoverCache[name]}, nil
}

// open a new log resources potentially avoid fetching that existing CRL needs
func (h *KinesisLogHandler) OpenNewLog(name string, _ int) (kvio.DLog, error) {
	return &KinesisLog{topic: h.topic, name: name, size: 0, cli: h.cli}, nil
}

// Drop the log
func (h *KinesisLogHandler) Drop(name string) error {
	delete(h.recoverCache, name)
	log.Printf("dropping log %v", name)
	streamDeleted := ""
	for streamName, meta := range h.stream2LogList {
		if _, ok := meta.logIds[name]; ok {
			delete(meta.logIds, name)
			if len(meta.logIds) == 0 {
				// drop the stream
				h.cli.DeleteStream(streamName)
				log.Printf("deleting stream %v", streamDeleted)
				streamDeleted = streamName
				log.Println(streamDeleted)
			}
		}
	}
	if len(streamDeleted) > 0 {
		delete(h.stream2LogList, streamDeleted)
	}
	return nil
}

// list objects
// it is only called during reopening
func (h *KinesisLogHandler) List() ([]string, error) {
	// use local info.
	var ret []string
	for wal := range h.recoverCache {
		ret = append(ret, wal)
	}
	return ret, nil
}

// disconnect from dls
func (h *KinesisLogHandler) Disconnect() error {
	return nil
}

// from store server replica trigger
func (h *KinesisLogHandler) IsSwitchLogRequired() bool {
	return false
}
