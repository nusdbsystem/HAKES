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

package hakesstorecli

import (
	"context"
	"fmt"

	pb "hakes-store/grpc-api"
	util "hakes-store/hakes-store-util"
	kv "hakes-store/hakes-store/kv"

	"google.golang.org/grpc"
)

type HakesStoreClient struct {
	*FTClient
}

func NewHakesStoreClient(region string, zkAddrs []string, rp util.RetryPolicy) *HakesStoreClient {
	ftc := NewFTClient(region, zkAddrs, rp)
	return &HakesStoreClient{
		FTClient: ftc,
	}
}

func (c *HakesStoreClient) Put(ctx context.Context, key, val []byte) error {
	task := func(conn *grpc.ClientConn) error {
		cli := pb.NewHakesStoreKVRClient(conn)
		reply, err := cli.Put(ctx, &pb.HakesStorePutRequest{Key: key, Val: val})
		if err != nil {
			return err
		}
		if !reply.GetSuccess() {
			return fmt.Errorf(reply.GetErrMsg())
		}
		return nil
	}
	if err := c.Invoke(task); err != nil {
		INFOLOG(fmt.Sprintf("failed in Put: %v", err))
		return err
	}
	return nil
}

func (c *HakesStoreClient) Get(ctx context.Context, key []byte) ([]byte, error) {
	var reply *pb.HakesStoreGetReply
	task := func(conn *grpc.ClientConn) error {
		cli := pb.NewHakesStoreKVRClient(conn)
		r, err := cli.Get(ctx, &pb.HakesStoreGetRequest{Key: key})
		if err != nil {
			INFOLOG(fmt.Sprintf("failed in Get request error %v", err))
			return err
		}
		if !r.GetSuccess() {
			INFOLOG(fmt.Sprintf("failed in Get not success %v", err))
			return fmt.Errorf(r.GetErrMsg())
		}
		if !r.GetFound() {
			INFOLOG(fmt.Sprintf("failed in Get not found %v", err))
			return kv.ErrKeyNotFound
		}
		reply = r
		return nil
	}
	if err := c.Invoke(task); err != nil {
		INFOLOG(fmt.Sprintf("failed in Get %v", err))
		return nil, err
	}
	return reply.GetVal(), nil
}

func (c *HakesStoreClient) Delete(ctx context.Context, key []byte) error {
	task := func(conn *grpc.ClientConn) error {
		cli := pb.NewHakesStoreKVRClient(conn)
		reply, err := cli.Del(ctx, &pb.HakesStoreDelRequest{Key: key})
		if err != nil {
			return err
		}
		if !reply.GetSuccess() {
			return fmt.Errorf(reply.GetErrMsg())
		}
		return nil
	}
	if err := c.Invoke(task); err != nil {
		INFOLOG(fmt.Sprintf("failed in Delete: %v", err))
		return err
	}
	return nil
}

func (c *HakesStoreClient) Scan(ctx context.Context, startKey, endKey []byte, count uint32, full bool) (uint32, [][]byte, [][]byte, error) {
	var reply *pb.HakesStoreScanReply
	task := func(conn *grpc.ClientConn) error {

		cli := pb.NewHakesStoreKVRClient(conn)
		var t pb.HakesStoreScanType
		if full {
			t = pb.HakesStoreScanType_FULLSCAN
		}
		if len(endKey) > 0 {
			t = pb.HakesStoreScanType_ENDKEYSCAN
		}
		if count > 0 {
			t = pb.HakesStoreScanType_FIXEDCOUNTSCAN
		}

		r, err := cli.Scan(ctx, &pb.HakesStoreScanRequest{Type: t, Count: count, StartKey: startKey, EndKey: endKey})
		if err != nil {
			return err
		}
		if !r.GetSuccess() {
			return fmt.Errorf(r.GetErrMsg())
		}
		reply = r
		return nil
	}
	if err := c.Invoke(task); err != nil {
		INFOLOG(fmt.Sprintf("failed in Scan: %v", err))
		return 0, nil, nil, err
	}
	return reply.GetCount(), reply.GetKeySet(), reply.GetValSet(), nil
}
