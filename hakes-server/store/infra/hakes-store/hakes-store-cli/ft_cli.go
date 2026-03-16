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
	"log"
	"path/filepath"
	"time"

	util "hakes-store/hakes-store-util"

	"github.com/go-zookeeper/zk"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

func INFOLOG(msg string) {
	log.Printf("%v [client]: %v\n", time.Now().UnixNano(), msg)
}

// the ft service client is expected to be embeded in the specific client implementation

type FTClient struct {
	// per connection
	cm *CliConnHandler

	// per region, we can use a map to store for many shards
	region string // the replica group

	// one per client, for now, we let the client maintain connection for its lifetime
	zkAddr           []string // zookeeper address
	cfgCon           *zk.Conn // connection to get configuration
	retryPolicy      util.RetryPolicy
	getConnWithRetry func() (*grpc.ClientConn, error) // get connection with the retry policy applied
}

func NewFTClient(region string, zkAddr []string, rp util.RetryPolicy) *FTClient {
	if rp == nil {
		// default to retry every 5 second for 3 times.
		rp = util.NewFixedIntervalRetry(5000, 3)
	}
	return &FTClient{
		region:      region,
		zkAddr:      zkAddr,
		retryPolicy: rp,
	}
}

func getCurrentViewAndLeader(conn *zk.Conn, region string) (util.ReplicaGroupView, string, error) {
	// repeat until we get a valid view and leader combination
	for {
		// get view
		liveReplicas, _, err := conn.Children(region)
		if err != nil {
			return nil, "", fmt.Errorf("zookeeper error duirng get live replicas: %v", err)
		}
		INFOLOG(fmt.Sprintf("replica available: %v", liveReplicas))
		// prepare the view info
		contForNewView := false
		view := util.ReplicaGroupView{}
		for _, znode := range liveReplicas {
			if addrByte, _, err := conn.Get(filepath.Join(region, znode)); err != nil {
				if err == zk.ErrNoNode {
					// the node to watch no longer exists -> live replica set changed, repeat the process to find who to watch now
					INFOLOG(fmt.Sprintf("node in view %v deleted, retry to get a new region view", string(znode)))
					// likely we are going to have a new watch event
					contForNewView = true
					break
				} else {
					// error with zookeeper
					err = fmt.Errorf("zookeeper error during get view: %v", err)
					// exit the FTReplica
					return nil, "", err
				}
			} else {
				view[string(addrByte)] = struct{}{}
			}
		}

		if contForNewView {
			continue
		}

		// get leader
		_, leaderNodePath, err := util.FindSmallestSeqNode(liveReplicas, util.ReplicaNodeName)
		if err != nil {
			return nil, "", fmt.Errorf("failed to get the leader replica: %v", err)
		}
		leaderNodeData, _, err := conn.Get(filepath.Join(region, leaderNodePath))
		if err != nil {
			if err == zk.ErrNoNode {
				// the node to watch no longer exists -> live replica set changed, repeat the process to find who to watch now
				INFOLOG(fmt.Sprintf("leader %v deleted, finding another one", string(leaderNodePath)))
				continue
			} else {
				// error with zookeeper
				err = fmt.Errorf("zookeeper error during get leader %s: %v", leaderNodePath, err)
				return nil, "", err
			}
		}
		curLeader := string(leaderNodeData)
		INFOLOG(fmt.Sprintf("found leader: %v", curLeader))
		if view.Contain(curLeader) {
			// a valid combo of leader and view is found, return
			return view, curLeader, nil
		}
		INFOLOG(fmt.Sprintf("leader %v is not in view %v", curLeader, view))
	}
}

func connectToReplica(curLeaderAddr string) (*grpc.ClientConn, error) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*3)
	defer cancel()
	conn, err := grpc.DialContext(ctx, curLeaderAddr, grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock())
	return conn, err
}

// redirection when connection is lost
func redirectConnection(cfgCon *zk.Conn, region string) (*grpc.ClientConn, error) {
	curView, curLeader, err := getCurrentViewAndLeader(cfgCon, region)
	if err != nil {
		INFOLOG(fmt.Sprintf("failed to obtain a valid view: %v", err))
		return nil, err
	}

	INFOLOG(fmt.Sprintf("retrieved new view with content: %v, with leader: %v", curView, curLeader))

	if conn, err := connectToReplica(curLeader); err != nil {
		INFOLOG(fmt.Sprintf("failed to setup connection with replica leader: %v", err))
		return nil, err
	} else {
		INFOLOG(fmt.Sprintf("connected to leader at %v", curLeader))
		return conn, nil
	}
}

// during initial boostrap and zk connection failure
func (c *FTClient) Setup() error {
	// connect to control plane (zookeeper service)
	conn, _, err := zk.Connect(c.zkAddr, time.Second)
	if err != nil {
		return err
	}

	// keep connection
	c.cfgCon = conn
	INFOLOG("connected to configuration service")

	c.getConnWithRetry = func() (*grpc.ClientConn, error) {
		pauser := c.retryPolicy.NewPauser()
		for {
			if svrCon, err := redirectConnection(conn, c.region); err == nil {
				return svrCon, nil
			} else {
				INFOLOG(fmt.Sprintf("failed to redirect connection: %v", err))
			}
			if !pauser.Pause() {
				INFOLOG("exhausted retries")
				return nil, util.ErrExhaustedRetry
			}
			INFOLOG("retrying")
		}
	}

	c.cm = NewCliConnHandler(c.getConnWithRetry, INFOLOG)
	INFOLOG("created new server client handler")

	// initialize connection
	// use default redirection the getConnWithRetry defined above
	if _, err = c.cm.GetConn(nil); err != nil {
		c.Close()
		INFOLOG("failed to establish server connection")
		return util.ErrNoConn
	}

	INFOLOG("bootstrap done")
	return nil
}

func (c *FTClient) Close() {
	if c.cm != nil {
		c.cm.Close()
	}
	INFOLOG("closed connection to replica server")
	if c.cfgCon != nil {
		c.cfgCon.Close()
	}
	INFOLOG("closed connection to configuration service")
}

// the invoke task should return the error of grpc call directly for retry
// only connection unavailability will be retried, other errors will return
// if no error, the task should return nil
func (c *FTClient) Invoke(task func(*grpc.ClientConn) error) error {
	pauser := c.retryPolicy.NewPauser()
	for {
		// obtain a ready connection with default redirection policy
		conn, err := c.cm.GetConn(nil)
		if err != nil {
			return util.ErrConnNotReady
		}

		// use the connection to execute task
		err = task(conn)

		// task successfule finished
		if err == nil {
			return nil
		}

		// check if task failure is due to connection unavailablity during task execution
		if code := status.Convert(err).Code(); code != codes.Unavailable {
			return err
		}
		// retry
		if !pauser.Pause() {
			INFOLOG("exhausted retries to invoke the task")
			return util.ErrExhaustedRetry
		}
		INFOLOG("retrying")
	}
}
