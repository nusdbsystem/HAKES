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

package hakesstore

import (
	"fmt"
	"log"
	"path/filepath"
	"time"

	util "hakes-store/hakes-store-util"

	"github.com/go-zookeeper/zk"
)

func INFOLOG(msg string) {
	log.Printf("%v [replica]: %v\n", time.Now().UnixMicro(), msg)
}

// ft manager impl
type FTReplica struct {
	// replica server to manage
	region  string        // root/region
	ownAddr string        // the address of this replica
	rs      ReplicaServer // replica server

	// zookeeper control plane address
	zkAddrSet  []string
	zkconn     *zk.Conn // connection to zookeeper
	zNodeAlloc string   // path to the allocated ephemeral sequence znode

	// state obtained from control plane
	isleader bool

	// ft replica manager output channel
	msgChn chan<- error
}

func NewFTReplica(region, ownAddr string, zkAddrSet []string, s ReplicaServer) (*FTReplica, <-chan error) {
	msgChn := make(chan error)
	return &FTReplica{
		region:    region,
		ownAddr:   ownAddr,
		zkAddrSet: zkAddrSet,

		rs:     s,
		msgChn: msgChn,
	}, msgChn
}

func (m *FTReplica) watchOnRegion() (bool, util.ReplicaGroupView, <-chan zk.Event, error) {
	for {
		liveReplicas, _, watch, err := m.zkconn.ChildrenW(m.region)
		if err != nil {
			return false, nil, nil, fmt.Errorf("zookeeper error duirng get live replicas: %v", err)
		}
		INFOLOG(fmt.Sprintf("replica available: %v", liveReplicas))

		// prepare the view info
		contForNewEvent := false
		view := util.ReplicaGroupView{}
		for _, znode := range liveReplicas {
			if addrByte, _, err := m.zkconn.Get(filepath.Join(m.region, znode)); err != nil {
				if err == zk.ErrNoNode {
					// the node to watch no longer exists -> live replica set changed, repeat the process to find who to watch now
					INFOLOG(fmt.Sprintf("node in view %v deleted, continue to get the next region event", string(znode)))
					// likely we are going to have a new watch event
					contForNewEvent = true
					break
				} else {
					// error with zookeeper
					err = fmt.Errorf("zookeeper error during get view: %v", err)
					// exit the FTReplica
					return false, nil, nil, err
				}
			} else {
				view[string(addrByte)] = struct{}{}
			}
		}

		if contForNewEvent {
			continue
		}

		_, toWatchNode, err := util.FindPrevSeqNode(liveReplicas, m.zNodeAlloc)
		if err != nil {
			if err != util.ErrPrevNodeNotFound {
				return false, nil, nil, err
			}
			INFOLOG("No prevNode, mode will switch to leader")
			// no previous znode means this replica is leader
			// create an empty channel as return
			return true, view, watch, nil
		}
		INFOLOG(fmt.Sprintf("try to watch on prevNode: %v", toWatchNode))

		// watch on the previous node
		prevZnodeData, _, err := m.zkconn.Get(filepath.Join(m.region, toWatchNode))
		if err != nil {
			if err == zk.ErrNoNode {
				// the node to watch no longer exists -> live replica set changed, repeat the process to find who to watch now
				INFOLOG(fmt.Sprintf("prevNode %v deleted, finding another one", string(toWatchNode)))
				continue
			} else {
				// error with zookeeper
				err = fmt.Errorf("zookeeper error during get leader: %v", err)
				// exit the FTReplica
				return false, nil, nil, err
			}
		}
		INFOLOG(fmt.Sprintf("watching on prevNode: %v", string(prevZnodeData)))
		// set this replica as follower
		return false, view, watch, nil
	}
}

func (m *FTReplica) processRegionEvent(event zk.Event) (<-chan zk.Event, error) {
	INFOLOG("pocessing prevNode lost event")
	// change the prev node to watch on and become leader if there is no prevnode
	if event.Type == zk.EventSession || event.Type == zk.EventNotWatching {
		return nil, fmt.Errorf("connection disruption with event: %v", event.Type.String())
	}

	tobeleader, view, regionWatch, err := m.watchOnRegion()
	if err != nil {
		return nil, fmt.Errorf("process prev node event: %v", err)
	}

	m.rs.ViewChange(view)

	// switch leader
	if tobeleader && (!m.isleader) {
		m.rs.PromoteToLeader()
		m.isleader = true
	} else if (!tobeleader) && m.isleader {
		m.rs.DemoteToFollower()
		m.isleader = false
	}

	return regionWatch, err
}

func (m *FTReplica) Start() {
	// connect to zk; we can pass a set of servers.
	c, _, err := zk.Connect(m.zkAddrSet, time.Second)
	if err != nil && err != zk.ErrNodeExists {
		panic(fmt.Errorf("cannot connect to zookeeper"))
	}
	m.zkconn = c

	aliveNode := filepath.Join(m.region, util.ReplicaNodeName)
	INFOLOG(fmt.Sprintf("this replica is at: %v", m.ownAddr))
	retPath, err := c.Create(aliveNode, []byte(m.ownAddr), zk.FlagEphemeral|zk.FlagSequence, zk.WorldACL(zk.PermAll))
	if err != nil && err != zk.ErrNodeExists {
		m.msgChn <- err
	}

	// keep the znode name indicating the liveness of this replica.
	m.zNodeAlloc = filepath.Base(retPath)
	INFOLOG(fmt.Sprintf("obtained position: %v", m.zNodeAlloc))

	// watch on the leader

	tobeleader, view, regionWatch, err := m.watchOnRegion()
	if err != nil {
		m.msgChn <- err
		// exit the ft replica manager
		return
	}

	m.rs.ViewChange(view)

	// switch the service mode // potential this view is passed as the mode change is done.
	if tobeleader {
		m.rs.PromoteToLeader()
		m.isleader = true
	} else {
		m.rs.DemoteToFollower()
		m.isleader = false
	}

	// monitor and act upon contorl events one by one
	for {
		event := <-regionWatch
		w, err := m.processRegionEvent(event)
		if err != nil {
			m.msgChn <- err
			// exit the ft replica manager
			return
		}
		regionWatch = w
	}
}
