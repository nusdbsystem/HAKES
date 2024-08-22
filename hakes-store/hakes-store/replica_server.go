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
	util "hakes-store/hakes-store-util"
)

type ModeMsgType int
type ModeMsg struct {
	t    ModeMsgType
	info string
}

const (
	DemoteToFollowerMsg ModeMsgType = 0
	PromoteToleaderMsg  ModeMsgType = 1
	ViewChangeMsg       ModeMsgType = 2
)

type ReplicaServer interface {
	Start()                           // start the local server
	GetErrChan() <-chan error         // return the error channel to process error events
	PromoteToLeader()                 // handle leader promotion with current view id and data
	DemoteToFollower()                // handle follower demotion with current view id and data
	ViewChange(util.ReplicaGroupView) // handle view change
	Stop()
}
