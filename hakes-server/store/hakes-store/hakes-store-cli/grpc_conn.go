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
	"errors"
	"fmt"
	"sync"

	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
)

// managed individual grpc replica connections, it does synchronization during reconnection.
type CliConnHandler struct {
	INFOLOG          func(string)                     // logging
	defaultRedirect  func() (*grpc.ClientConn, error) // redirection logic during connection loss (this can be overriden by the logic input during GetConn)
	pendingConnReset bool                             // if the connection is being reset
	conn             *grpc.ClientConn                 // connection to ftkv service server
	mu               sync.Mutex                       // goroutine access the client manager state in sequence
	cv               *sync.Cond                       // bind to mu
}

func NewCliConnHandler(defaultRedirect func() (*grpc.ClientConn, error), INFOLOG func(string)) *CliConnHandler {
	h := &CliConnHandler{
		INFOLOG:          INFOLOG,
		defaultRedirect:  defaultRedirect,
		pendingConnReset: false,
		conn:             nil,
	}
	// bind conditional variable to the mutex
	h.cv = sync.NewCond(&h.mu)
	return h
}

func (c *CliConnHandler) checkAlive() bool {
	return c.conn != nil && c.conn.GetState() == connectivity.Ready
}

// return a reference of the current connection if it is ready and calls the optional redirect functor to update the connection if lost.
func (c *CliConnHandler) GetConn(redirect func() (*grpc.ClientConn, error)) (*grpc.ClientConn, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.checkAlive() {
		return c.conn, nil
	}
	for {
		if !c.pendingConnReset {
			c.pendingConnReset = true
			break
		}
		// the connection is down
		c.cv.Wait()
		if c.checkAlive() {
			// woken threads should check once again if the connection is restored
			// if so return the connection
			return c.conn, nil
		}
	}

	c.INFOLOG("reconnect in progress")
	defer func() { c.pendingConnReset = false }()
	defer c.cv.Broadcast()

	if redirect == nil {
		if c.defaultRedirect == nil {
			c.INFOLOG("connection lost and no redirect function provided")
			return nil, errors.New("connection lost and no redirect function provided")
		}
		redirect = c.defaultRedirect
	}

	// the single goroutine that is responsible to restore connection
	conn, err := redirect()
	if err != nil {
		c.INFOLOG("failed to redirect connection")
		c.conn = nil
		return nil, fmt.Errorf("failed reconnection: %v", err)
	}
	c.conn = conn
	return conn, nil
}

func (c *CliConnHandler) Close() {
	if c.conn != nil {
		c.conn.Close()
	}
}
