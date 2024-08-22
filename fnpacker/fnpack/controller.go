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

package fnpack

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// create fn managed backend function names
func createFnName(name string, id int) string {
	return fmt.Sprintf("FPN-%v-%d", name, id)
}

// Funtion packer controller entry. One entry for each function group managed
const (
	FPENotFound = "NotFound"
	FPEReady    = "Ready"
	FPEDelete   = "Delete"
	FPECreate   = "Create"
)

type FPEInUseCount struct {
	count int32
}

// disregard the atomic first. protect it with WL.

func (c *FPEInUseCount) get() int32 {
	return atomic.LoadInt32(&c.count)
}

func (c *FPEInUseCount) incOne() int32 {
	return atomic.AddInt32(&c.count, 1)
}

func (c *FPEInUseCount) decOne() int32 {
	return atomic.AddInt32(&c.count, -1)
}

type FPEntry struct {
	status  string                    // not used later for synchronization
	name    string                    // function group name
	funcs   []string                  // status of func
	busy    []bool                    // status of func
	mapping map[string]int            // model to the serving instance
	in_use  map[string]*FPEInUseCount // model in-use count
	models  map[string]ModelStats     // model stats
	mu      sync.RWMutex
}

func NewFPEntry(name string, pool_size int) *FPEntry {
	return &FPEntry{
		status:  FPEReady,
		name:    name,
		funcs:   make([]string, pool_size),
		busy:    make([]bool, pool_size),
		mapping: make(map[string]int),
		in_use:  make(map[string]*FPEInUseCount),
		models:  make(map[string]ModelStats),
	}
}

func (e *FPEntry) getPoolSize() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.funcs)
}

func (e *FPEntry) PrintState() {
	e.mu.RLock()
	for i, f := range e.funcs {
		fmt.Printf("func %d used by %v\n", i, f)
	}
	for m, i := range e.mapping {
		ms := e.models[m]
		fmt.Printf("model %v last used func: %d, in use: %d, est next: %d, switch cost: %d, cost: %d\n", m, i, e.in_use[m].get(), ms.GenEstNext(), ms.GenSwitchCost(), ms.GenCost())
	}
	e.mu.RUnlock()
}

// Function packer controller
type FPController struct {
	fgroups map[string]*FPEntry
	mu      sync.RWMutex
}

func (c *FPController) List() string {
	res := "Fn: \n"
	for k := range c.fgroups {
		res += k + "\n"
	}
	return res
}

func (c *FPController) Add(name string, pool_size int) {
	c.mu.Lock()
	if _, found := c.fgroups[name]; !found {
		c.fgroups[name] = NewFPEntry(name, pool_size)
	}
	c.mu.Unlock()
}

func (c *FPController) Delete(name string) {
	c.mu.Lock()
	delete(c.fgroups, name)
	c.mu.Unlock()
}

func (c *FPController) CheckStatus(name string) string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if v, found := c.fgroups[name]; found {
		return v.status
	} else {
		return FPENotFound
	}
}

func (c *FPController) check_busy_or_exclusive(fpe *FPEntry, idx int, candidate string) bool {
	if fpe.busy[idx] {
		return true
	}
	last_claim := fpe.funcs[idx]
	if last_claim == "" || last_claim == candidate {
		return false
	}
	stats := fpe.models[last_claim]
	// for exclusive owner give some additional time before stealing
	time_since_last := stats.GenTimeSinceLastCall()
	if time_since_last < 2*stats.GenCost() {
		fmt.Printf("%v claim on %d not expired %d < 2 * %d\n", last_claim, idx, time_since_last, stats.GenCost())
		return true
	}
	fmt.Printf("%v claim on %d expired %d > 2 * %d\n", last_claim, idx, time_since_last, stats.GenCost())
	return false
}

// requires holding of write lock
func (c *FPController) steal_func(fpe *FPEntry, model_name string) int {
	target_id := -1
	for i := range fpe.funcs {
		if !c.check_busy_or_exclusive(fpe, i, model_name) {
			// reset the claim. the subtlety is that if that model come in later as low frequency request, it would take ownership even if non-concurrent
			fpe.funcs[i] = ""
			fpe.busy[i] = true
			if fpe.in_use[model_name].get() != 1 {
				panic(fmt.Errorf("steal even already in use"))
			}
			target_id = i
			break
		}
	}
	return target_id
}

// return the picked function name, backoff time, error
func (c *FPController) PickFnInstance(name, model_name string) (string, uint64, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	fpe, found := c.fgroups[name]
	if !found {
		return "", 0, fmt.Errorf("non-existent fgroup")
	}
	// try with read lock first
	fpe.mu.RLock()
	_, found = fpe.models[model_name]
	if !found {
		// switch to write lock to ceate stats
		fpe.mu.RUnlock()
		fpe.mu.Lock()
		fpe.models[model_name] = &DefaultModelStats{}
		fpe.in_use[model_name] = &FPEInUseCount{1}
		// first time definitely find from func pool
		target_id := c.steal_func(fpe, model_name)
		fpe.mapping[model_name] = target_id
		fpe.mu.Unlock()
		fpe.mu.RLock()
		defer fpe.mu.RUnlock()
		// downgrade back to RLock
		if target_id == -1 {
			// expected to set the backing instance larger than serving models
			fpe.in_use[model_name].decOne()
			return "", 0, fmt.Errorf("failed to allocate fn instance at create (All Busy!)")
		}
		return createFnName(name, target_id), 0, nil
	}
	// pick function instance
	fpe.mu.RUnlock()
	fpe.mu.Lock()
	defer fpe.mu.Unlock()
	last_in_use := fpe.in_use[model_name].incOne()
	if last_in_use == 2 {
		// claim ownership
		fpe.funcs[fpe.mapping[model_name]] = model_name
		return createFnName(name, fpe.mapping[model_name]), 0, nil
	} else if last_in_use > 2 {
		return createFnName(name, fpe.mapping[model_name]), 0, nil
	} else {
		if last_in_use != 1 {
			panic(fmt.Errorf("in use just incremented should be > 1"))
		}
		target_id := c.steal_func(fpe, model_name)
		fpe.mapping[model_name] = target_id
		if target_id == -1 {
			fpe.in_use[model_name].decOne()
			return "", 0, fmt.Errorf("failed to allcoate fn instance (All Busy)")
		}
		return createFnName(name, target_id), 0, nil
	}
}

// need to release the in use and possibly func
func (c *FPController) ReleaseAndUpdateStats(name, model_name string, start_time, end_time time.Time, exec_time uint64) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// check the target fgroup exists
	fpe, found := c.fgroups[name]
	if !found {
		return fmt.Errorf("non-existent fgroup")
	}

	fpe.mu.RLock()

	// check the model is indeed managed under the target fgroup
	_, found = fpe.models[model_name]
	if !found {
		fpe.mu.RUnlock()
		return fmt.Errorf("model %s not managed by %s", model_name, name)
	}

	// free the hold first
	fpe.in_use[model_name].decOne()

	// update stats
	fpe.models[model_name].UpdateStats(start_time, end_time, exec_time)
	if fpe.in_use[model_name].get() != 0 {
		fpe.mu.RUnlock()
		return nil
	}

	// upgrade to write lock
	fpe.mu.RUnlock()
	fpe.mu.Lock()
	defer fpe.mu.Unlock()

	// check again with exclusive access
	if fpe.in_use[model_name].get() != 0 {
		return nil
	}

	fpe.busy[fpe.mapping[model_name]] = false
	return nil
}

func (c *FPController) PrintState() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for i, v := range c.fgroups {
		fmt.Printf("---fgroup: %v \n", i)
		v.PrintState()
		fmt.Printf("---fgroup: %v \n", i)
	}
}

func NewFPController() FPController {
	return FPController{fgroups: make(map[string]*FPEntry)}
}
