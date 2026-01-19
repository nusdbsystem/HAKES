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
	"log"
	"sync"
	"testing"
	"time"
)

func TestNewFPController(t *testing.T) {
	c := NewFPController()
	if c.fgroups == nil {
		t.Errorf("FPController not properly initialized")
	}
}

func TestFPControllerAddnDelete(t *testing.T) {
	c := NewFPController()
	c.Add("foo1", 2)

	got := len(c.fgroups)
	want := 1
	if got != want {
		t.Errorf("got %q, wanted %q", got, want)
	}

	if c.fgroups["foo1"].mapping == nil || c.fgroups["foo1"].in_use == nil || c.fgroups["foo1"].models == nil {
		t.Errorf("FPEntry not properly initialized")
	}

	got = len(c.fgroups["foo1"].funcs)
	want = 2
	if got != want {
		t.Errorf("got %q, wanted %q", got, want)
	}

	// subsequent add
	c.Add("foo2", 4)
	got = len(c.fgroups)
	want = 2
	if got != want {
		t.Errorf("got %q, wanted %q", got, want)
	}

	// prevent override
	c.Add("foo1", 4)

	got = len(c.fgroups)
	want = 2
	if got != want {
		t.Errorf("got %q, wanted %q", got, want)
	}

	got = len(c.fgroups["foo1"].funcs)
	want = 2
	if got != want {
		t.Errorf("got %q, wanted %q", got, want)
	}

	// skip non-existent
	c.Delete("foo3")
	got = len(c.fgroups)
	want = 2
	if got != want {
		t.Errorf("got %q, wanted %q", got, want)
	}

	// delete one entry
	c.Delete("foo1")
	got = len(c.fgroups)
	want = 1
	if got != want {
		t.Errorf("got %q, wanted %q", got, want)
	}
}

func TestCreateFnName(t *testing.T) {
	got := createFnName("foo", 1)
	want := "FPN-foo-1"
	if got != want {
		t.Errorf("got %q, wanted %q", got, want)
	}
}

func checkFPEntryEqu(e1, e2 *FPEntry, t *testing.T) bool {
	if len(e1.funcs) != len(e2.funcs) {
		t.Errorf("Mismatched FPEntry funcs")
		return false
	}
	for i, v := range e1.funcs {
		if v != e2.funcs[i] {
			t.Errorf(fmt.Sprintf("func: %v found in e1 but not e2", v))
			return false
		}
	}

	if len(e1.mapping) != len(e2.mapping) {
		t.Errorf("Mismatched mapping count")
		return false
	}
	for i, v := range e1.mapping {
		if w, found := e2.mapping[i]; !found || w != v {
			t.Errorf("Mismatched mapping")
			return false
		}
	}

	if len(e1.in_use) != len(e2.in_use) {
		t.Errorf("Mismatched in use count")
		return false
	}
	for i, v := range e1.in_use {
		if w, found := e2.in_use[i]; !found || w.get() != v.get() {
			t.Errorf("Mismatched in use")
			return false
		}
	}

	if len(e1.models) != len(e2.models) {
		t.Errorf("Mismatched model stats count")
		return false
	}
	for i := range e1.models {
		if _, found := e2.models[i]; !found {
			t.Errorf("Mismatched model stats entries")
			return false
		}
	}
	return true
}

func checkControllerEqu(c1, c2 *FPController, t *testing.T) {
	if len(c1.fgroups) != len(c2.fgroups) {
		t.Errorf("Mismatched fgroups count")
	}
	for i, v := range c1.fgroups {
		if w, found := c2.fgroups[i]; !found || !checkFPEntryEqu(v, w, t) {
			t.Errorf("Mismatched FPEntry")
		}
	}
}

func TestFPControllerPickFnInstance(t *testing.T) {
	c := NewFPController()

	// pick non-existent function
	_, _, err := c.PickFnInstance("foo", "m1")

	if err == nil {
		t.Errorf("error not reported for non-existent fgroup")
	}

	c.Add("foo1", 2)
	picked, _, err := c.PickFnInstance("foo1", "m1")
	if err != nil {
		panic(err)
	}

	want := &FPController{
		fgroups: map[string]*FPEntry{
			"foo1": {
				status:  FPEReady,
				name:    "foo1",
				funcs:   []string{"", ""},
				mapping: map[string]int{"m1": 0},
				in_use:  map[string]*FPEInUseCount{"m1": {1}},
				models:  map[string]ModelStats{"m1": &DefaultModelStats{}},
			},
		},
	}
	want_picked := createFnName("foo1", 0)
	if picked != want_picked {
		t.Errorf("got %q, wanted %q", picked, want_picked)
	}
	checkControllerEqu(&c, want, t)

	// repeated pick
	picked, _, err = c.PickFnInstance("foo1", "m1")
	if err != nil {
		panic(err)
	}
	want.fgroups["foo1"].in_use["m1"].incOne()
	want.fgroups["foo1"].funcs[0] = "m1"
	checkControllerEqu(&c, want, t)

	// pick another
	picked, _, err = c.PickFnInstance("foo1", "m2")
	if err != nil {
		panic(err)
	}

	want_picked = createFnName("foo1", 1)
	if picked != want_picked {
		t.Errorf("got %q, wanted %q", picked, want_picked)
	}
	want.fgroups["foo1"].models["m2"] = &DefaultModelStats{}
	want.fgroups["foo1"].in_use["m2"] = &FPEInUseCount{1}
	want.fgroups["foo1"].mapping["m2"] = 1
	checkControllerEqu(&c, want, t)

	// picking exceeds func backend
	picked, _, err = c.PickFnInstance("foo1", "m3")
	if err == nil {
		t.Errorf("should report error on exceeding limits")
	}
	want.fgroups["foo1"].models["m3"] = &DefaultModelStats{}
	want.fgroups["foo1"].in_use["m3"] = &FPEInUseCount{0}
	want.fgroups["foo1"].mapping["m3"] = -1
	checkControllerEqu(&c, want, t)
}

func TestFPControllerReleaseAndUpdateStats(t *testing.T) {
	c := NewFPController()

	// gen test timestamps
	exec_tl := []uint64{6000, 300, 400, 7000, 500}
	st_ts, et_ts := gen_st_et(&exec_tl, 1000)

	// release non-existent function
	err := c.ReleaseAndUpdateStats("foo", "m0", st_ts[0], et_ts[0], exec_tl[0])

	if err == nil || err.Error() != "non-existent fgroup" {
		t.Errorf("error non-existent fgroup expected")
	}

	// release non-existent model
	c.Add("foo1", 2)
	c.PickFnInstance("foo1", "m1")
	err = c.ReleaseAndUpdateStats("foo1", "m0", st_ts[0], et_ts[0], exec_tl[0])
	if err == nil || err.Error() != "model m0 not managed by foo1" {
		t.Errorf("error model not managed expected")
	}

	// invalid ops should not modify the controller state
	want := &FPController{
		fgroups: map[string]*FPEntry{
			"foo1": {
				status:  FPEReady,
				name:    "foo1",
				funcs:   []string{"", ""},
				mapping: map[string]int{"m1": 0},
				in_use:  map[string]*FPEInUseCount{"m1": {1}},
				models:  map[string]ModelStats{"m1": &DefaultModelStats{}},
			},
		},
	}
	checkControllerEqu(&c, want, t)

	// test release not last finisher
	c.PickFnInstance("foo1", "m1")
	want.fgroups["foo1"].funcs[0] = "m1"
	c.ReleaseAndUpdateStats("foo1", "m1", st_ts[0], time.Now(), exec_tl[0])
	checkControllerEqu(&c, want, t)

	c.PickFnInstance("foo1", "m2")
	want.fgroups["foo1"].models["m2"] = &DefaultModelStats{}
	want.fgroups["foo1"].in_use["m2"] = &FPEInUseCount{1}
	want.fgroups["foo1"].mapping["m2"] = 1

	// test release last finisher
	c.ReleaseAndUpdateStats("foo1", "m1", st_ts[0], time.Now(), exec_tl[0])
	want.fgroups["foo1"].in_use["m1"] = &FPEInUseCount{0}
	checkControllerEqu(&c, want, t)

	// picking a released spot
	picked, _, err := c.PickFnInstance("foo1", "m3")
	if err != nil {
		log.Printf("all busy")
		// panic(err)
	}
	// wait for the ownership of m1 to expire
	time.Sleep(2 * 6000 * time.Microsecond)
	picked, _, err = c.PickFnInstance("foo1", "m3")
	if err != nil {
		panic(err)
	}

	want_picked := createFnName("foo1", 0)
	if picked != want_picked {
		t.Errorf("got %q, wanted %q", picked, want_picked)
	}

	want.fgroups["foo1"].funcs[0] = "" // m1 claim revoked by m3
	want.fgroups["foo1"].models["m3"] = &DefaultModelStats{}
	want.fgroups["foo1"].in_use["m3"] = &FPEInUseCount{1}
	want.fgroups["foo1"].mapping["m3"] = 0
	checkControllerEqu(&c, want, t)

	// test failed pick when all busy
	_, _, err = c.PickFnInstance("foo1", "m1")
	if err == nil || err.Error() != "Failed to allcoate fn instance (All Busy)" {
		t.Errorf("expected failure for func alloc")
	}
	want.fgroups["foo1"].mapping["m1"] = -1
	checkControllerEqu(&c, want, t)

	c.ReleaseAndUpdateStats("foo1", "m2", st_ts[0], et_ts[0], exec_tl[0])
	c.PickFnInstance("foo1", "m1")
	want.fgroups["foo1"].in_use["m2"] = &FPEInUseCount{0}
	want.fgroups["foo1"].in_use["m1"] = &FPEInUseCount{1}
	want.fgroups["foo1"].mapping["m1"] = 1
	checkControllerEqu(&c, want, t)
	c.ReleaseAndUpdateStats("foo1", "m3", st_ts[0], et_ts[0], exec_tl[0])
	c.ReleaseAndUpdateStats("foo1", "m1", st_ts[0], et_ts[0], exec_tl[0])
	want.fgroups["foo1"].in_use["m1"] = &FPEInUseCount{0}
	want.fgroups["foo1"].in_use["m3"] = &FPEInUseCount{0}
	checkControllerEqu(&c, want, t)

	c.PrintState()

	// test sequential (all should use func-0)
	log.Println("test sequential")
	for i := 0; i < 5; i++ {
		model_name := "m" + fmt.Sprint(i/2)
		picked, _, err := c.PickFnInstance("foo1", model_name)
		if err != nil {
			panic(err)
		}

		log.Printf("request %d on %v picked %v", i, model_name, picked)
		err = c.ReleaseAndUpdateStats("foo1", model_name, st_ts[i], et_ts[i], exec_tl[i])
		if err != nil {
			panic(err)
		}
	}

	// test concurrent
	log.Print("test concurrent")
	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			model_name := "m" + fmt.Sprint(id/2)
			var wg2 sync.WaitGroup
			for i := range exec_tl {
				wg2.Add(1)
				go func(idx int) {
					defer wg2.Done()
					picked, _, err := c.PickFnInstance("foo1", model_name)
					if err != nil {
						panic(err)
					}

					log.Printf("request %d on %v picked %v", idx, model_name, picked)
					time.Sleep(time.Duration(et_ts[idx].UnixMicro()-st_ts[idx].UnixMicro()) * time.Microsecond)
					err = c.ReleaseAndUpdateStats("foo1", model_name, st_ts[idx], et_ts[idx], exec_tl[idx])
					if err != nil {
						panic(err)
					}
				}(i)
			}
			wg2.Wait()
		}(i)
	}
	wg.Wait()
	c.PrintState()
}
