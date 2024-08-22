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
	"strconv"
	"sync"
	"testing"
	"time"
)

func convert_ts_2_time(ts string) (time.Time, error) {
	i, err := strconv.ParseInt(ts, 10, 64)
	if err != nil {
		panic(err)
	}
	return time.UnixMicro(i), nil
}
func convert_ts_list(ts_l *[]string) ([]time.Time, error) {
	ret := make([]time.Time, len(*ts_l))
	for i, v := range *ts_l {
		t, err := convert_ts_2_time(v)
		if err != nil {
			panic(err)
		}
		ret[i] = t
	}
	return ret, nil
}

func gen_st_et(exec_tl *[]uint64, comm_lat int64) ([]time.Time, []time.Time) {
	now := time.Now()
	st_ts := make([]time.Time, len(*exec_tl))
	et_ts := make([]time.Time, len(*exec_tl))
	for i, v := range *exec_tl {
		st_ts[i] = now
		et_ts[i] = now.Add(time.Duration(v)*time.Microsecond + time.Duration(comm_lat)*time.Microsecond)
	}
	return st_ts, et_ts
}

func TestUpdateStats(t *testing.T) {

	exec_tl := []uint64{6000, 300, 400, 7000, 500}

	fmt.Println("sequential exec")
	ms := &DefaultModelStats{}
	st_ts, et_ts := gen_st_et(&exec_tl, 1000)
	for i := 0; i < len(exec_tl); i++ {
		time.Sleep(time.Duration(et_ts[i].UnixMicro()-st_ts[i].UnixMicro()) * time.Microsecond)
		ms.UpdateStats(st_ts[i], et_ts[i], exec_tl[i])
		log.Printf("est next: %v, cost: %v, switch cost %v\n", ms.GenEstNext(), ms.GenCost(), ms.GenSwitchCost())
	}

	fmt.Println("concurrent exec")
	ms = &DefaultModelStats{}
	st_ts, et_ts = gen_st_et(&exec_tl, 1000)
	var wg sync.WaitGroup
	for i := 0; i < len(exec_tl); i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			time.Sleep(time.Duration(et_ts[id].UnixMicro()-st_ts[id].UnixMicro()) * time.Microsecond)
			ms.UpdateStats(st_ts[id], et_ts[id], exec_tl[id])
			log.Printf("est next: %v, cost: %v, switch cost %v\n", ms.GenEstNext(), ms.GenCost(), ms.GenSwitchCost())
		}(i)
	}
	wg.Wait()
	log.Printf("est next: %v, cost: %v, switch cost %v\n", ms.GenEstNext(), ms.GenCost(), ms.GenSwitchCost())
}
