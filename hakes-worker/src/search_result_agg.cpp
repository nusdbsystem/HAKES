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

#include "hakes-worker/search_result_agg.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>

namespace hakes_worker {

result_set search_result_agg(const std::vector<result_set> partial_results,
                             int k) {
  // return the top k element with the largest scores
  int candidate_count = 0;
  for (int i = 0; i < partial_results.size(); i++) {
    assert(partial_results[i].ids.size() == partial_results[i].scores.size());
    candidate_count += partial_results[i].ids.size();
  }
  // allocate the space
  std::vector<int64_t> concat_ids(candidate_count);
  std::vector<float> concat_scores(candidate_count);

  size_t start_offset = 0;
  for (int i = 0; i < partial_results.size(); i++) {
    auto& it = partial_results[i];
    memcpy(concat_ids.data() + start_offset, it.ids.data(),
           it.ids.size() * sizeof(int64_t));
    memcpy(concat_scores.data() + start_offset, it.scores.data(),
           it.scores.size() * sizeof(float));
    start_offset += it.ids.size();
  }

  // sort ids based on scores
  std::vector<int> idx(candidate_count);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&concat_scores](int i1, int i2) {
    return concat_scores[i1] > concat_scores[i2];
  });

  // get the top k
  result_set output;
  output.ids.resize(k);
  output.scores.resize(k);
  for (int i = 0; i < k; i++) {
    output.ids[i] = concat_ids[idx[i]];
    output.scores[i] = concat_scores[idx[i]];
  }

  return output;
}

}  // namespace hakes_worker
