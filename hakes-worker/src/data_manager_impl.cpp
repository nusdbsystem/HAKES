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

#include "hakes-worker/data_manager_impl.h"

#include <cassert>
#include <cstring>

#include "hakes-worker/search_result_agg.h"
#include "utils/hexutil.h"

namespace hakes_worker {

bool DataManagerImpl::Initialize() { return true; }

hakes::SearchWorkerRerankResponse DataManagerImpl::MergeSearchResults(
    const std::vector<hakes::SearchWorkerRerankResponse> resps, int k) {
  hakes::SearchWorkerRerankResponse merged_resp;
  std::vector<result_set> partial_results;
  for (int i = 0; i < resps.size(); i++) {
    auto ids_bytes =
        hakes::hex_decode(resps[i].ids.c_str(), resps[i].ids.size());
    auto scores_bytes =
        hakes::hex_decode(resps[i].scores.c_str(), resps[i].scores.size());
    auto ids_count = ids_bytes.size() / sizeof(int64_t);
    const int64_t* ids = reinterpret_cast<const int64_t*>(ids_bytes.data());
    const float* scores = reinterpret_cast<const float*>(scores_bytes.data());
    auto scores_count = scores_bytes.size() / sizeof(float);
    assert(ids_count == scores_count);
    result_set partial_result;
    partial_result.ids = std::vector<int64_t>(ids, ids + ids_count);
    partial_result.scores = std::vector<float>(scores, scores + scores_count);
    partial_results.push_back(partial_result);
  }
  auto merged_result = search_result_agg(partial_results, k);

  merged_resp.ids =
      hakes::hex_encode(reinterpret_cast<const char*>(merged_result.ids.data()),
                        merged_result.ids.size() * sizeof(int64_t));
  merged_resp.scores = hakes::hex_encode(
      reinterpret_cast<const char*>(merged_result.scores.data()),
      merged_result.scores.size() * sizeof(float));
  merged_resp.status = true;
  merged_resp.msg = "merge success";
  return merged_resp;
}

}  // namespace hakes_worker