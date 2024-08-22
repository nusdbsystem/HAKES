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

#include <cassert>
#include <memory>
#include <random>

#include "hakes-worker/common/data_manager.h"
#include "hakes-worker/no-sgx/data_manager_impl.h"
#include "utils/hexutil.h"

inline std::unique_ptr<int64_t[]> gen_ids(int n, int seed) {
  std::unique_ptr<int64_t[]> ids{new int64_t[n]};
  std::mt19937 gen(seed);
  std::uniform_int_distribution<> dis(0, 10000);

  for (int i = 0; i < n; i++) {
    ids[i] = dis(gen);
    printf("id: %ld\n", ids[i]);
  }
  return ids;
}

inline std::unique_ptr<float[]> gen_scores(int n, int seed) {
  std::unique_ptr<float[]> scores{new float[n]};
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = 0; i < n; i++) {
    scores[i] = dis(gen);
    printf("score: %f\n", scores[i]);
  }
  return scores;
}

// just test the data manager
int main(int argc, char* argv[]) {
  hakes_worker::DataManagerImpl data_manager;
  data_manager.Initialize();

  // prepare a set of rerank responses
  std::vector<hakes::SearchWorkerRerankResponse> resps;
  for (int i = 0; i < 10; i++) {
    hakes::SearchWorkerRerankResponse resp;
    resp.status = true;
    resp.msg = "success";
    resp.ids = hakes::encode_hex_int64s(gen_ids(10, i).get(), 10);
    resp.scores = hakes::hex_encode(
        reinterpret_cast<const char*>(gen_scores(10, i).get()),
        10 * sizeof(float));
    resp.aux = "aux";
    resps.push_back(resp);
  }

  // merge the responses
  int k = 10;
  hakes::SearchWorkerRerankResponse merged_resp =
      data_manager.MergeSearchResults(resps, k);

  // print the merged response
  printf("aux: %s\n", merged_resp.aux.c_str());
  printf("status: %d\n", merged_resp.status);
  printf("msg: %s\n", merged_resp.msg.c_str());
  size_t parsed_count;
  auto ids = hakes::decode_hex_int64s(merged_resp.ids, &parsed_count);
  printf("ids size: %ld\n", parsed_count);
  assert(parsed_count == k);
  auto scores_byte =
      hakes::hex_decode(merged_resp.scores.c_str(), merged_resp.scores.size());
  parsed_count = scores_byte.size() / sizeof(float);
  auto scores_float = reinterpret_cast<const float*>(scores_byte.data());
  assert(parsed_count == k);
  for (int i = 0; i < k; i++) {
    printf("[rank-%d] id: %ld score: %f\n", i, ids[i], scores_float[i]);
  }

  return 0;
}