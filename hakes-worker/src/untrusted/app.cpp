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
#include "hakes-worker/untrusted/data_manager_u.h"
#include "utils/crypto_ext.h"
#include "utils/hexutil.h"

#define ENCLAVE_FILENAME "Worker_Enclave.signed.so"

uint8_t secret_key_buf[] = {0x1f, 0x86, 0x6a, 0x3b, 0x65, 0xb6, 0xae, 0xea,
                            0xad, 0x57, 0x34, 0x53, 0xd1, 0x03, 0x8c, 0x01};

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
    scores[i] = static_cast<float>(dis(gen));
    printf("score: %f\n", scores[i]);
  }
  return scores;
}

// just test the data manager
int main(int /*argc*/, char* /*argv[]*/) {
  hakes_worker::DataManagerU data_manager(ENCLAVE_FILENAME);
  data_manager.Initialize();

  // prepare a set of rerank responses
  std::vector<hakes::SearchWorkerRerankResponse> resps;
  int d = 10;
  for (int i = 0; i < 2; i++) {
    hakes::SearchWorkerRerankResponse resp;
    resp.status = true;
    resp.msg = "success";
    resp.ids = hakes::encode_hex_int64s(gen_ids(d, i).get(), d);
    auto scores = gen_scores(d, i);
    printf("hex unencrypted bytes: %s\n",
           hakes::hex_encode(reinterpret_cast<const uint8_t*>(scores.get()),
                             d * sizeof(float))
               .c_str());
    auto encrypted_scores_len =
        hakes::get_aes_encrypted_size(d * sizeof(float));
    auto encrypted_scores = hakes::encrypt_content_with_key_aes(
        reinterpret_cast<const uint8_t*>(scores.get()), d * sizeof(float),
        std::string((const char*)secret_key_buf, sizeof(secret_key_buf)));
    resp.scores =
        hakes::hex_encode(encrypted_scores.c_str(), encrypted_scores.size());
    printf("hex_encrypted_scores (%d bytes): %s\n", resp.scores.size(),
           resp.scores.c_str());
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
  assert(parsed_count == k);
  auto encrypted_scores =
      hakes::hex_decode(merged_resp.scores.c_str(), merged_resp.scores.size());

  auto scores = hakes::decrypt_content_with_key_aes(
      reinterpret_cast<const uint8_t*>(encrypted_scores.data()),
      encrypted_scores.size(),
      std::string((const char*)secret_key_buf, sizeof(secret_key_buf)));

  parsed_count = scores.size() / sizeof(float);
  auto scores_float = reinterpret_cast<const float*>(scores.data());
  assert(parsed_count == k);
  for (int i = 0; i < k; i++) {
    printf("[rank-%d] id: %ld score: %f\n", i, ids[i], scores_float[i]);
  }

  return 0;
}