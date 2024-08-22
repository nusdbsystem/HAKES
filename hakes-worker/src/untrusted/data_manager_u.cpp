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

#include "hakes-worker/untrusted/data_manager_u.h"

#include <sgx_urts.h>

#include "Enclave_u.h"
#include "message/searchservice.h"
#include "utils/hexutil.h"

namespace hakes_worker {

bool DataManagerU::Initialize() {
  sgx_launch_token_t t;
  int updated = 0;
  memset(t, 0, sizeof(sgx_launch_token_t));
  auto sgxStatus = sgx_create_enclave(
      enclave_file_name_.c_str(), SGX_DEBUG_FLAG, &t, &updated, &eid_, NULL);
  if (sgxStatus != SGX_SUCCESS) {
    printf("Failed to create Enclave : error %d - %#x.\n", sgxStatus,
           sgxStatus);
    return false;
  }
  printf("Enclave launched.\n");
  initialized_ = true;
  return true;
}

hakes::SearchWorkerRerankResponse DataManagerU::MergeSearchResults(
    const std::vector<hakes::SearchWorkerRerankResponse> resps, int k,
    const std::string user_id, const std::string ks_addr,
    const uint16_t ks_port) {
  std::string encoded_ids;
  std::string encoded_scores;
  encoded_ids.reserve(resps[0].ids.size() * resps.size());
  encoded_scores.reserve(resps[0].scores.size() * resps.size());
  uint32_t id_offsets[resps.size() + 1];
  uint32_t score_offsets[resps.size() + 1];
  for (int i = 0; i < resps.size(); i++) {
    id_offsets[i] = encoded_ids.size();
    score_offsets[i] = encoded_scores.size();
    encoded_ids.append(resps[i].ids);
    encoded_scores.append(resps[i].scores);
  }
  id_offsets[resps.size()] = encoded_ids.size();
  score_offsets[resps.size()] = encoded_scores.size();
  // encode the offsets
  std::string encoded_id_offsets(reinterpret_cast<char*>(id_offsets),
                                 sizeof(id_offsets));
  std::string encoded_score_offsets(reinterpret_cast<char*>(score_offsets),
                                    sizeof(score_offsets));
  encoded_ids.append(encoded_id_offsets);
  encoded_scores.append(encoded_score_offsets);

  hakes::SearchWorkerRerankResponse merge_resp;

  size_t ids_buf_len = k * sizeof(int64_t) + 4096;
  char merged_ids[ids_buf_len];
  memset(merged_ids, 0, sizeof(merged_ids));
  size_t scores_buf_len = k * sizeof(float) + 4096;
  char merged_scores[scores_buf_len];
  memset(merged_scores, 0, sizeof(merged_scores));

  // the ecall logic

  sgx_status_t sgx_status;
  size_t output_ids_len, output_scores_len;
  auto status = ecall_merge(
      eid_, &sgx_status,
      user_id.c_str(), user_id.size(), ks_addr.c_str(), ks_addr.size(), ks_port, encoded_ids.c_str(),
      encoded_ids.size(), encoded_scores.c_str(), encoded_scores.size(), k,
      merged_ids, ids_buf_len, merged_scores, scores_buf_len, &output_ids_len,
      &output_scores_len);
  if ((status != SGX_SUCCESS) || (sgx_status != SGX_SUCCESS)) {
    printf("failed to merge");
    merge_resp.status = false;
    merge_resp.msg = "merge error";
    merge_resp.ids = "";
    merge_resp.scores = "";
    merge_resp.aux = "merge error";
    return merge_resp;
  } else {
    merge_resp.status = true;
    merge_resp.msg = "merge success";
    merge_resp.ids = hakes::hex_encode(merged_ids, output_ids_len);
    merge_resp.scores = hakes::hex_encode(merged_scores, output_scores_len);
    merge_resp.aux = "merge success";
    return merge_resp;
  }
}

}  // namespace hakes_worker