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
#include <cstring>
#include <memory>
#include <string>

#include "Enclave_t.h"
#include "hakes-worker/common/search_result_agg.h"
#include "message/keyservice_worker.h"
#include "ratls-channel/common/channel_client.h"
#include "utils/base64.h"
#include "utils/cache.h"
#include "utils/hexutil.h"
#include "utils/tcrypto_ext.h"

namespace {

thread_local char user_id_to_use[100];
thread_local size_t user_id_to_use_len;

thread_local uint8_t input_key_buf[BUFSIZ];
thread_local size_t input_key_size;

hakes::SimpleCache<std::string> key_cache{1};
thread_local bool key_cache_release_needed = false;
ratls::RatlsChannelClient* channel = nullptr;

inline std::string GetKeyCacheKey() {
  if (user_id_to_use_len == 0) {
    return "";
  }
  return std::string(user_id_to_use, user_id_to_use_len);
  ;
}

int decode_decryption_keys(const std::string& ks_response) {
  // decode keys from reply message of key service
  std::string user_id;
  std::string input_key_b64;
  std::string model_id;
  std::string model_key_b64;
  int ret = hakes::DecodeKeyServiceWorkerReply(
      ks_response, &user_id, &input_key_b64, &model_id, &model_key_b64);

  // base64 decode the keys
  if (ret != 0) return ret;
  auto input_key_bytes = hakes::base64_decode(
      (const uint8_t*)input_key_b64.data(), input_key_b64.size());

  if ((user_id.size() != user_id_to_use_len) ||
      (memcmp(user_id.c_str(), user_id_to_use, user_id_to_use_len) != 0)) {
    ocall_debug_print_string("unmatched expected id");
    return -1;
  }

  // cache into local execution context
  assert(input_key_bytes.size() < BUFSIZ);
  memset(input_key_buf, 0, input_key_bytes.size()+1);
  memcpy(input_key_buf, input_key_bytes.c_str(), input_key_bytes.size());
  input_key_size = input_key_bytes.size();

  // store the keys in cache
  key_cache.AddToCache(GetKeyCacheKey(),
                       std::unique_ptr<std::string>(
                           new std::string(std::move(input_key_bytes))));
  return ret;
}

void ClearRequestContext() {
  memset(input_key_buf, 0, BUFSIZ);
  // key checked ensure that id to use are populated
  if (key_cache_release_needed) {
    assert(user_id_to_use_len > 0);
    key_cache.Release(GetKeyCacheKey());
    key_cache_release_needed = false;
  }
  user_id_to_use_len = 0;
}

inline void printf(const char* msg) { ocall_debug_print_string(msg); }

sgx_status_t fetch_from_ks(const std::string& request,
                           const std::string& ks_addr, uint16_t ks_port,
                           std::string* ks_response) {
  assert(ks_response);
  if (!channel) {
    channel = new ratls::RatlsChannelClient{ks_port, std::move(ks_addr),
                                            ratls::CheckContext{}};
    if (channel->Initialize() == -1) {
      printf("failed to setup ratls ctx\n");
      return SGX_ERROR_SERVICE_UNAVAILABLE;
    }
  }
  if (channel->Connect() == -1) {
    printf("failed to setup ratls connection\n");
    return SGX_ERROR_SERVICE_UNAVAILABLE;
  }
  if (channel->Send(std::move(request)) == -1) {
    printf("failed to send request to key service\n");
    channel->Close();
    return SGX_ERROR_UNEXPECTED;
  };

  if (channel->Read(ks_response) == -1) {
    printf("failed to get keys from key service\n");
    ks_response->empty();
    channel->Close();
    return SGX_ERROR_UNEXPECTED;
  }
  // channel.Close();
  channel->CloseConnection();
  return SGX_SUCCESS;
}

std::string decode_ids(const char* ids, size_t ids_len) {
  return hakes::hex_decode(ids, ids_len);
}

sgx_status_t decrypt_content_with_key_aes(const uint8_t* src, size_t size,
                                          const uint8_t* decryption_key,
                                          uint8_t** output) {
  size_t cipher_text_size = hakes::get_aes_decrypted_size(size);

  // temporary small variables
  uint8_t iv_buf[AES_GCM_IV_SIZE] = {0};
  uint8_t tag_buf[AES_GCM_TAG_SIZE] = {0};
  uint8_t aad_buf[AES_GCM_AAD_SIZE] = {0};

  // buffer for decrypted result. ownership transfer at the end to output.
  uint8_t* result = (uint8_t*)malloc(cipher_text_size + 1);
  sgx_status_t ret = (result == NULL) ? SGX_ERROR_OUT_OF_MEMORY : SGX_SUCCESS;

  if (ret == SGX_SUCCESS) {
    // copy contents
    const uint8_t* p = src;
    p += cipher_text_size;
    memcpy(iv_buf, p, AES_GCM_IV_SIZE);
    p += AES_GCM_IV_SIZE;
    memcpy(tag_buf, p, AES_GCM_TAG_SIZE);
    p += AES_GCM_TAG_SIZE;
    memcpy(aad_buf, p, AES_GCM_AAD_SIZE);

    // decrypt
    ret = sgx_rijndael128GCM_decrypt(
        (const sgx_aes_gcm_128bit_key_t*)decryption_key, src,
        (uint32_t)cipher_text_size, result, iv_buf, AES_GCM_IV_SIZE, aad_buf,
        AES_GCM_AAD_SIZE, (const sgx_aes_gcm_128bit_tag_t*)tag_buf);
    if (ret != SGX_SUCCESS) {
      ocall_debug_print_string("Failed to decrypt content");
    }
    result[cipher_text_size] = '\0';
  }

  // assign the result to output if success; free the resource otherwise.
  if (ret != SGX_SUCCESS) {
    free(result);
    return ret;
  }
  *output = result;

  return ret;
}

std::string decode_scores(const char* scores, size_t scores_len) {
  auto scores_bytes = hakes::hex_decode(scores, scores_len);
  uint8_t* decrytped_scores = nullptr;
  auto decrypted_size = hakes::get_aes_decrypted_size(scores_bytes.size());
  sgx_status_t success = decrypt_content_with_key_aes(
      (const uint8_t*)scores_bytes.data(), scores_bytes.size(), input_key_buf,
      &decrytped_scores);
  if (success != SGX_SUCCESS) {
    ocall_debug_print_string(std::to_string(success).c_str());
    return "";
  } else {
    auto ret = std::string((const char*)decrytped_scores, decrypted_size);
    free(decrytped_scores);
    return ret;
  }
}

} // anonymous namespace

sgx_status_t ecall_test(const char* in, size_t in_len, char* out,
                        size_t out_len) {
  memcpy(out, in, in_len);
  memset(out + in_len, 0, out_len - in_len);
  return SGX_SUCCESS;
}

sgx_status_t ecall_merge(const char* user_id, size_t user_id_size,
                         const char* ks_addr, size_t addr_len, uint16_t ks_port,
                         const char* ids, size_t ids_len, const char* scores,
                         size_t scores_len, int k, char* merged_ids,
                         size_t ids_buf_len, char* merged_scores,
                         size_t scores_buf_len, size_t* output_ids_size,
                         size_t* output_scores_size) {
  // ready key
  memcpy(user_id_to_use, user_id, user_id_size);
  user_id_to_use_len = user_id_size;
  int cache_ret = -1;
  do {
    cache_ret = key_cache.CheckAndTakeRef(GetKeyCacheKey());
  } while (cache_ret == -1);
  key_cache_release_needed = true;
  if (cache_ret == 0) {
    // fetch key from ks
    std::string ks_response;
    auto create_key_request_msg = [&]() -> std::string {
      return hakes::GetKeyRequest(std::string(user_id, user_id_size), "")
          .EncodeTo();
    };
    auto status =
        fetch_from_ks(create_key_request_msg(), std::string(ks_addr, addr_len),
                      ks_port, &ks_response);
    if (status != SGX_SUCCESS) return status;
    auto dec_ret = decode_decryption_keys(std::move(ks_response));
    if (dec_ret != 0) {
      ClearRequestContext();
      return SGX_ERROR_ECALL_NOT_ALLOWED;
    }
  } else {
    // get from cache
    std::string* ret = nullptr;
    do {
      ret = key_cache.RetrieveFromCache(GetKeyCacheKey());
    } while (!ret);
    memcpy(input_key_buf, ret->data(), ret->size());
    input_key_size = ret->size();
  }
  key_cache.Release(GetKeyCacheKey());
  key_cache_release_needed = false;

  // decode ids and scores
  const uint32_t* ids_off_start =
      reinterpret_cast<const uint32_t*>(ids + ids_len - sizeof(uint32_t));
  const uint32_t* scores_off_start =
      reinterpret_cast<const uint32_t*>(scores + scores_len - sizeof(uint32_t));
  auto ids_offset = reinterpret_cast<const uint32_t*>(ids + *ids_off_start);
  auto scores_offset =
      reinterpret_cast<const uint32_t*>(scores + *scores_off_start);

  // prepare partial results

  int partial_result_parts = (ids_len - *ids_off_start) / sizeof(uint32_t) - 1;

  std::vector<hakes_worker::result_set> partial_results;
  partial_results.reserve(partial_result_parts);
  for (int i = 0; i < partial_result_parts; i++) {
    hakes_worker::result_set partial_result;
    auto ids_bytes =
        decode_ids(ids + ids_offset[i], ids_offset[i + 1] - ids_offset[i]);
    auto scores_bytes = decode_scores(scores + scores_offset[i],
                                      scores_offset[i + 1] - scores_offset[i]);
    const int64_t* ids = reinterpret_cast<const int64_t*>(ids_bytes.data());
    const float* scores = reinterpret_cast<const float*>(scores_bytes.data());
    auto ids_count = ids_bytes.size() / sizeof(int64_t);
    auto scores_count = scores_bytes.size() / sizeof(float);
    assert(ids_count == scores_count);
    partial_result.ids = std::vector<int64_t>(ids, ids + ids_count);
    partial_result.scores = std::vector<float>(scores, scores + scores_count);
    partial_results.push_back(partial_result);
  }
  auto merged_result = search_result_agg(partial_results, k);

  if (merged_result.ids.size() * sizeof(int64_t) > ids_buf_len ||
      merged_result.scores.size() * sizeof(float) > scores_buf_len) {
    return SGX_ERROR_INVALID_PARAMETER;
  }

  memcpy(merged_ids, merged_result.ids.data(),
         merged_result.ids.size() * sizeof(int64_t));

  // encode the merged scores
  uint8_t* enc_merged_scores = nullptr;
  size_t enc_merged_scores_size = hakes::get_aes_encrypted_size(
      merged_result.scores.size() * sizeof(float));
  hakes::encrypt_content_with_key_aes(
      (const uint8_t*)merged_result.scores.data(),
      merged_result.scores.size() * sizeof(float), input_key_buf,
      &enc_merged_scores);
  memcpy(merged_scores, enc_merged_scores, enc_merged_scores_size);

  free(enc_merged_scores);

  memset(merged_ids + merged_result.ids.size() * sizeof(int64_t), 0,
         ids_buf_len - merged_result.ids.size() * sizeof(int64_t));
  memset(merged_scores + enc_merged_scores_size, 0,
         scores_buf_len - enc_merged_scores_size);
  *output_ids_size = merged_result.ids.size() * sizeof(int64_t);
  *output_scores_size = enc_merged_scores_size;
  return SGX_SUCCESS;
}
