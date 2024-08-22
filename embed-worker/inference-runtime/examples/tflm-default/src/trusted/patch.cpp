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

#include "Enclave_t.h"
#include "embed-worker/inference-runtime/examples/tflm-default/src/common/abstract.h"
#include "utils/tcrypto_ext.h"

namespace embed_worker {

std::unique_ptr<ModelContext> load_model(const char* model_id,
                                         size_t model_id_len,
                                         const char* dec_key, void* store,
                                         int* status) {
  assert(status);
  assert(dec_key);
  assert(model_id);
  assert(model_id_len > 0);

  char* loaded{nullptr};
  size_t load_len{0};

  char fetch_id[model_id_len + sizeof(model_file_suffix)];
  memcpy(fetch_id, model_id, model_id_len);
  memcpy(fetch_id + model_id_len, model_file_suffix, sizeof(model_file_suffix));
  size_t fetch_id_len = sizeof(fetch_id);
  ocall_load_content(fetch_id, fetch_id_len, &loaded, &load_len, store);
  // copy to enclave memory
  char* cipher_text = (char*)malloc(load_len + 1);
  if (cipher_text == NULL) {
    *status = SGX_ERROR_OUT_OF_MEMORY;
    ocall_free_loaded(fetch_id, fetch_id_len, store);
    return NULL;
  }
  memcpy(cipher_text, loaded, load_len);
  cipher_text[load_len] = '\0';
  ocall_free_loaded(fetch_id, fetch_id_len, store);

  // decrypt
  uint8_t* content;
  *status = hakes::decrypt_content_with_key_aes(
      (const uint8_t*)cipher_text, load_len, (const uint8_t*)dec_key, &content);
  free(cipher_text);
  cipher_text = nullptr;
  if (*status != SGX_SUCCESS)
    return NULL;  // note that the caller only return a buffer if success.

  // prepare return
  return std::unique_ptr<ModelContext>(
      new ModelContextImpl((char*)content, load_len));
}

}  // namespace embed_worker