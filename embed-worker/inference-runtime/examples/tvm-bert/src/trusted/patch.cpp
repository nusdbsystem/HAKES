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

#include <string.h>

#include "Enclave_t.h"
#include "embed-worker/inference-runtime/examples/tvm-default/src/common/abstract.h"
#include "utils/tcrypto_ext.h"

namespace embed_worker {

size_t load_and_decode(const char* id, size_t len, const char* dec_key,
                       void* store, char** out, int* status) {
  char* loaded = nullptr;
  size_t loaded_len = 0;
  *status = ocall_load_content(id, len, &loaded, &loaded_len, store);
  if (*status != SGX_SUCCESS) return 0;
  char* cipher_text = (char*)malloc(loaded_len + 1);
  if (cipher_text == NULL) {
    *status = SGX_ERROR_OUT_OF_MEMORY;
    ocall_free_loaded(id, len, store);
    return 0;
  }
  memcpy(cipher_text, loaded, loaded_len);
  cipher_text[loaded_len] = '\0';
  ocall_free_loaded(id, len, store);

  // decrypt
  uint8_t* content = nullptr;
  *status =
      hakes::decrypt_content_with_key_aes((const uint8_t*)cipher_text, loaded_len,
                                   (const uint8_t*)dec_key, &content);
  free(cipher_text);
  cipher_text = nullptr;
  if (*status != SGX_SUCCESS) return 0;

  // prepare return
  *out = reinterpret_cast<char*>(content);
  // for nenc
  // *out = reinterpret_cast<char*>(cipher_text);
  return loaded_len;
};

}  // namespace embed_worker
