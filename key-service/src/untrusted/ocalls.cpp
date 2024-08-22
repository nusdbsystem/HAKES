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

/**
 * implement the ocalls defined in worker edl files.
 * packaged into libworker.a
 */

#include <cassert>
#include <climits>

#include "Enclave_u.h"  // include this helps to solve the undefined reference error for ocalls. Enclave_u.c is compiled as c.
#include "store-client/seal_format.h"
#include "store-client/store.h"
#include "utils/base64.h"

// implementation of ocalls
void ocall_save_kv(const void* key_buf, size_t key_len, const void* value_buf,
                   size_t value_len, void* store_ptr) {
  auto key_b64 = hakes::base64_encode((const uint8_t*)key_buf, key_len);
  auto value_b64 = hakes::base64_encode((const uint8_t*)value_buf, value_len);
  printf("Save key value: key: %s, value: %s\n", key_b64.c_str(),
         value_b64.c_str());
  auto store = static_cast<hakes::Store*>(store_ptr);
  store->Put(
      hakes::CreateSealedFileName(std::string((const char*)key_buf, key_len)),
      std::string((const char*)value_buf, value_len));
}

void ocall_get_kv(const void* key_buf, size_t key_len, void* value_buf,
                  size_t value_buf_len, size_t* value_len, void* store_ptr) {
  printf("store get: key of size : %s\n",
         std::string((const char*)key_buf, key_len).c_str());

  std::string value;
  auto store = static_cast<hakes::Store*>(store_ptr);
  if (store->Get(hakes::CreateSealedFileName(
                     std::string((const char*)key_buf, key_len)),
                 &value) == -1) {
    *value_len = 0;  // store error
    return;
  }
  if (value_buf_len > value.size()) {
    memcpy(value_buf, value.data(), value.size());
  }
  // Note that if insufficient trusted buffer
  // indicate the desired buffer size for enclave
  *value_len = value.size();
}

void ocall_debug_print(const void* s, size_t len) {
  assert(len < INT_MAX);
  printf("DEBUG PRINT: %.*s\n", (int)len, (const char*)s);
}

void ocall_debug_print_string(const char* s) { printf("DEBUG PRINT: %s\n", s); }
// implementation of ocalls
