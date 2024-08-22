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

#ifndef HAKES_UTILS_CRYPTO_EXT_H_
#define HAKES_UTILS_CRYPTO_EXT_H_

#include <string>

#define AES_GCM_IV_SIZE 12
#define AES_GCM_TAG_SIZE 16
#define AES_GCM_AAD_SIZE 4

namespace hakes {

inline size_t get_aes_decrypted_size(size_t size) {
  return size - AES_GCM_IV_SIZE - AES_GCM_TAG_SIZE - AES_GCM_AAD_SIZE;
}

inline size_t get_aes_encrypted_size(size_t size) {
  return size + AES_GCM_IV_SIZE + AES_GCM_TAG_SIZE + AES_GCM_AAD_SIZE;
}

std::string encrypt_content_with_key_aes(const uint8_t* src, size_t size,
                                         const std::string& encryption_key);

std::string decrypt_content_with_key_aes(const uint8_t* src, size_t size,
                                         const std::string& decryption_key);

}  // namespace hakes

#endif  // HAKES_UTILS_CRYPTO_EXT_H_
