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

#ifndef HAKES_UTILS_SEAL_T_H_
#define HAKES_UTILS_SEAL_T_H_

#include <sgx_trts.h>
#include <sgx_tseal.h>
#include <stdio.h>
#include <string.h>

namespace hakes {

const char aad_mac_dummy[BUFSIZ] = "aad mac text";  // gy0106 to be replaced.

inline uint32_t get_sealed_data_size(uint32_t aad_mac_text_size,
                                     uint32_t data_to_seal_size) {
  return sgx_calc_sealed_data_size(aad_mac_text_size, data_to_seal_size);
}

inline uint32_t get_unsealed_data_size(const uint8_t* sealed_data) {
  return sgx_get_encrypt_txt_len((const sgx_sealed_data_t*)sealed_data);
}

sgx_status_t seal_data(const char* aad_mac_text, uint32_t aad_mac_text_size,
                       const char* data_to_seal, uint32_t data_to_seal_size,
                       uint8_t* output, uint32_t output_buf_size);

sgx_status_t unseal_data(const uint8_t* sealed_data, uint32_t data_size,
                         const char* aad_mac_text, uint32_t aad_mac_text_size,
                         char* output, uint32_t output_buf_size);

}  // namespace hakes

#endif  // HAKES_UTILS_SEAL_T_H_
