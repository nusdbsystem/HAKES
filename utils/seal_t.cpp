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

#include "seal_t.h"

#include <stdlib.h>

namespace hakes {

sgx_status_t seal_data(const char* aad_mac_text, uint32_t aad_mac_text_size,
                       const char* data_to_seal, uint32_t data_to_seal_size,
                       uint8_t* output, uint32_t output_buf_size) {
  uint32_t sealed_data_size =
      get_sealed_data_size(aad_mac_text_size, data_to_seal_size);
  // error with sgx_calc_sealed_data_size
  if (sealed_data_size == UINT32_MAX) return SGX_ERROR_UNEXPECTED;
  // error with insufficient memory to host the result
  if (sealed_data_size > output_buf_size) return SGX_ERROR_INVALID_PARAMETER;

  uint8_t* temp_sealed_buf = (uint8_t*)malloc(sealed_data_size);
  if (temp_sealed_buf == NULL) return SGX_ERROR_OUT_OF_MEMORY;
  sgx_status_t status =
      sgx_seal_data(aad_mac_text_size, (const uint8_t*)aad_mac_text,
                    data_to_seal_size, (const uint8_t*)data_to_seal,
                    sealed_data_size, (sgx_sealed_data_t*)temp_sealed_buf);
  if (status == SGX_SUCCESS) memcpy(output, temp_sealed_buf, sealed_data_size);
  free(temp_sealed_buf);
  return status;
}

sgx_status_t unseal_data(const uint8_t* sealed_data, uint32_t data_size,
                         const char* aad_mac_text, uint32_t aad_mac_text_size,
                         char* output, uint32_t output_buf_size) {
  uint32_t mac_text_len =
      sgx_get_add_mac_txt_len((const sgx_sealed_data_t*)sealed_data);
  uint32_t decrypt_data_len = get_unsealed_data_size(sealed_data);
  if (mac_text_len == UINT32_MAX || decrypt_data_len == UINT32_MAX) {
    return SGX_ERROR_UNEXPECTED;
  }
  // is this checking really needed? i.e. decrypted length in each part must be
  // smaller than the original sealed data length
  if (mac_text_len > data_size || decrypt_data_len > data_size) {
    return SGX_ERROR_INVALID_PARAMETER;
  }
  // error with insufficient memory to host the result
  if (decrypt_data_len > output_buf_size) return SGX_ERROR_INVALID_PARAMETER;

  uint8_t* de_mac_text = (uint8_t*)malloc(mac_text_len);
  if (de_mac_text == NULL) return SGX_ERROR_OUT_OF_MEMORY;
  uint8_t* temp_unsealed_buf = (uint8_t*)malloc(decrypt_data_len);
  if (temp_unsealed_buf == NULL) {
    free(de_mac_text);
    return SGX_ERROR_OUT_OF_MEMORY;
  }

  sgx_status_t status =
      sgx_unseal_data((const sgx_sealed_data_t*)sealed_data, de_mac_text,
                      &mac_text_len, temp_unsealed_buf, &decrypt_data_len);
  if (status == SGX_SUCCESS) {
    // debug checking that aad mac dummy matches
    if ((mac_text_len == aad_mac_text_size) &&
        (memcmp(de_mac_text, aad_mac_text, mac_text_len) == 0)) {
      memcpy(output, temp_unsealed_buf, decrypt_data_len);
    }
  }
  free(de_mac_text);
  free(temp_unsealed_buf);
  return status;
}

}  // namespace hakes
