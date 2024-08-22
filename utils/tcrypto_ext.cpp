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

#include "tcrypto_ext.h"

#include <cstring>

namespace hakes {
namespace {
// for now we use fixed IV and aad for prototype
// for production, different value should be used for encryption (need to
// support update and thread synchronization on them)
uint8_t enc_iv_buf[AES_GCM_IV_SIZE] = {0};
uint8_t enc_aad_buf[AES_GCM_AAD_SIZE] = {0};
}  // anonymous namespace

/**
 * @brief
 *
 * @param src
 * @param size
 * @param decryption_key
 * @param output caller is responsible to free the resources
 * @return sgx_status_t
 */
sgx_status_t decrypt_content_with_key_aes(const uint8_t* src, size_t size,
                                          const uint8_t* decryption_key,
                                          uint8_t** output) {
  size_t cipher_text_size = get_aes_decrypted_size(size);

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

/**
 * @brief
 *
 * @param src
 * @param size
 * @param encryption_key
 * @param output caller is responsible to free the resources.
 * @return sgx_status_t
 */
sgx_status_t encrypt_content_with_key_aes(const uint8_t* src, size_t size,
                                          const uint8_t* encryption_key,
                                          uint8_t** output) {
  size_t whole_cipher_text_size = get_aes_encrypted_size(size);

  // allocate temporary buffers for decryption operation. freed at the end.
  uint8_t* whole_cipher_text = (uint8_t*)malloc(whole_cipher_text_size);
  if (whole_cipher_text == NULL) return SGX_ERROR_OUT_OF_MEMORY;

  // temporary variables
  uint8_t tag_buf[AES_GCM_TAG_SIZE] = {0};

  // encrypt
  sgx_status_t ret = sgx_rijndael128GCM_encrypt(
      (const sgx_aes_gcm_128bit_key_t*)encryption_key, src, (uint32_t)size,
      whole_cipher_text, enc_iv_buf, AES_GCM_IV_SIZE, enc_aad_buf,
      AES_GCM_AAD_SIZE, (sgx_aes_gcm_128bit_tag_t*)tag_buf);

  // free the resource when failure.
  if (ret != SGX_SUCCESS) {
    free(whole_cipher_text);
    return ret;
  }

  // assemble the message
  uint8_t* pos = whole_cipher_text + size;
  memcpy(pos, enc_iv_buf, AES_GCM_IV_SIZE);
  pos += AES_GCM_IV_SIZE;
  memcpy(pos, tag_buf, AES_GCM_TAG_SIZE);
  pos += AES_GCM_TAG_SIZE;
  memcpy(pos, enc_aad_buf, AES_GCM_AAD_SIZE);

  // assign the result to output if success;
  *output = whole_cipher_text;

  return ret;
}

}  // namespace hakes
