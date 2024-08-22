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

#include "crypto_ext.h"

#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>

#include <cstring>

// for now we use fixed IV and aad for prototype
uint8_t enc_iv_buf[AES_GCM_IV_SIZE] = {0};
uint8_t enc_aad_buf[AES_GCM_AAD_SIZE] = {0};

namespace hakes {

std::string encrypt_content_with_key_aes(const uint8_t* src, size_t size,
                                         const std::string& encryption_key) {
  EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
  EVP_EncryptInit_ex(
      ctx, EVP_aes_128_gcm(), NULL,
      reinterpret_cast<const unsigned char*>(encryption_key.c_str()),
      reinterpret_cast<const unsigned char*>(enc_iv_buf));

  int len;
  size_t cipher_text_size = get_aes_encrypted_size(size);
  unsigned char encrypted[cipher_text_size];
  memset(encrypted, 0, cipher_text_size);
  EVP_EncryptUpdate(ctx, NULL, &len, enc_aad_buf, AES_GCM_AAD_SIZE);
  EVP_EncryptUpdate(ctx, encrypted, &len, src, size);
  EVP_EncryptFinal_ex(ctx, encrypted + len, &len);
  unsigned char tag[16];
  EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, tag);
  EVP_CIPHER_CTX_free(ctx);
  printf("Tag:\n");
  BIO_dump_fp(stdout, (const char*)tag, 16);

  memcpy(encrypted + size, enc_iv_buf, AES_GCM_IV_SIZE);
  memcpy(encrypted + size + AES_GCM_IV_SIZE, tag, AES_GCM_TAG_SIZE);
  memcpy(encrypted + size + AES_GCM_IV_SIZE + AES_GCM_TAG_SIZE, enc_aad_buf,
         AES_GCM_AAD_SIZE);

  return std::string(reinterpret_cast<char*>(encrypted), cipher_text_size);
}

std::string decrypt_content_with_key_aes(const uint8_t* src, size_t size,
                                         const std::string& decryption_key) {
  uint8_t iv_buf[AES_GCM_IV_SIZE] = {0};
  uint8_t tag_buf[AES_GCM_TAG_SIZE] = {0};
  uint8_t aad_buf[AES_GCM_AAD_SIZE] = {0};

  size_t cipher_text_size = get_aes_decrypted_size(size);
  {
    const uint8_t* p = src;
    p += cipher_text_size;
    memcpy(iv_buf, p, AES_GCM_IV_SIZE);
    p += AES_GCM_IV_SIZE;
    memcpy(tag_buf, p, AES_GCM_TAG_SIZE);
    p += AES_GCM_TAG_SIZE;
    memcpy(aad_buf, p, AES_GCM_AAD_SIZE);
  }

  EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
  EVP_DecryptInit_ex(
      ctx, EVP_aes_128_gcm(), NULL,
      reinterpret_cast<const unsigned char*>(decryption_key.c_str()),
      reinterpret_cast<const unsigned char*>(iv_buf));
  EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, 16, tag_buf);

  int plaintext_len;
  int len;
  unsigned char decrypted[cipher_text_size];
  EVP_DecryptUpdate(ctx, decrypted, &len,
                    reinterpret_cast<const unsigned char*>(src),
                    cipher_text_size);
  EVP_CIPHER_CTX_free(ctx);

  return std::string{reinterpret_cast<char*>(decrypted), len};
}

}  // namespace hakes
