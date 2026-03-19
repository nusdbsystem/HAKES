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

// removed the old intel code that uses openssl. use wolfssl produced by ratls.
// The codes is needed by worker enclave.
#include "base64.h"

#include <openssl/bio.h>
#include <openssl/evp.h>

#include <cstring>

namespace hakes {

std::string base64_encode(const uint8_t *src, size_t src_size) {
  BIO *b64, *bmem;
  char *bstr;
  int len;

  b64 = BIO_new(BIO_f_base64());
  bmem = BIO_new(BIO_s_mem());

  /* Single line output, please */
  BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);

  BIO_push(b64, bmem);

  if (BIO_write(b64, src, (int)src_size) == -1) {
    BIO_free(bmem);
    BIO_free(b64);
    return NULL;
  }

  BIO_flush(b64);

  len = BIO_get_mem_data(bmem, &bstr);
  std::string ret(bstr, len);

  BIO_free(bmem);
  BIO_free(b64);

  return ret;
}

std::string base64_decode(const uint8_t *src, size_t src_size) {
  BIO *b64, *bmem;
  char *buf;

  buf = (char *)malloc(src_size + 1);
  if (buf == NULL) return NULL;
  memset(buf, 0, src_size + 1);

  b64 = BIO_new(BIO_f_base64());
  BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);

  bmem = BIO_new_mem_buf(src, (int)src_size);

  BIO_push(b64, bmem);

  auto output_len = BIO_read(b64, buf, (int)src_size);
  if (output_len == -1) {
    free(buf);
    return NULL;
  }

  BIO_free_all(bmem);

  return std::string(buf, output_len);
}

}  // namespace hakes
