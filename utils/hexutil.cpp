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

#include "hexutil.h"

#include <cstdint>
#include <cstdlib>
#include <memory>

namespace hakes {
namespace {
const char _hextable[] = "0123456789abcdef";
}  // anonymous namespace

std::string hex_encode(const void* vsrc, size_t len) {
  const char* src = (const char*)vsrc;
  char ret[len * 2 + 1];
  char* bp = ret;
  for (size_t i = 0; i < len; ++i) {
    *bp = _hextable[(uint8_t)src[i] >> 4];
    ++bp;
    *bp = _hextable[(uint8_t)src[i] & 0xf];
    ++bp;
  }
  ret[len * 2] = 0;
  return std::string(ret, len * 2);
}

std::string hex_decode(const char* vsrc, size_t len) {
  const char* src = (const char*)vsrc;
  char ret[len / 2];
  char* bp = ret;
  for (size_t i = 0; i < len; i += 2) {
    *bp = (uint8_t)((src[i] >= 'a' ? src[i] - 'a' + 10 : src[i] - '0') << 4);
    *bp |=
        (uint8_t)(src[i + 1] >= 'a' ? src[i + 1] - 'a' + 10 : src[i + 1] - '0');
    ++bp;
  }
  return std::string(ret, len / 2);
}

}  // namespace hakes
