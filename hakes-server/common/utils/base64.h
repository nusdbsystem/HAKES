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

#ifndef HAKES_UTILS_BASE64_H_
#define HAKES_UTILS_BASE64_H_

#include <string>

namespace hakes {

std::string base64_encode(const uint8_t* src, size_t src_size);
std::string base64_decode(const uint8_t* src, size_t src_size);

}  // namespace hakes

#endif  // HAKES_UTILS_BASE64_H_
