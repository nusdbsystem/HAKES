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

#ifndef HAKES_UTILS_OWMESSAGE_H_
#define HAKES_UTILS_OWMESSAGE_H_

#include <cstdio>
#include <string>

#include "utils/json.h"

namespace ow_message {

constexpr char log_sentinel[] = "XXX_THE_END_OF_A_WHISK_ACTIVATION_XXX";

inline void flush_openwhisk_log_sentinel() {
  printf("%s\n", log_sentinel);
  fflush(stdout);
  printf("%s\n", log_sentinel);
  fflush(stdout);
}

inline std::string extract_ow_input(const std::string& platform_message) {
  auto input = json::JSON::Load(platform_message);
  return input["value"].ToString();
}

inline std::string package_ow_response(bool error,
                                       const std::string& response, bool is_response_json) {
  flush_openwhisk_log_sentinel();
  if (is_response_json) {
    return error ? "{\"error\": " + std::move(response) + "}"
                 : "{\"msg\": " + std::move(response) + "}";
  }
  return error ? "{\"error\": \"" + std::move(response) + "\"}"
               : "{\"msg\": \"" + std::move(response) + "\"}";
}

inline std::string extract_ow_response(const std::string& platform_message) {
  auto input = json::JSON::Load(platform_message);
  if (input.hasKey("error")) {
    return input["error"].ToString();
  } else if (input.hasKey("msg")) {
    return input["msg"].ToString();
  } else {
    return "{\"error\": \"invalid ow response\"}";
  }
}

}  // namespace ow_message

#endif  // HAKES_UTILS_OWMESSAGE_H_
