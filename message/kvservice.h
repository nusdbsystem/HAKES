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

#ifndef HAKES_MESSAGE_KVSERVICE_H_
#define HAKES_MESSAGE_KVSERVICE_H_

#include <string>
#include <vector>

namespace hakes {

struct KVServiceRequest {
  std::string type;
  std::vector<std::string> keys;
  std::vector<std::string> values;
};

std::string encode_kvservice_request(const KVServiceRequest& request);

bool decode_kvservice_request(const std::string& request_str,
                              KVServiceRequest* request);

struct KVServiceResponse {
  bool status;
  std::vector<std::string> values;
};

std::string encode_kvservice_response(const KVServiceResponse& response);

bool decode_kvservice_response(const std::string& response_str,
                               KVServiceResponse* response);

}  // namespace hakes

#endif  // HAKES_MESSAGE_KVSERVICE_H_
