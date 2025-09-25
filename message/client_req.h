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

#ifndef HAKES_MESSAGE_CLIENT_H_
#define HAKES_MESSAGE_CLIENT_H_

// top level API used by client to communicate with the hakes worker

#include <memory>
#include <string>
#include <vector>

namespace hakes_worker {

enum HakesRequestDataType : uint8_t {
  kRaw = 0,
  kVector = 1,
};

struct HakesAddRequest {
  // embed part
  std::string model_name;
  // data part
  int n;
  int d;  // vector data dimension; set to -1 for raw data
  HakesRequestDataType data_type;
  std::string data;
  std::string ids;
};

void encode_hakes_add_request(const HakesAddRequest& request,
                              std::string* data);

bool decode_hakes_add_request(const std::string& request_str,
                              HakesAddRequest* request);

struct HakesAddResponse {
  bool status;
  std::string msg;
};

void encode_hakes_add_response(const HakesAddResponse& response,
                               std::string* data);

bool decode_hakes_add_response(const std::string& response_str,
                               HakesAddResponse* response);

struct HakeSearchRequest {
  // embed part
  std::string model_name;
  // search part
  int n;
  int d;  // vector data dimension; set to -1 for raw data
  int k;
  int nprobe;
  int k_factor;
  uint8_t metric_type;
  int index_version = -1;
  // data part
  HakesRequestDataType data_type;
  std::string data;
};

void encode_hakes_search_request(const HakeSearchRequest& request,
                                 std::string* data);

bool decode_hakes_search_request(const std::string& request_str,
                                 HakeSearchRequest* request);

struct HakesSearchResponse {
  bool status;
  std::string msg;
  std::string ids;
  std::string scores;
  std::vector<std::string> data;
};

void encode_hakes_search_response(const HakesSearchResponse& response,
                                  std::string* data);

bool decode_hakes_search_response(const std::string& response_str,
                                  HakesSearchResponse* response);

}  // namespace hakes_worker

#endif  // HAKES_MESSAGE_CLIENT_H_