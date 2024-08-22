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

#include "kvservice.h"

#include "utils/hexutil.h"
#include "utils/json.h"

namespace hakes {

std::string encode_kvservice_request(const KVServiceRequest& request) {
  json::JSON ret;
  ret["type"] = request.type;
  ret["keys"] = json::Array();
  for (auto& key : request.keys) {
    ret["keys"].append(std::move(key));
  }
  ret["values"] = json::Array();
  for (auto& value : request.values) {
    ret["values"].append(std::move(value));
  }
  return ret.dump();
}

bool decode_kvservice_request(const std::string& request_str,
                              KVServiceRequest* request) {
  auto input = json::JSON::Load(request_str);
  request->type = input["type"].ToString();
  request->keys.clear();
  for (auto& key : input["keys"].ArrayRange()) {
    request->keys.emplace_back(key.ToString());
  }
  request->values.clear();
  for (auto& value : input["values"].ArrayRange()) {
    request->values.emplace_back(value.ToString());
  }
  return true;
}

std::string encode_kvservice_response(const KVServiceResponse& response) {
  json::JSON ret;
  ret["status"] = response.status;
  ret["values"] = json::Array();
  for (auto& value : response.values) {
    ret["values"].append(std::move(value));
  }
  return ret.dump();
}

bool decode_kvservice_response(const std::string& response_str,
                               KVServiceResponse* response) {
  auto input = json::JSON::Load(response_str);
  response->status = input["status"].ToBool();
  response->values.clear();
  for (auto& value : input["values"].ArrayRange()) {
    response->values.emplace_back(value.ToString());
  }
  return true;
}

}  // namespace hakes
