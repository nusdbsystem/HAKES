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

#include "keyservice_worker.h"

#include <cassert>
#include <cstring>

#include "utils/base64.h"
#include "utils/json.h"

namespace hakes {

std::string GetKeyRequest::EncodeTo() const {
  json::JSON ret;
  ret["user_id"] = user_id_;
  ret["model_id"] = model_id_;
  return ret.dump();
};

GetKeyRequest GetKeyRequest::DecodeFrom(const std::string& src) {
  auto src_json = json::JSON::Load(src);
  assert(src_json.hasKey("user_id"));
  assert(src_json.hasKey("model_id"));
  return {src_json["user_id"].ToString(), src_json["model_id"].ToString()};
}

KeyServiceWorkerReply::KeyServiceWorkerReply(const std::string& user_id,
                                             const std::string& input_key,
                                             const std::string& model_id,
                                             const std::string& model_key)
    : success_(true) {
  json::JSON ret;
  ret["user_id"] = user_id;
  ret["input_key"] = input_key;
  ret["model_id"] = model_id;
  ret["model_key"] = model_key;
  msg_ = ret.dump();
}

std::string KeyServiceWorkerReply::EncodeTo() const {
  json::JSON ret;
  ret["success"] = success_;
  ret["msg"] = msg_;
  return ret.dump();
}

int DecodeKeyServiceWorkerReply(const std::string& reply, std::string* user_id,
                                std::string* input_key, std::string* model_id,
                                std::string* model_key) {
  auto reply_json = json::JSON::Load(reply);
  assert(reply_json.hasKey("success"));
  assert(reply_json.hasKey("msg"));
  if (!reply_json["success"].ToBool()) return -1;
  auto keys = json::JSON::Load(reply_json["msg"].ToString());
  assert(keys.hasKey("user_id"));
  assert(keys.hasKey("input_key"));
  assert(keys.hasKey("model_id"));
  assert(keys.hasKey("model_key"));
  *user_id = keys["user_id"].ToString();
  *input_key = keys["input_key"].ToString();
  *model_id = keys["model_id"].ToString();
  *model_key = keys["model_key"].ToString();
  return 0;
}

}  // namespace hakes
