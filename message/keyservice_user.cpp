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

#include "keyservice_user.h"

#include <cassert>

#include "utils/json.h"

namespace hakes {

std::string KeyServiceRequest::EncodeTo() const {
  json::JSON ret;
  ret["type"] = type_;
  ret["user_id"] = user_id_;
  ret["payload"] = payload_;
  return ret.dump();
}

KeyServiceRequest DecodeKeyServiceRequest(const std::string& request) {
  auto src_json = json::JSON::Load(request);
  assert(src_json.hasKey("type"));
  assert(src_json.hasKey("user_id"));
  assert(src_json.hasKey("payload") ||
         src_json["type"].ToInt() == USER_REGISTER);
  return {
      (int)src_json["type"].ToInt(), src_json["user_id"].ToString(),
      ((src_json["type"].ToInt() == USER_REGISTER)
           ? ""
           : src_json["payload"].ToString())  // empty string if for register
  };
}

std::string KeyServiceReply::EncodeTo() const {
  json::JSON ret;
  ret["success"] = success_;
  ret["reply"] = reply_;
  return ret.dump();
}

KeyServiceReply DecodeKeyServiceReply(const std::string& reply) {
  auto src_json = json::JSON::Load(reply);
  assert(src_json.hasKey("success"));
  assert(src_json.hasKey("reply"));
  return {src_json["success"].ToBool(), src_json["reply"].ToString()};
}

std::string AddRequestKeyRequest::EncodeTo() const {
  json::JSON ret;
  ret["model_id"] = model_id_;
  ret["mrenclave"] = mrenclave_;
  ret["decrypt_key"] = decrypt_key_;
  return ret.dump();
}

AddRequestKeyRequest DecodeAddRequestKeyRequest(const std::string& request) {
  auto src_json = json::JSON::Load(request);
  assert(src_json.hasKey("model_id"));
  assert(src_json.hasKey("mrenclave"));
  assert(src_json.hasKey("decrypt_key"));
  return {src_json["model_id"].ToString(), src_json["mrenclave"].ToString(),
          src_json["decrypt_key"].ToString()};
}

std::string UpsertModelKeyRequest::EncodeTo() const {
  json::JSON ret;
  ret["model_id"] = model_id_;
  ret["decrypt_key"] = decrypt_key_;
  return ret.dump();
}

UpsertModelKeyRequest DecodeUpsertModelKeyRequest(const std::string& request) {
  auto src_json = json::JSON::Load(request);
  assert(src_json.hasKey("model_id"));
  assert(src_json.hasKey("decrypt_key"));
  return {src_json["model_id"].ToString(), src_json["decrypt_key"].ToString()};
}

std::string GrantModelAccessRequest::EncodeTo() const {
  json::JSON ret;
  ret["model_id"] = model_id_;
  ret["mrenclave"] = mrenclave_;
  ret["user_id"] = user_id_;
  return ret.dump();
}

GrantModelAccessRequest DecodeGrantModelAccessRequest(
    const std::string& request) {
  auto src_json = json::JSON::Load(request);
  assert(src_json.hasKey("model_id"));
  assert(src_json.hasKey("mrenclave"));
  assert(src_json.hasKey("user_id"));
  return {src_json["model_id"].ToString(), src_json["mrenclave"].ToString(),
          src_json["user_id"].ToString()};
}

}  // namespace hakes
