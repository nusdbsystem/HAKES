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

#include "embed.h"

#include <cassert>
#include <fstream>

#include "utils/base64.h"
#include "utils/json.h"

namespace hakes {

std::string EmbedWorkerRequest::EncodeTo() const {
  json::JSON ret;
  ret["encrypted_sample"] = encrypted_sample_;
  ret["model_type"] = model_name_;
  ret["user_id"] = user_id_;
  ret["key_service_address"] = key_service_address_;
  ret["key_service_port"] = key_service_port_;
  return ret.dump();
}

EmbedWorkerRequest DecodeEmbedWorkerRequest(const std::string& request) {
  auto input = json::JSON::Load(request);
  auto encrypted_sample_src = input["encrypted_sample"].ToString();
  EmbedWorkerRequest ret;
  ret.encrypted_sample_ = base64_decode(
      (const uint8_t*)encrypted_sample_src.data(), encrypted_sample_src.size());
  ret.model_name_ = input["model_type"].ToString();
  ret.user_id_ = input["user_id"].ToString();
  ret.key_service_address_ = input["key_service_address"].ToString();
  assert(input["key_service_port"].ToInt() < UINT16_MAX);
  ret.key_service_port_ =
      static_cast<uint16_t>(input["key_service_port"].ToInt());
  return ret;
}

std::string EmbedWorkerResponse::EncodeTo() const {
  json::JSON ret;
  ret["status"] = status;
  ret["output"] = output;
  if (!aux.empty()) {
    ret["aux"] = aux;
  }
  return ret.dump();
}

EmbedWorkerResponse DecodeEmbedWorkerResponse(const std::string& response) {
  auto input = json::JSON::Load(response);
  EmbedWorkerResponse ret;
  ret.status = input["status"].ToBool();
  if (input.hasKey("output")) {
    ret.output = input["output"].ToString();
  }
  if (input.hasKey("aux")) {
    ret.aux = input["aux"].ToString();
  }
  return ret;
}

std::string EmbedFnPackerRequest::EncodeTo() const {
  json::JSON req;
  req["encrypted_sample"] = encrypted_sample_;
  req["model_type"] = model_name_;
  req["user_id"] = user_id_;
  req["key_service_address"] = key_service_address_;
  req["key_service_port"] = key_service_port_;
  json::JSON ret;
  ret["name"] = name;
  ret["request"] = req;
  return ret.dump();
}

EmbedFnPackerRequest DecodeEmbedFnPackerRequest(const std::string& request) {
  auto input = json::JSON::Load(request);
  EmbedFnPackerRequest ret;
  if (!input.hasKey("name") || !input.hasKey("request")) {
    return ret;
  }
  auto req = input["request"];
  if (!req.hasKey("encrypted_sample") || !req.hasKey("model_type") ||
      !req.hasKey("user_id") || !req.hasKey("key_service") ||
      !req.hasKey("key_service_port")) {
    return ret;
  }
  auto encrypted_sample_src = req["encrypted_sample"].ToString();
  ret.encrypted_sample_ = base64_decode(
      (const uint8_t*)encrypted_sample_src.data(), encrypted_sample_src.size());
  ret.model_name_ = req["model_type"].ToString();
  ret.user_id_ = req["user_id"].ToString();
  ret.key_service_address_ = req["key_service_address"].ToString();
  assert(req["key_service_port"].ToInt() < UINT16_MAX);
  ret.key_service_port_ =
      static_cast<uint16_t>(req["key_service_port"].ToInt());
  ret.name = input["name"].ToString();
  return ret;
}

std::string EmbedFnPackerResponse::EncodeTo() const {
  json::JSON ret;
  if (!status) {
    ret["error"] = output;
    return ret.dump();
  }
  json::JSON msg;
  msg["status"] = status;
  msg["output"] = output;
  if (!aux.empty()) {
    msg["aux"] = aux;
  }
  ret["msg"] = msg;
  return ret.dump();
}

EmbedFnPackerResponse DecodeEmbedFnPackerResponse(const std::string& response) {
  auto input = json::JSON::Load(response);
  EmbedFnPackerResponse ret;
  if (!input.hasKey("msg")) {
    if (input.hasKey("error")) {
      ret.status = false;
      ret.output = input["error"].ToString();
    } else {
      ret.status = false;
      ret.output = response;
    }
  } else {
    auto msg = input["msg"];
    if (!msg.hasKey("status")) {
      ret.status = false;
      ret.output = response;
      return ret;
    }
    ret.status = msg["status"].ToBool();
    if (msg.hasKey("output")) {
      ret.output = msg["output"].ToString();
    }
    if (msg.hasKey("aux")) {
      ret.aux = msg["aux"].ToString();
    }
  }
  return ret;
}

}  // namespace hakes
