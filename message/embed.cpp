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
  ret["data"] = data_;
  ret["model_type"] = model_name_;
  return ret.dump();
}

EmbedWorkerRequest DecodeEmbedWorkerRequest(const std::string& request) {
  auto input = json::JSON::Load(request);
  auto data = input["data"].ToString();
  EmbedWorkerRequest ret;
  ret.data_ = base64_decode(
      (const uint8_t*)data.data(), data.size());
  ret.model_name_ = input["model_type"].ToString();
  return ret;
}

std::string EmbedWorkerResponse::EncodeTo() const {
  json::JSON ret;
  ret["status"] = status;
  ret["output"] = output;
  return ret.dump();
}

EmbedWorkerResponse DecodeEmbedWorkerResponse(const std::string& response) {
  auto input = json::JSON::Load(response);
  EmbedWorkerResponse ret;
  ret.status = input["status"].ToBool();
  if (input.hasKey("output")) {
    ret.output = input["output"].ToString();
  }
  return ret;
}

std::string EmbedFnPackerRequest::EncodeTo() const {
  json::JSON req;
  req["data"] = data_;
  req["model_type"] = model_name_;
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
  if (!req.hasKey("data") || !req.hasKey("model_type") ||
      !req.hasKey("user_id") || !req.hasKey("key_service") ||
      !req.hasKey("key_service_port")) {
    return ret;
  }
  auto data = req["data"].ToString();
  ret.data_ = base64_decode(
      (const uint8_t*)data.data(), data.size());
  ret.model_name_ = req["model_type"].ToString();
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
  }
  return ret;
}

}  // namespace hakes
