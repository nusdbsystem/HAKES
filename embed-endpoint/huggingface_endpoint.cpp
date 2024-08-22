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

#include "embed-endpoint/huggingface_endpoint.h"

#include <cstring>

#include "utils/hexutil.h"
#include "utils/json.h"

namespace hakes_embedendpoint {

namespace {

HuggingFaceConfig ParseHuggingFaceConfig(const std::string& cfg) {
  auto j = json::JSON::Load(cfg);

  // get HuggingFace API key
  if (!j.hasKey("endpoint")) {
    throw std::invalid_argument("endpoint is not found in the config");
  }
  if (!j.hasKey("api_key")) {
    throw std::invalid_argument("api key is not found in the config");
  }

  return HuggingFaceConfig{j["endpoint"].ToString(), j["api_key"].ToString()};
}

std::string EncodeHuggingFaceRequest(const std::string& input) {
  json::JSON ret;
  ret["inputs"] = input;
  return ret.dump();
}

std::string ExtractVectorFromResponse(const std::string& response) {
  auto wrap_json = "{\"embedding\":" + response + "}";
  auto j = json::JSON::Load(wrap_json);
  int d = j["embedding"].size();
  std::unique_ptr<char[]> vecs = std::unique_ptr<char[]>(new char[4 * d]);
  int idx = 0;
  for (auto& value : j["embedding"].ArrayRange()) {
    float f = static_cast<float>(value.ToFloat());
    std::memcpy(vecs.get() + idx, &f, sizeof(float));
    idx += sizeof(float);
  }
  return hakes::hex_encode(vecs.get(), 4 * d);
}

}  // anonymous namespace

HuggingFaceEndpoint::HuggingFaceEndpoint(const std::string& config)
    : config_(ParseHuggingFaceConfig(config)), http_() {}

bool HuggingFaceEndpoint::Initialize() { return true; }

bool HuggingFaceEndpoint::HandleEmbedOp(uint64_t handle_id,
                                        const std::string& data,
                                        std::string* vecs) {
  std::string request = EncodeHuggingFaceRequest(data);
  http_.claim();
  printf("api_key: %s\n", config_.api_key.c_str());
  auto response = http_.post(config_.endpoint, request,
                             "Authorization: Bearer " + config_.api_key);
  http_.release();
  printf("response: %s\n", response.c_str());

  vecs->assign(ExtractVectorFromResponse(response));
  return true;
}

void HuggingFaceEndpoint::Close() {}

}  // namespace hakes_embedendpoint