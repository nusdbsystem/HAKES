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

#include "embed-endpoint/openai_endpoint.h"

#include <cstring>

#include "utils/hexutil.h"
#include "utils/json.h"

namespace hakes_embedendpoint {

namespace {

OpenAIConfig ParseOpenAIConfig(const std::string& cfg) {
  auto j = json::JSON::Load(cfg);

  // get OpenAI API key
  if (!j.hasKey("endpoint")) {
    throw std::invalid_argument("endpoint is not found in the config");
  }
  if (!j.hasKey("api_key")) {
    throw std::invalid_argument("api key is not found in the config");
  }
  if (!j.hasKey("model_name")) {
    throw std::invalid_argument("model name is not a string");
  }

  return OpenAIConfig{j["endpoint"].ToString(), j["api_key"].ToString(),
                      j["model_name"].ToString()};
}

std::string EncodeOpenAIRequest(const std::string& input,
                                const std::string& model_name) {
  json::JSON ret;
  ret["input"] = input;
  ret["model"] = model_name;
  return ret.dump();
}

std::string ExtractVectorFromResponse(const std::string& response) {
  auto j = json::JSON::Load(response);
  if (!j.hasKey("data")) {
    throw std::invalid_argument(
        "embedding vector is not found in the response");
  }
  int data_count = j["data"].size();
  int d = 0;
  for (auto& value : j["data"].ArrayRange()) {
    if (!value.hasKey("embedding")) {
      throw std::invalid_argument(
          "embedding vector is not found in the response");
    }
    d += value["embedding"].size();
  }

  std::unique_ptr<char[]> vecs = std::unique_ptr<char[]>(new char[4 * d]);
  int idx = 0;
  for (auto& value : j["data"].ArrayRange()) {
    for (auto& v : value["embedding"].ArrayRange()) {
      float f = static_cast<float>(v.ToFloat());
      std::memcpy(vecs.get() + idx, &f, sizeof(float));
      idx += sizeof(float);
    }
  }
  return hakes::hex_encode(vecs.get(), 4 * d);
}

}  // anonymous namespace

OpenAIEndpoint::OpenAIEndpoint(const std::string& config)
    : config_(ParseOpenAIConfig(config)), http_() {}

bool OpenAIEndpoint::Initialize() { return true; }

bool OpenAIEndpoint::HandleEmbedOp(uint64_t handle_id, const std::string& data,
                                   std::string* vecs) {
  std::string request = EncodeOpenAIRequest(data, config_.model_name);
  http_.claim();
  printf("api_key: %s\n", config_.api_key.c_str());
  auto response = http_.post(config_.endpoint, request,
                             "Authorization: Bearer " + config_.api_key);
  http_.release();
  printf("response: %s\n", response.c_str());

  vecs->assign(ExtractVectorFromResponse(response));
  return true;
}

void OpenAIEndpoint::Close() {}

}  // namespace hakes_embedendpoint
