/*
 * Copyright 2024 The HAKES Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "embed-endpoint/ollama_endpoint.h"

#include <cstring>

#include "utils/hexutil.h"
#include "utils/json.h"

namespace hakes_embedendpoint {

namespace {

OllamaConfig ParseOllamaConfig(const std::string& cfg) {
  auto j = json::JSON::Load(cfg);

  // get Ollama base URL
  if (!j.hasKey("base_url")) {
    throw std::invalid_argument("base_url is not found in the config");
  }
  if (!j.hasKey("model")) {
    throw std::invalid_argument("model is not found in the config");
  }

  return OllamaConfig{j["base_url"].ToString(), j["model"].ToString()};
}

std::string EncodeOllamaRequest(const std::string& input, const std::string& model) {
  json::JSON ret;
  ret["model"] = model;
  ret["input"] = input;
  return ret.dump();
}

std::string ExtractVectorFromResponse(const std::string& response) {
  auto j = json::JSON::Load(response);
  
  // Ollama returns "embeddings" format (batch)
  if (j.hasKey("embeddings")) {
    int total_d = 0;
    int batch_size = j["embeddings"].size();
     
    // all embeddings have the same dimension
    total_d = j["embeddings"][0].size() * batch_size;
    
    std::unique_ptr<char[]> vecs = std::unique_ptr<char[]>(new char[4 * total_d]);
    int idx = 0;
    for (auto& embedding : j["embeddings"].ArrayRange()) {
      for (auto& value : embedding.ArrayRange()) {
        float f = static_cast<float>(value.ToFloat());
        std::memcpy(vecs.get() + idx, &f, sizeof(float));
        idx += sizeof(float);
      }
    }
    return hakes::hex_encode(vecs.get(), 4 * total_d);
  } else {
    throw std::invalid_argument(
        "'embeddings' field not found in the response");
  }
}

}  // anonymous namespace

OllamaEndpoint::OllamaEndpoint(const std::string& config)
    : config_(ParseOllamaConfig(config)), http_() {}

bool OllamaEndpoint::Initialize() { return true; }

bool OllamaEndpoint::HandleEmbedOp(uint64_t handle_id,
                                   const std::string& data,
                                   std::string* vecs) {
  std::string request = EncodeOllamaRequest(data, config_.model);
  http_.claim();
  
  // Build the full URL for Ollama embeddings API (uses /api/embed, not /api/embeddings)
  std::string url = config_.base_url + "/api/embed";
  
  // Prepare headers
  std::string headers = "Content-Type: application/json";
  
  printf("Ollama URL: %s\n", url.c_str());
  printf("Ollama request: %s\n", request.c_str());
  
  auto response = http_.post(url, request, headers);
  http_.release();
  
  printf("Ollama response: %s\n", response.c_str());

  vecs->assign(ExtractVectorFromResponse(response));
  return true;
}

void OllamaEndpoint::Close() {}

}  // namespace hakes_embedendpoint 