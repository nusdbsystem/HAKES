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

#ifndef HAKES_EMBEDENDPOINT_OLLAMA_ENDPOINT_H_
#define HAKES_EMBEDENDPOINT_OLLAMA_ENDPOINT_H_

#include "embed-endpoint/endpoint.h"
#include "utils/http.h"

namespace hakes_embedendpoint {

struct OllamaConfig {
  std::string base_url;
  std::string model;
};

class OllamaEndpoint : public EmbedEndpoint {
 public:
  OllamaEndpoint(const std::string& config);
  virtual ~OllamaEndpoint() = default;

  // delete copy and move constructors and assigment operators
  OllamaEndpoint(const OllamaEndpoint&) = delete;
  OllamaEndpoint& operator=(const OllamaEndpoint&) = delete;
  OllamaEndpoint(OllamaEndpoint&&) = delete;
  OllamaEndpoint& operator=(OllamaEndpoint&&) = delete;

  bool Initialize() override;

  bool HandleEmbedOp(uint64_t handle_id, const std::string& data,
                     std::string* vecs) override;

  void Close() override;

 private:
  OllamaConfig config_;
  hakes::HttpClient http_;
};

}  // namespace hakes_embedendpoint

#endif  // HAKES_EMBEDENDPOINT_OLLAMA_ENDPOINT_H_ 