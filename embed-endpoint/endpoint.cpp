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

#include <embed-endpoint/endpoint.h>

// supported external endpoints
#include "huggingface_endpoint.h"
#include "openai_endpoint.h"

namespace hakes_embedendpoint {
std::unique_ptr<EmbedEndpoint> CreateEmbedEndpoint(
    const std::string& embed_endpoint_type,
    const std::string& embed_endpoint_config) {
  if (embed_endpoint_type == "openai") {
    return std::make_unique<OpenAIEndpoint>(embed_endpoint_config);
  } else if (embed_endpoint_type == "huggingface") {
    return std::make_unique<HuggingFaceEndpoint>(embed_endpoint_config);
  } else {
    printf("Error: unknown embed endpoint type: %s\n",
           embed_endpoint_type.c_str());
    return nullptr;
  }
}
}  // namespace hakes_embedendpoint
