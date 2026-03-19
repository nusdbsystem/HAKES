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

#ifndef HAKES_EMBEDENDPOINT_ENDPOINT_H_
#define HAKES_EMBEDENDPOINT_ENDPOINT_H_

#include <memory>
#include <string>

namespace hakes_embedendpoint {

class EmbedEndpoint {
 public:
  EmbedEndpoint() = default;
  virtual ~EmbedEndpoint() = default;

  // delete copy and move constructors and assigment operators
  EmbedEndpoint(const EmbedEndpoint&) = delete;
  EmbedEndpoint& operator=(const EmbedEndpoint&) = delete;
  EmbedEndpoint(EmbedEndpoint&&) = delete;
  EmbedEndpoint& operator=(EmbedEndpoint&&) = delete;

  virtual bool Initialize() = 0;

  virtual bool HandleEmbedOp(uint64_t handle_id, const std::string& data,
                             std::string* vecs) = 0;

  virtual void Close() = 0;
};

std::unique_ptr<EmbedEndpoint> CreateEmbedEndpoint(
    const std::string& embed_endpoint_type,
    const std::string& embed_endpoint_config);

}  // namespace hakes_embedendpoint

#endif  // HAKES_EMBEDENDPOINT_ENDPOINT_H_
