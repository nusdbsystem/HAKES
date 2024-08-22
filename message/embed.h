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

#ifndef HAKES_MESSAGE_EMBED_H_
#define HAKES_MESSAGE_EMBED_H_

#include <memory>
#include <string>

namespace hakes {

struct EmbedWorkerRequest {
  EmbedWorkerRequest() = default;
  ~EmbedWorkerRequest() = default;

  EmbedWorkerRequest(const EmbedWorkerRequest& other) = default;
  EmbedWorkerRequest(EmbedWorkerRequest&& other) = default;
  EmbedWorkerRequest& operator=(const EmbedWorkerRequest& other) = default;
  EmbedWorkerRequest& operator=(EmbedWorkerRequest&& other) = default;

  /**
   * @brief encode the request into a string
   */
  std::string EncodeTo() const;

  std::string encrypted_sample_;
  std::string model_name_;
  std::string user_id_;
  std::string key_service_address_;
  uint16_t key_service_port_;
};

EmbedWorkerRequest DecodeEmbedWorkerRequest(const std::string& request);

struct EmbedWorkerResponse {
  std::string EncodeTo() const;

  bool status;
  std::string output;
  std::string aux;
};

EmbedWorkerResponse DecodeEmbedWorkerResponse(const std::string& response);

struct EmbedFnPackerRequest : EmbedWorkerRequest {
  std::string EncodeTo() const;

  std::string name;
};

EmbedFnPackerRequest DecodeEmbedFnPackerRequest(const std::string& request);

struct EmbedFnPackerResponse : EmbedWorkerResponse {
  std::string EncodeTo() const;
};

EmbedFnPackerResponse DecodeEmbedFnPackerResponse(const std::string& response);

}  // namespace hakes

#endif  // HAKES_MESSAGE_EMBED_H_
