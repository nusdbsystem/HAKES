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

#ifndef HAKES_MESSAGE_KEYSERVICE_WORKER_H_
#define HAKES_MESSAGE_KEYSERVICE_WORKER_H_

#include <string>

namespace hakes {
class GetKeyRequest {
 public:
  GetKeyRequest(const std::string& user_id, const std::string& model_id)
      : user_id_(user_id), model_id_(model_id) {}
  ~GetKeyRequest() = default;

  std::string EncodeTo() const;

  static GetKeyRequest DecodeFrom(const std::string& src);

  inline const std::string& user_id() const { return user_id_; }
  inline const std::string& model_id() const { return model_id_; }

 private:
  const std::string user_id_;
  const std::string model_id_;
};

struct KeyServiceWorkerReply {
  KeyServiceWorkerReply(const std::string& error_msg)
      : success_(false), msg_(error_msg) {}
  KeyServiceWorkerReply(const std::string& user_id,
                        const std::string& input_key,
                        const std::string& model_id,
                        const std::string& model_key);
  ~KeyServiceWorkerReply() = default;
  std::string EncodeTo() const;
  bool success_;
  std::string msg_;
};

extern int DecodeKeyServiceWorkerReply(const std::string& reply,
                                       std::string* user_id,
                                       std::string* input_key,
                                       std::string* model_id,
                                       std::string* model_key);

}  // namespace hakes

#endif  // HAKES_MESSAGE_KEYSERVICE_WORKER_H_
