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

#ifndef HAKES_MESSAGE_KEYSERVICE_USER_H_
#define HAKES_MESSAGE_KEYSERVICE_USER_H_

#include <string>

namespace hakes {

enum KeyServiceRequestType : int {
  USER_REGISTER,
  ADD_REQUEST_KEY,
  UPSERT_MODEL_KEY,
  GRANT_MODEL_ACCESS
};

struct KeyServiceRequest {
  KeyServiceRequest(int request_type, const std::string& user_id,
                    const std::string& payload)
      : type_(request_type), user_id_(user_id), payload_(payload) {}
  ~KeyServiceRequest() = default;
  std::string EncodeTo() const;

  int type_;
  std::string user_id_;
  std::string payload_;
};

extern KeyServiceRequest DecodeKeyServiceRequest(const std::string& request);

struct KeyServiceReply {
  KeyServiceReply(bool success, const std::string& reply)
      : success_(success), reply_(reply) {}
  ~KeyServiceReply() = default;
  std::string EncodeTo() const;
  bool success_;  // fail if the proposed_id_ has been taken.
  std::string reply_;
};

extern KeyServiceReply DecodeKeyServiceReply(const std::string& reply);

struct AddRequestKeyRequest {
  AddRequestKeyRequest(const std::string& model_id,
                       const std::string& mrenclave,
                       const std::string& decrypt_key)
      : model_id_(model_id), mrenclave_(mrenclave), decrypt_key_(decrypt_key) {}
  ~AddRequestKeyRequest() = default;
  std::string EncodeTo() const;

  std::string model_id_;
  std::string mrenclave_;
  std::string decrypt_key_;
};

extern AddRequestKeyRequest DecodeAddRequestKeyRequest(
    const std::string& request);

struct UpsertModelKeyRequest {
  UpsertModelKeyRequest(const std::string& model_id,
                        const std::string& decrypt_key)
      : model_id_(model_id), decrypt_key_(decrypt_key) {}
  ~UpsertModelKeyRequest() = default;
  std::string EncodeTo() const;

  std::string model_id_;
  std::string decrypt_key_;
};

extern UpsertModelKeyRequest DecodeUpsertModelKeyRequest(
    const std::string& request);

struct GrantModelAccessRequest {
  GrantModelAccessRequest(const std::string& model_id,
                          const std::string& mrenclave,
                          const std::string& user_id)
      : model_id_(model_id), mrenclave_(mrenclave), user_id_(user_id) {}
  ~GrantModelAccessRequest() = default;
  std::string EncodeTo() const;

  std::string model_id_;
  std::string mrenclave_;
  std::string user_id_;
};

extern GrantModelAccessRequest DecodeGrantModelAccessRequest(
    const std::string& request);

}  // namespace hakes

#endif  // HAKES_MESSAGE_KEYSERVICE_USER_H_
