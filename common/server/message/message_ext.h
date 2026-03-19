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

#ifndef HAKES_MESSAGE_MESSAGE_EXT_H_
#define HAKES_MESSAGE_MESSAGE_EXT_H_
// for V3

#include "message.h"

namespace hakes {

struct ExtendedAddRequest : public AddRequest {
  bool add_to_refine_only = false;
  bool assigned = false;
  int64_t* assign = nullptr;
  std::unique_ptr<int64_t[]> assign_holder;
  int index_version = -1;
};

void encode_extended_add_request(const ExtendedAddRequest& request,
                                 std::string* data);
bool decode_extended_add_request(const std::string& request_str,
                                 ExtendedAddRequest* request);

struct ExtendedAddResponse : public AddResponse {
  int n = 0;
  int64_t* assign = nullptr;
  std::unique_ptr<int64_t[]> assign_holder;
  int vecs_t_d = 0;
  float_t* vecs_t = nullptr;
  std::unique_ptr<float_t[]> vecs_t_holder;
  int index_version = -1;
};

void encode_extended_add_response(const ExtendedAddResponse& response,
                                  std::string* data);
bool decode_extended_add_response(const std::string& response_str,
                                  ExtendedAddResponse* response);

struct RerankRequest {
  int n = 0;
  int d = 0;
  float* vecs = nullptr;
  std::unique_ptr<float[]> vecs_holder;
  int k = 0;
  uint8_t metric_type = 1;
  int64_t* k_base_count = nullptr;
  std::unique_ptr<int64_t[]> k_base_count_holder;
  int64_t* base_labels = nullptr;
  std::unique_ptr<int64_t[]> base_labels_holder;
  float* base_distances = nullptr;
  std::unique_ptr<float[]> base_distances_holder;
};

void encode_rerank_request(const RerankRequest& request, std::string* data);
bool decode_rerank_request(const std::string& request_str,
                           RerankRequest* request);

struct GetIndexResponse {
  bool status;
  std::string msg;
  int index_version;
  std::string params;
};

void encode_get_index_response(const GetIndexResponse& request,
                               std::string* data);
bool decode_get_index_response(const std::string& response_str,
                               GetIndexResponse* response);

struct UpdateIndexRequest {
  std::string params;
};

void encode_update_index_request(const UpdateIndexRequest& request,
                                 std::string* data);
bool decode_update_index_request(const std::string& request_str,
                                 UpdateIndexRequest* request);

struct UpdateIndexResponse {
  bool status;
  std::string msg;
  int index_version;
};

void encode_update_index_response(const UpdateIndexResponse& request,
                                  std::string* data);
bool decode_update_index_response(const std::string& response_str,
                                  UpdateIndexResponse* response);

}  // namespace hakes

#endif  // HAKES_MESSAGE_MESSAGE_EXT_H_
