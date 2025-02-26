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

#ifndef HAKES_MESSAGE_SEARCHSERVICE_H_
#define HAKES_MESSAGE_SEARCHSERVICE_H_

#include <string>
#include <memory>

namespace hakes {


std::string encode_hex_floats(const float* vecs, size_t count);
std::unique_ptr<float[]> decode_hex_floats(const std::string& vecs_str, size_t* count);

std::string encode_hex_int64s(const int64_t* vecs, size_t count);
std::unique_ptr<int64_t[]> decode_hex_int64s(const std::string& vecs_str, size_t* count);

struct SearchWorkerAddRequest {
  int d;
  std::string collection_name;
  std::string vecs;
  std::string ids;
  std::string user_id;
  std::string ks_addr;
  uint16_t ks_port = 0;
};

std::string encode_search_worker_add_request(const SearchWorkerAddRequest& request);

bool decode_search_worker_add_request(const std::string& request_str, SearchWorkerAddRequest* request);

struct SearchWorkerAddResponse {
  bool status;
  std::string msg;
  std::string aux;
};

std::string encode_search_worker_add_response(const SearchWorkerAddResponse& response);

bool decode_search_worker_add_response(const std::string& response_str, SearchWorkerAddResponse* response);

struct SearchWorkerSearchRequest {
  int d;
  int k;
  int nprobe;
  int k_factor;
  uint8_t metric_type;
  std::string collection_name;
  std::string vecs;
  std::string user_id;
  std::string ks_addr;
  uint16_t ks_port = 0;
};

std::string encode_search_worker_search_request(const SearchWorkerSearchRequest& request);

bool decode_search_worker_search_request(const std::string& request_str, SearchWorkerSearchRequest* request);

struct SearchWorkerSearchResponse {
  bool status;
  std::string msg;
  std::string ids;
  std::string aux;
};

std::string encode_search_worker_search_response(const SearchWorkerSearchResponse& response);

bool decode_search_worker_search_response(const std::string& response_str, SearchWorkerSearchResponse* response);


struct SearchWorkerRerankRequest {
  int d;
  int k;
  int nprobe;
  uint8_t metric_type;
  std::string collection_name;
  std::string vecs;
  std::string input_ids;
  std::string user_id;
  std::string ks_addr;
  uint16_t ks_port = 0;
};

std::string encode_search_worker_rerank_request(const SearchWorkerRerankRequest& request);

bool decode_search_worker_rerank_request(const std::string& request_str, SearchWorkerRerankRequest* request);

struct SearchWorkerRerankResponse {
  bool status;
  std::string msg;
  std::string ids;
  std::string scores;
  std::string aux;
};

std::string encode_search_worker_rerank_response(const SearchWorkerRerankResponse& response);

bool decode_search_worker_rerank_response(const std::string& response_str, SearchWorkerRerankResponse* response);

}  // namespace hakes

#endif  // HAKES_MESSAGE_SEARCHSERVICE_H_
