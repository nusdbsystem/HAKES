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

#include "searchservice.h"

#include <cstring>

#include "utils/hexutil.h"
#include "utils/json.h"

namespace hakes {

std::string encode_hex_floats(const float* vecs, size_t count) {
  return hex_encode(reinterpret_cast<const char*>(vecs), count * sizeof(float));
}

std::unique_ptr<float[]> decode_hex_floats(const std::string& vecs_str,
                                           size_t* count) {
  auto vecs_byte_str = hex_decode(vecs_str.c_str(), vecs_str.size());
  *count = vecs_byte_str.size() / sizeof(float);
  auto vecs = std::unique_ptr<float[]>(new float[*count]);
  std::memcpy(vecs.get(), vecs_byte_str.data(), vecs_byte_str.size());
  return vecs;
}

std::string encode_hex_int64s(const int64_t* vecs, size_t count) {
  return hex_encode(reinterpret_cast<const char*>(vecs),
                    count * sizeof(int64_t));
}

std::unique_ptr<int64_t[]> decode_hex_int64s(const std::string& vecs_str,
                                             size_t* count) {
  auto vecs_byte_str = hex_decode(vecs_str.c_str(), vecs_str.size());
  *count = vecs_byte_str.size() / sizeof(int64_t);
  auto vecs = std::unique_ptr<int64_t[]>(new int64_t[*count]);
  std::memcpy(vecs.get(), vecs_byte_str.data(), vecs_byte_str.size());
  return vecs;
}

std::string encode_search_worker_load_request(
    const SearchWorkerLoadRequest& request) {
  json::JSON ret;
  ret["collection_name"] = request.collection_name;
  ret["mode"] = request.mode;
  return ret.dump();
}

bool decode_search_worker_load_request(const std::string& request_str,
                                       SearchWorkerLoadRequest* request) {
  auto input = json::JSON::Load(request_str);
  request->collection_name = input["collection_name"].ToString();
  request->mode = input.hasKey("mode") ? input["mode"].ToInt() : 0;
  return true;
}

std::string encode_search_worker_load_response(
    const SearchWorkerLoadResponse& response) {
  json::JSON ret;
  ret["status"] = response.status;
  ret["msg"] = response.msg;
  return ret.dump();
}

bool decode_search_worker_load_response(const std::string& response_str,
                                        SearchWorkerLoadResponse* response) {
  auto input = json::JSON::Load(response_str);
  response->status = input["status"].ToBool();
  response->msg = input["msg"].ToString();
  return true;
}

std::string encode_search_worker_add_request(
    const SearchWorkerAddRequest& request) {
  json::JSON ret;
  ret["d"] = request.d;
  ret["collection_name"] = request.collection_name;
  ret["vecs"] = request.vecs;
  ret["ids"] = request.ids;
  return ret.dump();
}

bool decode_search_worker_add_request(const std::string& request_str,
                                      SearchWorkerAddRequest* request) {
  auto input = json::JSON::Load(request_str);
  request->d = input["d"].ToInt();
  request->collection_name = input["collection_name"].ToString();
  request->vecs = input["vecs"].ToString();
  request->ids = input["ids"].ToString();
  return true;
}

std::string encode_search_worker_add_response(
    const SearchWorkerAddResponse& response) {
  json::JSON ret;
  ret["status"] = response.status;
  ret["msg"] = response.msg;
  return ret.dump();
}

bool decode_search_worker_add_response(const std::string& response_str,
                                       SearchWorkerAddResponse* response) {
  auto input = json::JSON::Load(response_str);
  response->status = input["status"].ToBool();
  response->msg = input["msg"].ToString();
  return true;
}

std::string encode_search_worker_search_request(
    const SearchWorkerSearchRequest& request) {
  json::JSON ret;
  ret["d"] = request.d;
  ret["k"] = request.k;
  ret["nprobe"] = request.nprobe;
  ret["k_factor"] = request.k_factor;
  ret["metric_type"] = request.metric_type;
  ret["collection_name"] = request.collection_name;
  ret["vecs"] = request.vecs;
  return ret.dump();
}

bool decode_search_worker_search_request(const std::string& request_str,
                                         SearchWorkerSearchRequest* request) {
  auto input = json::JSON::Load(request_str);
  request->d = input["d"].ToInt();
  request->k = input["k"].ToInt();
  request->nprobe = input["nprobe"].ToInt();
  request->k_factor = input["k_factor"].ToInt();
  request->metric_type = input["metric_type"].ToInt();
  request->collection_name = input["collection_name"].ToString();
  request->vecs = input["vecs"].ToString();
  return true;
}

std::string encode_search_worker_search_response(
    const SearchWorkerSearchResponse& response) {
  json::JSON ret;
  ret["status"] = response.status;
  ret["msg"] = response.msg;
  ret["ids"] = response.ids;
  ret["scores"] = response.scores;
  return ret.dump();
}

bool decode_search_worker_search_response(
    const std::string& response_str, SearchWorkerSearchResponse* response) {
  auto input = json::JSON::Load(response_str);
  response->status = input["status"].ToBool();
  response->msg = input["msg"].ToString();
  response->ids = input["ids"].ToString();
  response->scores = input["scores"].ToString();
  return true;
}

std::string encode_search_worker_rerank_request(
    const SearchWorkerRerankRequest& request) {
  json::JSON ret;
  ret["d"] = request.d;
  ret["k"] = request.k;
  ret["metric_type"] = request.metric_type;
  ret["collection_name"] = request.collection_name;
  ret["vecs"] = request.vecs;
  ret["input_ids"] = request.input_ids;
  return ret.dump();
}

bool decode_search_worker_rerank_request(const std::string& request_str,
                                         SearchWorkerRerankRequest* request) {
  auto input = json::JSON::Load(request_str);
  request->d = input["d"].ToInt();
  request->k = input["k"].ToInt();
  request->metric_type = input["metric_type"].ToInt();
  request->collection_name = input["collection_name"].ToString();
  request->vecs = input["vecs"].ToString();
  request->input_ids = input["input_ids"].ToString();
  return true;
}

std::string encode_search_worker_rerank_response(
    const SearchWorkerRerankResponse& response) {
  json::JSON ret;
  ret["status"] = response.status;
  ret["msg"] = response.msg;
  ret["ids"] = response.ids;
  ret["scores"] = response.scores;
  return ret.dump();
}

bool decode_search_worker_rerank_response(
    const std::string& response_str, SearchWorkerRerankResponse* response) {
  auto input = json::JSON::Load(response_str);
  response->status = input["status"].ToBool();
  response->msg = input["msg"].ToString();
  response->ids = input["ids"].ToString();
  response->scores = input["scores"].ToString();
  return true;
}

std::string encode_search_worker_delete_request(
    const SearchWorkerDeleteRequest& request) {
  json::JSON ret;
  ret["collection_name"] = request.collection_name;
  ret["ids"] = request.ids;
  return ret.dump();
}

bool decode_search_worker_delete_request(const std::string& request_str,
                                         SearchWorkerDeleteRequest* request) {
  auto input = json::JSON::Load(request_str);
  request->collection_name = input["collection_name"].ToString();
  request->ids = input["ids"].ToString();
  return true;
}

std::string encode_search_worker_delete_response(
    const SearchWorkerDeleteResponse& response) {
  json::JSON ret;
  ret["status"] = response.status;
  ret["msg"] = response.msg;
  return ret.dump();
}

bool decode_search_worker_delete_response(
    const std::string& response_str, SearchWorkerDeleteResponse* response) {
  auto input = json::JSON::Load(response_str);
  response->status = input["status"].ToBool();
  response->msg = input["msg"].ToString();
  return true;
}

std::string encode_search_worker_checkpoint_request(
    const SearchWorkerCheckpointRequest& request) {
  json::JSON ret;
  ret["collection_name"] = request.collection_name;
  return ret.dump();
}

bool decode_search_worker_checkpoint_request(
    const std::string& request_str, SearchWorkerCheckpointRequest* request) {
  auto input = json::JSON::Load(request_str);
  request->collection_name = input["collection_name"].ToString();
  return true;
}

std::string encode_search_worker_checkpoint_response(
    const SearchWorkerCheckpointResponse& response) {
  json::JSON ret;
  ret["status"] = response.status;
  ret["msg"] = response.msg;
  return ret.dump();
}

bool decode_search_worker_checkpoint_response(
    const std::string& response_str, SearchWorkerCheckpointResponse* response) {
  auto input = json::JSON::Load(response_str);
  response->status = input["status"].ToBool();
  response->msg = input["msg"].ToString();
  return true;
}

}  // namespace hakes
