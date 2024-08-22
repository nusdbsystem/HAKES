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

#include "message/message.h"

namespace hakes {

void encode_add_request(const AddRequest& request, std::string* data) {
  if (!data) {
    throw std::invalid_argument("data is nullptr");
  }
  // prepare the request data
  json::JSON json_request;
  json_request["n"] = request.n;
  json_request["d"] = request.d;
  json_request["ids"] = hex_encode(reinterpret_cast<const char*>(request.ids),
                                   request.n * sizeof(int64_t));
  json_request["vecs"] = hex_encode(reinterpret_cast<const char*>(request.vecs),
                                    request.n * sizeof(float) * request.d);
  data->append(json_request.dump());
}

bool decode_add_request(const std::string& request_str, AddRequest* request) {
  if (!request) {
    throw std::invalid_argument("request is nullptr");
  }
  auto data = json::JSON::Load(request_str);
  if (data.IsNull()) {
    return false;
  }
  if (!data.hasKey("n") || !data.hasKey("d") || !data.hasKey("vecs") ||
      !data.hasKey("ids")) {
    return false;
  }

  request->n = data["n"].ToInt();
  request->d = data["d"].ToInt();
  // decode ids
  auto ids_hex = data["ids"].ToString();
  auto ids_byte_str = hex_decode(ids_hex.c_str(), ids_hex.size());
  request->ids_holder.reset(new int64_t[request->n]);
  std::memcpy(request->ids_holder.get(), ids_byte_str.data(),
              ids_byte_str.size());
  request->ids = request->ids_holder.get();

  // decode vecs
  auto vecs_hex = data["vecs"].ToString();
  auto vecs_byte_str = hex_decode(vecs_hex.c_str(), vecs_hex.size());
  request->vecs_holder.reset(new float[request->n * request->d]);
  std::memcpy(request->vecs_holder.get(), vecs_byte_str.data(),
              vecs_byte_str.size());
  request->vecs = request->vecs_holder.get();

  return true;
}

void encode_add_response(const AddResponse& response, std::string* data) {
  if (!data) {
    throw std::invalid_argument("data is nullptr");
  }
  json::JSON json_response;
  json_response["status"] = response.status;
  json_response["msg"] = response.msg;
  data->append(json_response.dump());
}

bool decode_add_response(const std::string& response_str,
                         AddResponse* response) {
  if (!response) {
    throw std::invalid_argument("response is nullptr");
  }
  auto data = json::JSON::Load(response_str);
  if (data.IsNull()) {
    return false;
  }

  if (!data.hasKey("status") || !data.hasKey("msg")) {
    return false;
  }
  response->status = data["status"].ToBool();
  response->msg = data["msg"].ToString();
  return true;
}

void encode_search_request(const SearchRequest& request, std::string* data) {
  if (!data) {
    throw std::invalid_argument("data is nullptr");
  }
  json::JSON json_request;
  json_request["n"] = request.n;
  json_request["d"] = request.d;
  json_request["vecs"] = hex_encode(reinterpret_cast<const char*>(request.vecs),
                                    request.n * sizeof(float) * request.d);
  json_request["nprobe"] = request.nprobe;
  json_request["k"] = request.k;
  json_request["k_factor"] = request.k_factor;
  json_request["metric_type"] = static_cast<int>(request.metric_type);
  json_request["require_pa"] = request.require_pa;
  data->append(json_request.dump());
}

bool decode_search_request(const std::string& request_str,
                           SearchRequest* request) {
  if (!request) {
    throw std::invalid_argument("request is nullptr");
  }
  auto data = json::JSON::Load(request_str);
  if (data.IsNull()) {
    return false;
  }
  if (!data.hasKey("n") || !data.hasKey("d") || !data.hasKey("vecs") ||
      !data.hasKey("nprobe") || !data.hasKey("k") || !data.hasKey("k_factor") ||
      !data.hasKey("metric_type") || !data.hasKey("require_pa")) {
    return false;
  }
  request->n = data["n"].ToInt();
  request->d = data["d"].ToInt();
  // decode vecs
  auto vecs_hex = data["vecs"].ToString();
  auto vecs_byte_str = hex_decode(vecs_hex.c_str(), vecs_hex.size());
  request->vecs_holder.reset(new float[request->n * request->d]);
  std::memcpy(request->vecs_holder.get(), vecs_byte_str.data(),
              vecs_byte_str.size());
  request->vecs = request->vecs_holder.get();
  request->nprobe = data["nprobe"].ToInt();
  request->k = data["k"].ToInt();
  request->k_factor = data["k_factor"].ToInt();
  request->metric_type =
      static_cast<SearchMetricType>(data["metric_type"].ToInt());
  request->require_pa = data["require_pa"].ToBool();
  return true;
}

void encode_search_response(const SearchResponse& response, std::string* data) {
  if (!data) {
    throw std::invalid_argument("data is nullptr");
  }
  json::JSON json_response;
  json_response["status"] = response.status;
  json_response["msg"] = response.msg;
  if (!response.status) {
    // early return for error response
    data->append(json_response.dump());
    return;
  }

  json_response["n"] = response.n;
  json_response["k"] = response.k;
  json_response["scores"] =
      hex_encode(reinterpret_cast<const char*>(response.scores),
                 response.n * sizeof(float) * response.k);
  json_response["ids"] = hex_encode(reinterpret_cast<const char*>(response.ids),
                                    response.n * sizeof(int64_t) * response.k);
  json_response["require_pa"] = response.require_pa;
  if (response.require_pa) {
    json_response["pas"] =
        hex_encode(reinterpret_cast<const char*>(response.pas),
                   response.n * sizeof(int64_t) * response.k);
  }
  json_response["index_version"] = response.index_version;
  data->append(json_response.dump());
}

bool decode_search_response(const std::string& response_str,
                            SearchResponse* response) {
  if (!response) {
    throw std::invalid_argument("response is nullptr");
  }
  auto data = json::JSON::Load(response_str);
  if (data.IsNull()) {
    return false;
  }
  if (!data.hasKey("status") || !data.hasKey("msg")) {
    return false;
  }
  response->status = data["status"].ToBool();
  response->msg = data["msg"].ToString();
  if (!response->status) {
    // early return for error response
    return true;
  }

  if (!data.hasKey("n") || !data.hasKey("k") || !data.hasKey("scores") ||
      !data.hasKey("ids") || !data.hasKey("require_pa")) {
    return false;
  }
  response->n = data["n"].ToInt();
  response->k = data["k"].ToInt();
  // decode scores
  auto scores_hex = data["scores"].ToString();
  auto scores_byte_str = hex_decode(scores_hex.c_str(), scores_hex.size());
  response->scores_holder.reset(new float[response->n * response->k]);
  std::memcpy(response->scores_holder.get(), scores_byte_str.data(),
              scores_byte_str.size());
  response->scores = response->scores_holder.get();
  // decode ids
  auto ids_hex = data["ids"].ToString();
  auto ids_byte_str = hex_decode(ids_hex.c_str(), ids_hex.size());
  response->ids_holder.reset(new int64_t[response->n * response->k]);
  std::memcpy(response->ids_holder.get(), ids_byte_str.data(),
              ids_byte_str.size());
  response->ids = response->ids_holder.get();
  // decode pas
  response->require_pa = data["require_pa"].ToBool();
  if (response->require_pa) {
    if (!data.hasKey("pas")) {
      return false;
    }
    auto pas_hex = data["pas"].ToString();
    auto pas_byte_str = hex_decode(pas_hex.c_str(), pas_hex.size());
    response->pas_holder.reset(new int64_t[response->n * response->k]);
    std::memcpy(response->pas_holder.get(), pas_byte_str.data(),
                pas_byte_str.size());
    response->pas = response->pas_holder.get();
  }
  if (data.hasKey("index_version")) {
    response->index_version = data["index_version"].ToInt();
  } else {
    response->index_version = -1;
  }
  return true;
}

}  // namespace hakes
