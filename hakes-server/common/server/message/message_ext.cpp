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

#include "message_ext.h"

#include <stdexcept>

namespace hakes {

void encode_extended_add_request(const ExtendedAddRequest& request,
                                 std::string* data) {
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
  json_request["index_version"] = request.index_version;
  json_request["add_to_refine_only"] = request.add_to_refine_only;
  json_request["assigned"] = request.assigned;
  if (!request.assigned) {
    data->append(json_request.dump());
    return;
  }
  if (request.assign) {
    json_request["assign"] =
        hex_encode(reinterpret_cast<const char*>(request.assign),
                   request.n * sizeof(int64_t));
    data->append(json_request.dump());
  }
}

bool decode_extended_add_request(const std::string& request_str,
                                 ExtendedAddRequest* request) {
  if (!request) {
    throw std::invalid_argument("request is nullptr");
  }
  auto data = json::JSON::Load(request_str);
  if (data.IsNull()) {
    return false;
  }
  if (!data.hasKey("n") || !data.hasKey("d") || !data.hasKey("vecs") ||
      !data.hasKey("ids") || !data.hasKey("add_to_refine_only") ||
      !data.hasKey("assigned")) {
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

  // decode index version
  if (data.hasKey("index_version")) {
    request->index_version = data["index_version"].ToInt();
  } else {
    request->index_version = -1;
  }
  // decode add to refine
  request->add_to_refine_only = data["add_to_refine_only"].ToBool();

  // decode assigned
  request->assigned = data["assigned"].ToBool();
  if (!request->assigned) {
    return true;
  }

  // decode assign
  if (!data.hasKey("assign")) {
    return false;
  }
  auto assign_hex = data["assign"].ToString();
  auto assign_byte_str = hex_decode(assign_hex.c_str(), assign_hex.size());
  request->assign_holder.reset(new int64_t[request->n]);
  std::memcpy(request->assign_holder.get(), assign_byte_str.data(),
              assign_byte_str.size());
  request->assign = request->assign_holder.get();
  return true;
}

void encode_extended_add_response(const ExtendedAddResponse& response,
                                  std::string* data) {
  if (!data) {
    throw std::invalid_argument("data is nullptr");
  }
  json::JSON json_response;
  json_response["status"] = response.status;
  json_response["msg"] = response.msg;
  json_response["n"] = response.n;
  if (response.assign) {
    json_response["assign"] =
        hex_encode(reinterpret_cast<const char*>(response.assign),
                   response.n * sizeof(int64_t));
  }
  if (response.vecs_t_d > 0) {
    json_response["vecs_t_d"] = response.vecs_t_d;
    json_response["vecs_t"] =
        hex_encode(reinterpret_cast<const char*>(response.vecs_t),
                   response.n * sizeof(float) * response.vecs_t_d);
  }
  json_response["index_version"] = response.index_version;
  data->append(json_response.dump());
}

bool decode_extended_add_response(const std::string& response_str,
                                  ExtendedAddResponse* response) {
  if (!response) {
    throw std::invalid_argument("response is nullptr");
  }
  auto data = json::JSON::Load(response_str);
  if (data.IsNull()) {
    return false;
  }
  if (!data.hasKey("status") || !data.hasKey("msg") || !data.hasKey("n")) {
    return false;
  }

  response->status = data["status"].ToBool();
  response->msg = data["msg"].ToString();
  response->n = data["n"].ToInt();

  // decode assign
  if (data.hasKey("assign")) {
    auto assign_hex = data["assign"].ToString();
    auto assign_byte_str = hex_decode(assign_hex.c_str(), assign_hex.size());
    response->assign_holder.reset(new int64_t[response->n]);
    std::memcpy(response->assign_holder.get(), assign_byte_str.data(),
                assign_byte_str.size());
    response->assign = response->assign_holder.get();
  }
  if (data.hasKey("vecs_t_d") && data.hasKey("vecs_t")) {
    response->vecs_t_d = data["vecs_t_d"].ToInt();
    auto vecs_t_hex = data["vecs_t"].ToString();
    auto vecs_t_byte_str = hex_decode(vecs_t_hex.c_str(), vecs_t_hex.size());
    response->vecs_t_holder.reset(new float[response->n * response->vecs_t_d]);
    std::memcpy(response->vecs_t_holder.get(), vecs_t_byte_str.data(),
                vecs_t_byte_str.size());
    response->vecs_t = response->vecs_t_holder.get();
  }

  if (data.hasKey("index_version")) {
    response->index_version = data["index_version"].ToInt();
  } else {
    response->index_version = -1;
  }

  return true;
}

void encode_rerank_request(const RerankRequest& request, std::string* data) {
  if (!data) {
    throw std::invalid_argument("data is nullptr");
  }
  json::JSON json_request;
  json_request["n"] = request.n;
  json_request["d"] = request.d;
  json_request["vecs"] = hex_encode(reinterpret_cast<const char*>(request.vecs),
                                    request.n * sizeof(float) * request.d);
  json_request["k"] = request.k;
  json_request["metric_type"] = static_cast<int>(request.metric_type);
  json_request["k_base_count"] =
      hex_encode(reinterpret_cast<const char*>(request.k_base_count),
                 request.n * sizeof(int64_t));
  int64_t total_base = 0;
  for (int i = 0; i < request.n; i++) {
    total_base += request.k_base_count[i];
  }
  json_request["base_labels"] =
      hex_encode(reinterpret_cast<const char*>(request.base_labels),
                 total_base * sizeof(int64_t));
  json_request["base_distances"] =
      hex_encode(reinterpret_cast<const char*>(request.base_distances),
                 total_base * sizeof(float));
  data->append(json_request.dump());
}

bool decode_rerank_request(const std::string& request_str,
                           RerankRequest* request) {
  if (!request) {
    throw std::invalid_argument("request is nullptr");
  }
  auto data = json::JSON::Load(request_str);
  if (data.IsNull()) {
    return false;
  }
  if (!data.hasKey("n") || !data.hasKey("d") || !data.hasKey("vecs") ||
      !data.hasKey("k") || !data.hasKey("k_base_count") ||
      !data.hasKey("base_labels") || !data.hasKey("base_distances") ||
      !data.hasKey("metric_type")) {
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

  request->k = data["k"].ToInt();
  request->metric_type = data["metric_type"].ToInt();

  // decode k_base_count
  auto k_base_count_hex = data["k_base_count"].ToString();
  auto k_base_count_byte_str =
      hex_decode(k_base_count_hex.c_str(), k_base_count_hex.size());
  request->k_base_count_holder.reset(new int64_t[request->n]);
  std::memcpy(request->k_base_count_holder.get(), k_base_count_byte_str.data(),
              k_base_count_byte_str.size());
  request->k_base_count = request->k_base_count_holder.get();

  // decode base_labels
  auto base_labels_hex = data["base_labels"].ToString();
  auto base_labels_byte_str =
      hex_decode(base_labels_hex.c_str(), base_labels_hex.size());
  request->base_labels_holder.reset(
      new int64_t[base_labels_byte_str.size() / sizeof(int64_t)]);
  std::memcpy(request->base_labels_holder.get(), base_labels_byte_str.data(),
              base_labels_byte_str.size());
  request->base_labels = request->base_labels_holder.get();

  // decode base_distances
  auto base_distances_hex = data["base_distances"].ToString();
  auto base_distances_byte_str =
      hex_decode(base_distances_hex.c_str(), base_distances_hex.size());
  request->base_distances_holder.reset(
      new float[base_distances_byte_str.size() / sizeof(float)]);
  std::memcpy(request->base_distances_holder.get(),
              base_distances_byte_str.data(), base_distances_byte_str.size());
  request->base_distances = request->base_distances_holder.get();

  return true;
}

void encode_get_index_response(const GetIndexResponse& response,
                               std::string* data) {
  if (!data) {
    throw std::invalid_argument("data is nullptr");
  }
  json::JSON json_response;
  json_response["status"] = response.status;
  json_response["msg"] = response.msg;
  json_response["index_version"] = response.index_version;
  json_response["params"] =
      hex_encode(reinterpret_cast<const char*>(response.params.data()),
                 response.params.size());
  data->append(json_response.dump());
}

bool decode_get_index_response(const std::string& response_str,
                               GetIndexResponse* response) {
  if (!response) {
    throw std::invalid_argument("response is nullptr");
  }
  auto data = json::JSON::Load(response_str);
  if (data.IsNull()) {
    return false;
  }
  if (!data.hasKey("status") || !data.hasKey("msg") ||
      !data.hasKey("index_version") || !data.hasKey("params")) {
    return false;
  }

  response->index_version = data["index_version"].ToInt();
  auto params_hex = data["params"].ToString();
  response->params = hex_decode(params_hex.c_str(), params_hex.size());
  return true;
}

void encode_update_index_request(const UpdateIndexRequest& request,
                                 std::string* data) {
  if (!data) {
    throw std::invalid_argument("data is nullptr");
  }
  json::JSON json_request;
  json_request["params"] =
      hex_encode(reinterpret_cast<const char*>(request.params.data()),
                 request.params.size());
  data->append(json_request.dump());
}

bool decode_update_index_request(const std::string& request_str,
                                 UpdateIndexRequest* request) {
  if (!request) {
    throw std::invalid_argument("request is nullptr");
  }
  auto data = json::JSON::Load(request_str);
  if (data.IsNull()) {
    return false;
  }
  if (!data.hasKey("params")) {
    return false;
  }

  auto params_hex = data["params"].ToString();
  request->params = hex_decode(params_hex.c_str(), params_hex.size());
  return true;
}

void encode_update_index_response(const UpdateIndexResponse& response,
                                  std::string* data) {
  if (!data) {
    throw std::invalid_argument("data is nullptr");
  }
  json::JSON json_response;
  json_response["status"] = response.status;
  json_response["msg"] = response.msg;
  json_response["index_version"] = response.index_version;
  data->append(json_response.dump());
}

bool decode_update_index_response(const std::string& response_str,
                                  UpdateIndexResponse* response) {
  if (!response) {
    throw std::invalid_argument("response is nullptr");
  }
  auto data = json::JSON::Load(response_str);
  if (data.IsNull()) {
    return false;
  }
  if (!data.hasKey("status") || !data.hasKey("msg") ||
      !data.hasKey("index_version")) {
    return false;
  }

  response->status = data["status"].ToBool();
  response->msg = data["msg"].ToString();
  response->index_version = data["index_version"].ToInt();
  return true;
}

}  // namespace hakes
