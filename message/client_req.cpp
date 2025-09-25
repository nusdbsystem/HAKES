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

#include "client_req.h"

#include <cstring>
#include <stdexcept>

#include "utils/hexutil.h"
#include "utils/json.h"

namespace hakes_worker {

void encode_hakes_add_request(const HakesAddRequest& request,
                              std::string* data) {
  if (!data) {
    throw std::invalid_argument("data is nullptr");
  }
  // prepare the request data
  json::JSON json_request;
  json_request["n"] = request.n;
  json_request["d"] = request.d;
  json_request["ids"] = request.ids;
  json_request["data"] = request.data;
  data->append(json_request.dump());
}

bool decode_hakes_add_request(const std::string& request_str,
                              HakesAddRequest* request) {
  if (!request) {
    throw std::invalid_argument("request is nullptr");
  }
  auto data = json::JSON::Load(request_str);
  if (data.IsNull()) {
    // printf("data is null\n");
    return false;
  }
  if (!data.hasKey("n") || !data.hasKey("d") || !data.hasKey("data") ||
      !data.hasKey("ids")) {
    // printf("data: %s\n", data.dump().c_str());
    return false;
  }

  request->n = data["n"].ToInt();
  request->d = data["d"].ToInt();
  request->ids = data["ids"].ToString();
  request->data = data["data"].ToString();
  return true;
}

void encode_hakes_add_response(const HakesAddResponse& response,
                               std::string* data) {
  if (!data) {
    throw std::invalid_argument("data is nullptr");
  }
  // prepare the response data
  json::JSON json_response;
  json_response["status"] = response.status;
  json_response["msg"] = response.msg;
  if (!response.status) {
    data->append(json_response.dump());
    return;
  }
  data->append(json_response.dump());
}

bool decode_hakes_add_response(const std::string& response_str,
                               HakesAddResponse* response) {
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

void encode_hakes_search_request(const HakeSearchRequest& request,
                                 std::string* data) {
  if (!data) {
    throw std::invalid_argument("data is nullptr");
  }
  // prepare the request data
  json::JSON json_request;
  json_request["model_type"] = request.model_name;

  json_request["n"] = request.n;
  json_request["d"] = request.d;
  json_request["k"] = request.k;
  json_request["nprobe"] = request.nprobe;
  json_request["k_factor"] = request.k_factor;
  json_request["metric_type"] = request.metric_type;
  json_request["index_version"] = request.index_version;
  json_request["data_type"] = (int)request.data_type;
  json_request["data"] = request.data;
  data->append(json_request.dump());
}

bool decode_hakes_search_request(const std::string& request_str,
                                 HakeSearchRequest* request) {
  if (!request) {
    throw std::invalid_argument("request is nullptr");
  }
  auto data = json::JSON::Load(request_str);
  if (data.IsNull()) {
    // printf("data is null\n");
    return false;
  }
  if (!data.hasKey("n") || !data.hasKey("d") || !data.hasKey("k") ||
      !data.hasKey("nprobe") || !data.hasKey("k_factor") ||
      !data.hasKey("metric_type") || !data.hasKey("index_version") ||
      !data.hasKey("data_type") || !data.hasKey("data")) {
    printf("data: %s\n", data.dump().c_str());
    return false;
  }

  request->n = data["n"].ToInt();
  request->d = data["d"].ToInt();
  request->k = data["k"].ToInt();
  request->nprobe = data["nprobe"].ToInt();
  request->k_factor = data["k_factor"].ToInt();
  request->metric_type = data["metric_type"].ToInt();
  request->index_version = data["index_version"].ToInt();
  request->data_type = (HakesRequestDataType)data["data_type"].ToInt();
  request->data = data["data"].ToString();

  if (data.hasKey("model_type")) {
    request->model_name = data["model_type"].ToString();
  }
  return true;
}

void encode_hakes_search_response(const HakesSearchResponse& response,
                                  std::string* data) {
  if (!data) {
    throw std::invalid_argument("data is nullptr");
  }
  // prepare the response data
  json::JSON json_response;
  json_response["status"] = response.status;
  json_response["msg"] = response.msg;
  json_response["ids"] = response.ids;
  json_response["scores"] = response.scores;
  if (!response.status) {
    data->append(json_response.dump());
    return;
  }
  json_response["data"] = json::Array();
  for (auto& data : response.data) {
    json_response["data"].append(std::move(data));
  }
  data->append(json_response.dump());
}

bool decode_hakes_search_response(const std::string& response_str,
                                  HakesSearchResponse* response) {
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
  if (data.hasKey("ids")) {
    response->ids = data["ids"].ToString();
  }

  if (data.hasKey("scores")) {
    response->scores = data["scores"].ToString();
  }

  if (data.hasKey("data")) {
    response->data.clear();
    for (auto& data : data["data"].ArrayRange()) {
      response->data.push_back(data.ToString());
    }
  }

  return true;
}

}  // namespace hakes_worker
