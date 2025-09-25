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

#include "hakes-worker/workerimpl.h"

#include <iostream>

#include "embed-endpoint/endpoint.h"
#include "message/client_req.h"
#include "message/embed.h"
#include "message/kvservice.h"
#include "message/searchservice.h"

namespace hakes_worker {

bool WorkerImpl::Initialize() { return true; }

bool WorkerImpl::HandleKvOp(uint64_t handle_id,
                            const std::string& sample_request,
                            std::string* output) {
  auto log_record_prefix = "[id-" + std::to_string(handle_id) + "]";

  HakesAddResponse response;
  response.status = false;

  // now only handles add
  HakesAddRequest request;
  auto success = decode_hakes_add_request(sample_request, &request);
  if (!success) {
    std::cout << log_record_prefix << ":failed to decode request\n";
    response.msg = "failed to decode request";
    encode_hakes_add_response(response, output);
    return false;
  }

  if (request.n != 1) {
    std::cout << log_record_prefix
              << ": only support single record per request\n";
    response.msg = "only support single record per request";
    encode_hakes_add_response(response, output);
    return false;
  }

  hakes::SearchWorkerAddRequest add_request;
  add_request.d = request.d;
  add_request.ids = request.ids;

  auto embed_start = std::chrono::high_resolution_clock::now();

  if (request.data_type == HakesRequestDataType::kRaw) {
    if (config_.GetEmbedEndpointAddr().empty()) {
      std::cout << log_record_prefix << ":embed endpoint not set\n";
      response.msg = "embed endpoint not set for raw data";
      encode_hakes_add_response(response, output);
      return false;
    }

    if (config_.GetEmbedEndpointType() == "hakes-embed") {
      // prepare embed request
      hakes::EmbedWorkerRequest embed_request;
      embed_request.model_name_ = request.model_name;
      embed_request.data_ = request.data;

      std::string embed_request_str = embed_request.EncodeTo();

      http_.claim();
      std::string embed_url =
          "http://" + config_.GetEmbedEndpointAddr() + "/run";
      std::string embed_response_str = http_.post(embed_url, embed_request_str);
      http_.release();

      // get the encrypted vector
      hakes::EmbedWorkerResponse embed_response =
          hakes::DecodeEmbedWorkerResponse(embed_response_str);
      if (!embed_response.status) {
        printf("embed response: %s\n", embed_response_str.c_str());
        std::cout << log_record_prefix << ":failed to get encrypted vector\n";
        response.msg = "failed to get encrypted vector";
        encode_hakes_add_response(response, output);
        return false;
      }

      // parse the output to get the vector
      add_request.vecs = embed_response.output;
    } else {
      // external embed service
      auto endpoint = hakes_embedendpoint::CreateEmbedEndpoint(
          config_.GetEmbedEndpointType(), config_.GetEmbedEndpointConfig());
      if (endpoint == nullptr) {
        return 1;
      }

      if (!endpoint->Initialize()) {
        return 1;
      }
      std::string vecs;
      if (!endpoint->HandleEmbedOp(0, request.data, &vecs)) {
        return 1;
      }
      add_request.vecs = vecs;
      endpoint->Close();
    }
  } else {
    add_request.vecs = request.data;
  }

  auto embed_end = std::chrono::high_resolution_clock::now();
  printf("embed time: %ld us\n",
         std::chrono::duration_cast<std::chrono::microseconds>(embed_end -
                                                               embed_start)
             .count());

  // encode the add request
  std::string add_request_str =
      hakes::encode_search_worker_add_request(add_request);

  // send the add request

  auto server_count = config_.ServerCount();

  std::vector<std::string> urls;
  std::vector<std::string> encoded_requests;
  urls.reserve(server_count);
  encoded_requests.reserve(server_count);
  for (int i = 0; i < server_count; i++) {
    urls.push_back("http://" + config_.GetSearchAddressByID(i) + "/add");
    encoded_requests.push_back(add_request_str);
  }

  multi_http_.claim();
  std::vector<std::string> add_resps = multi_http_.post(urls, encoded_requests);
  multi_http_.release();

  for (const auto& resp : add_resps) {
    hakes::SearchWorkerAddResponse add_response;
    if (!hakes::decode_search_worker_add_response(resp, &add_response)) {
      std::cout << log_record_prefix << ":failed to decode response\n";
      response.msg = "failed to decode search service add response";
      encode_hakes_add_response(response, output);
      return false;
    } else if (!add_response.status) {
      std::cout << log_record_prefix << ":failed to add\n";
      response.msg = "failed to add";
      encode_hakes_add_response(response, output);
      return false;
    }
  }

  // add the key value to hakes-store
  size_t count = 0;
  auto ids = hakes::decode_hex_int64s(add_request.ids, &count);
  assert(count == 1);

  auto search_end = std::chrono::high_resolution_clock::now();
  printf("search time: %ld us\n",
         std::chrono::duration_cast<std::chrono::microseconds>(search_end -
                                                               embed_end)
             .count());

  // add to store
  hakes::KVServiceRequest kv_request;
  kv_request.type = "put";
  kv_request.keys = {std::to_string(ids[0])};
  kv_request.values = {request.data};

  http_.claim();
  std::string kv_resp = http_.post("http://" + config_.GetStoreAddr() + "/kv",
                                   hakes::encode_kvservice_request(kv_request));

  // decode kv_resp
  hakes::KVServiceResponse kv_response;
  if (!hakes::decode_kvservice_response(kv_resp, &kv_response)) {
    std::cout << log_record_prefix << ":failed to decode kv response\n";
    response.msg = "failed to decode kv response";
    encode_hakes_add_response(response, output);
    return false;
  } else if (!kv_response.status) {
    std::cout << log_record_prefix << ":failed to add to store\n";
    response.msg = "failed to add to store";
    encode_hakes_add_response(response, output);
    return false;
  }

  response.status = true;
  response.msg = "success";
  encode_hakes_add_response(response, output);
  auto kv_end = std::chrono::high_resolution_clock::now();

  printf(
      "kv time: %ld us\n",
      std::chrono::duration_cast<std::chrono::microseconds>(kv_end - search_end)
          .count());

  return true;
}

bool WorkerImpl::HandleSearchOp(uint64_t handle_id,
                                const std::string& sample_request,
                                std::string* output) {
  auto log_record_prefix = "[id-" + std::to_string(handle_id) + "]";

  printf("sample request: %s\n", sample_request.c_str());

  HakesSearchResponse response;
  response.status = false;

  auto embed_start = std::chrono::high_resolution_clock::now();

  // now only handles add
  HakeSearchRequest request;
  auto success = decode_hakes_search_request(sample_request, &request);
  if (!success) {
    std::cout << log_record_prefix << ":failed to decode request\n";
    fflush(stdout);
    response.msg = "failed to decode request";
    encode_hakes_search_response(response, output);
    return false;
  }

  hakes::SearchWorkerSearchRequest search_request;
  search_request.d = request.d;
  search_request.k = request.k;
  search_request.k_factor = request.k_factor;
  search_request.metric_type = request.metric_type;
  search_request.nprobe = request.nprobe;

  if (request.data_type == HakesRequestDataType::kRaw) {
    if (config_.GetEmbedEndpointAddr().empty()) {
      std::cout << log_record_prefix << ":embed endpoint not set\n";
      response.msg = "embed endpoint not set for raw data";
      encode_hakes_search_response(response, output);
      return false;
    }

    if (config_.GetEmbedEndpointType() == "hakes-embed") {
      // prepare embed request
      hakes::EmbedWorkerRequest embed_request;
      embed_request.model_name_ = request.model_name;
      embed_request.data_ = request.data;

      std::string embed_request_str = embed_request.EncodeTo();
      std::string embed_url =
          "http://" + config_.GetEmbedEndpointAddr() + "/run";
      printf("embed url: %s\n", embed_url.c_str());
      printf("embed request: %s\n", embed_request_str.c_str());
      // fflush(stdout);
      http_.claim();
      std::string embed_response_str = http_.post(embed_url, embed_request_str);
      http_.release();
      // get the encrypted vector
      printf("embed response: %s\n", embed_response_str.c_str());
      hakes::EmbedWorkerResponse embed_response =
          hakes::DecodeEmbedWorkerResponse(embed_response_str);
      if (!embed_response.status) {
        std::cout << log_record_prefix << ":failed to get encrypted vector\n";
        response.msg = "failed to get encrypted vector";
        // fflush(stdout);
        encode_hakes_search_response(response, output);
        return false;
      }

      // parse the output to get the vector
      search_request.vecs = embed_response.output;
    } else {
      // external embed service
      auto endpoint = hakes_embedendpoint::CreateEmbedEndpoint(
          config_.GetEmbedEndpointType(), config_.GetEmbedEndpointAddr());
      if (endpoint == nullptr) {
        return 1;
      }

      if (!endpoint->Initialize()) {
        return 1;
      }
      std::string vecs;
      if (!endpoint->HandleEmbedOp(0, request.data, &vecs)) {
        return 1;
      }
      search_request.vecs = vecs;
      endpoint->Close();
    }
  } else {
    search_request.vecs = request.data;
  }

  auto embed_end = std::chrono::high_resolution_clock::now();
  printf("embed time: %ld us\n",
         std::chrono::duration_cast<std::chrono::microseconds>(embed_end -
                                                               embed_start)
             .count());

  // encode the search request
  std::string search_request_str =
      hakes::encode_search_worker_search_request(search_request);

  // send the search request
  auto preferred_server = config_.GetPreferredSearchAddress();
  std::string search_url = "http://" + preferred_server + "/search";
  printf("search url: %s\n", search_url.c_str());
  printf("search request: %s\n", search_request_str.c_str());
  // fflush(stdout);

  http_.claim();
  std::string search_response_str = http_.post(search_url, search_request_str);
  http_.release();

  hakes::SearchWorkerSearchResponse search_response;
  if (!hakes::decode_search_worker_search_response(search_response_str,
                                                   &search_response)) {
    std::cout << log_record_prefix << ":failed to decode response\n";
    response.msg = "failed to decode search service search response";
    encode_hakes_search_response(response, output);
    fflush(stdout);
    return false;
  } else if (!search_response.status) {
    std::cout << log_record_prefix << ":failed to search\n";
    response.msg = "failed to search";
    encode_hakes_search_response(response, output);
    fflush(stdout);
    return false;
  }

  auto base_end = std::chrono::high_resolution_clock::now();
  printf("base time: %ld us\n",
         std::chrono::duration_cast<std::chrono::microseconds>(base_end -
                                                               embed_end)
             .count());

  // send rerank request
  hakes::SearchWorkerRerankRequest rerank_request;
  rerank_request.d = request.d;
  rerank_request.k = request.k;
  rerank_request.metric_type = request.metric_type;
  rerank_request.vecs = search_request.vecs;
  rerank_request.input_ids = search_response.ids;
  auto encoded_rerank_request =
      hakes::encode_search_worker_rerank_request(rerank_request);

  std::vector<std::string> urls;
  std::vector<std::string> encoded_requests;
  urls.reserve(config_.ServerCount());
  encoded_requests.reserve(config_.ServerCount());
  for (int i = 0; i < config_.ServerCount(); i++) {
    urls.push_back("http://" + config_.GetSearchAddressByID(i) + "/rerank");
    encoded_requests.push_back(encoded_rerank_request);
  }

  printf("rerank urls: %ld first: %s\n", urls.size(), urls[0].c_str());
  printf("rerank request: %s\n", encoded_rerank_request.c_str());
  // fflush(stdout);

  multi_http_.claim();
  auto rerank_resps = multi_http_.post(urls, encoded_requests);
  multi_http_.release();

  std::vector<hakes::SearchWorkerRerankResponse> partial_results;
  partial_results.reserve(rerank_resps.size());

  for (const auto& rerank_resp : rerank_resps) {
    printf("rerank response: %s\n", rerank_resp.c_str());

    hakes::SearchWorkerRerankResponse rerank_response;
    if (!hakes::decode_search_worker_rerank_response(rerank_resp,
                                                     &rerank_response)) {
      std::cout << log_record_prefix << ":failed to decode rerank response\n";
      response.msg = "failed to decode rerank response";
      encode_hakes_search_response(response, output);
      fflush(stdout);
      return false;
    } else if (!rerank_response.status) {
      std::cout << log_record_prefix << ":failed to rerank\n";
      response.msg = "failed to rerank";
      encode_hakes_search_response(response, output);
      fflush(stdout);
      return false;
    }
    partial_results.push_back(std::move(rerank_response));
  }

  auto rerank_end = std::chrono::high_resolution_clock::now();
  printf("rerank time: %ld us\n",
         std::chrono::duration_cast<std::chrono::microseconds>(rerank_end -
                                                               base_end)
             .count());
  // fflush(stdout);

  // merge the rerank results
  auto merged_result = dm_->MergeSearchResults(partial_results, request.k);
  if (!merged_result.status) {
    std::cout << log_record_prefix << ":failed to merge rerank results\n";
    response.msg = "failed to merge rerank results";
    encode_hakes_search_response(response, output);
    fflush(stdout);
    return false;
  }

  response.ids = merged_result.ids;
  response.scores = merged_result.scores;

  auto merge_end = std::chrono::high_resolution_clock::now();
  printf("merge time: %ld us\n",
         std::chrono::duration_cast<std::chrono::microseconds>(merge_end -
                                                               rerank_end)
             .count());

  // if no store service then just return
  if (config_.GetStoreAddr().empty()) {
    response.status = true;
    response.msg = "success";
    encode_hakes_search_response(response, output);
    fflush(stdout);
    return true;
  }

  // retrieve data from store
  size_t count = 0;
  auto ids = hakes::decode_hex_int64s(merged_result.ids, &count);
  for (size_t i = 0; i < count; i++) {
    printf("id: %ld\n", ids[i]);
  }

  hakes::KVServiceRequest kv_request;
  kv_request.type = "get";
  kv_request.keys.reserve(count);
  for (size_t i = 0; i < count; i++) {
    if (ids[i] != -1) {
      kv_request.keys.emplace_back(std::to_string(ids[i]));
    }
  }
  if (kv_request.keys.empty()) {
    response.status = true;
    response.msg = "all deleted";
    encode_hakes_search_response(response, output);
    fflush(stdout);
    return true;
  }

  auto kv_service_url = "http://" + config_.GetStoreAddr() + "/kv";
  auto kv_request_str = hakes::encode_kvservice_request(kv_request);
  printf("kv url: %s\n", kv_service_url.c_str());
  printf("kv request: %s\n", kv_request_str.c_str());
  // fflush(stdout);

  http_.claim();
  std::string kv_resp =
      http_.post(std::move(kv_service_url), std::move(kv_request_str));

  // decode kv_resp
  hakes::KVServiceResponse kv_response;
  if (!hakes::decode_kvservice_response(kv_resp, &kv_response)) {
    std::cout << log_record_prefix << ":failed to decode kv response\n";
    response.msg = "failed to decode kv response";
    encode_hakes_search_response(response, output);
    fflush(stdout);
    return false;
  } else if (!kv_response.status) {
    std::cout << log_record_prefix << ":failed to get from store\n";
    response.msg = "failed to get from store";
    encode_hakes_search_response(response, output);
    printf("kv response: %s\n", kv_resp.c_str());
    fflush(stdout);
    return false;
  }

  response.status = true;
  response.msg = "success";
  response.data = kv_response.values;
  encode_hakes_search_response(response, output);

  printf("last response: %s\n", output->c_str());
  auto kv_end = std::chrono::high_resolution_clock::now();
  printf(
      "kv time: %ld us\n",
      std::chrono::duration_cast<std::chrono::microseconds>(kv_end - merge_end)
          .count());

  // fflush(stdout);
  return true;
}

void WorkerImpl::Close() {}

}  // namespace hakes_worker
