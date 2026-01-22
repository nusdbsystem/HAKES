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

#include <openssl/hmac.h>
#include <openssl/sha.h>

#include <algorithm>
#include <iostream>
#include <sstream>

#include "embed-endpoint/endpoint.h"
#include "server/message/client_req.h"
#include "server/message/embed.h"
#include "server/message/kvservice.h"
#include "server/message/searchservice.h"
#include "utils/json.h"

namespace hakes_worker {

// JWT secret key - in production, this should be configurable and secure
const std::string JWT_SECRET = "hakes-secret-key-change-in-production";

std::string base64url_encode(const std::string& input) {
  std::string b64 = hakes::base64_encode(
      reinterpret_cast<const uint8_t*>(input.data()), input.size());
  // Replace '+' with '-', '/' with '_', and remove '=' padding
  std::replace(b64.begin(), b64.end(), '+', '-');
  std::replace(b64.begin(), b64.end(), '/', '_');
  // Remove padding
  size_t pos = b64.find('=');
  if (pos != std::string::npos) {
    b64 = b64.substr(0, pos);
  }
  return b64;
}

std::string base64url_decode(const std::string& input) {
  std::string b64 = input;
  // Replace '-' with '+', '_' with '/'
  std::replace(b64.begin(), b64.end(), '-', '+');
  std::replace(b64.begin(), b64.end(), '_', '/');
  // Add padding if needed
  while (b64.size() % 4 != 0) {
    b64 += '=';
  }
  return hakes::base64_decode(reinterpret_cast<const uint8_t*>(b64.data()),
                              b64.size());
}

std::string create_jwt_token(const std::string& user_id,
                             const std::string& roles, long long exp) {
  // Create header: {"alg":"HS256","typ":"JWT"}
  json::JSON header = json::Object();
  header["alg"] = "HS256";
  header["typ"] = "JWT";
  std::string header_b64 = base64url_encode(header.dump());

  // Create payload: {"sub":"user_id","roles":"roles","exp":exp}
  json::JSON payload = json::Object();
  payload["sub"] = user_id;
  payload["roles"] = roles;
  payload["exp"] = exp;
  std::string payload_b64 = base64url_encode(payload.dump());

  // Create signature
  std::string data = header_b64 + "." + payload_b64;
  unsigned char* hmac_result =
      HMAC(EVP_sha256(), JWT_SECRET.c_str(), JWT_SECRET.size(),
           reinterpret_cast<const unsigned char*>(data.c_str()), data.size(),
           nullptr, nullptr);
  std::string signature =
      base64url_encode(std::string(reinterpret_cast<char*>(hmac_result), 32));

  return data + "." + signature;
}

bool validate_jwt_token(const std::string& token, std::string* user_id,
                        std::string* roles) {
  // Split token into parts
  size_t dot1 = token.find('.');
  size_t dot2 = token.find('.', dot1 + 1);
  if (dot1 == std::string::npos || dot2 == std::string::npos) {
    return false;
  }

  std::string header_b64 = token.substr(0, dot1);
  std::string payload_b64 = token.substr(dot1 + 1, dot2 - dot1 - 1);
  std::string signature_b64 = token.substr(dot2 + 1);

  // Verify signature
  std::string data = header_b64 + "." + payload_b64;
  unsigned char* hmac_result =
      HMAC(EVP_sha256(), JWT_SECRET.c_str(), JWT_SECRET.size(),
           reinterpret_cast<const unsigned char*>(data.c_str()), data.size(),
           nullptr, nullptr);
  std::string expected_signature =
      base64url_encode(std::string(reinterpret_cast<char*>(hmac_result), 32));

  if (signature_b64 != expected_signature) {
    return false;
  }

  // Decode payload
  std::string payload_json = base64url_decode(payload_b64);
  try {
    json::JSON payload = json::JSON::Load(payload_json);

    // Check expiration
    if (payload.hasKey("exp")) {
      long long exp = payload["exp"].ToInt();
      long long now = std::chrono::duration_cast<std::chrono::seconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
      if (now > exp) {
        return false;  // Token expired
      }
    }

    // Extract user info
    if (payload.hasKey("sub")) {
      *user_id = payload["sub"].ToString();
    }
    if (payload.hasKey("roles")) {
      *roles = payload["roles"].ToString();
    }

    return true;
  } catch (const std::exception& e) {
    return false;
  }
}

bool WorkerImpl::Initialize() { return true; }

bool WorkerImpl::ValidateRequestToken(json::JSON json_input,
                                      std::string* user_id, std::string* roles,
                                      std::string* output) {
  // Validate token
  if (!json_input.hasKey("token")) {
    *output = "{\"error\": \"Missing token\"}";
    return false;
  }
  std::string token = json_input["token"].ToString();
  if (!validate_jwt_token(token, user_id, roles)) {
    *output = "{\"error\": \"Invalid token\"}";
    return false;
  }
  return true;
}

bool WorkerImpl::HandleLogin(uint64_t handle_id, const std::string& input,
                             std::string* output) {
  // Login logic - validate credentials and return token
  try {
    auto json_input = json::JSON::Load(input);
    if (!json_input.hasKey("username") || !json_input.hasKey("password")) {
      *output = "{\"error\": \"Missing username or password\"}";
      return false;
    }
    std::string username = json_input["username"].ToString();
    std::string password = json_input["password"].ToString();

    // TODO: Get user from store and validate password
    // For now, accept any username/password and return mock data
    std::string user_id = username;
    std::string roles = "user";
    long long exp = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count() +
                    3600;  // 1 hour

    std::string token = create_jwt_token(user_id, roles, exp);

    json::JSON response = json::Object();
    response["access_token"] = token;
    response["sub"] = user_id;
    response["exp"] = (long long)exp;
    response["roles"] = roles;

    *output = response.dump();
    return true;
  } catch (const std::exception& e) {
    *output = "{\"error\": \"Invalid login request\"}";
    return false;
  }
}

bool WorkerImpl::HandleLogout(uint64_t handle_id, const std::string& input,
                              std::string* output) {
  // Simple logout - just return success
  *output = "{\"message\": \"Logged out successfully\"}";
  return true;
}

bool WorkerImpl::HandleListCollections(uint64_t handle_id,
                                       const std::string& input,
                                       std::string* output) {
  try {
    auto json_input = json::JSON::Load(input);

    std::string user_id, roles;
    if (!ValidateRequestToken(json_input, &user_id, &roles, output)) {
      return false;
    }

    auto log_record_prefix = "[id-" + std::to_string(handle_id) + "]";

    // Get preferred search worker address
    std::string search_addr = config_.GetPreferredSearchAddress();
    if (search_addr.empty()) {
      *output = "{\"error\": \"No search worker available\"}";
      return false;
    }

    // Make request to search worker /list endpoint
    http_.claim();
    std::string list_url = "http://" + search_addr + "/list";
    std::string list_request = "{}";  // Empty JSON request
    std::string list_response_str = http_.post(list_url, list_request);
    http_.release();

    // Parse the response
    try {
      auto list_response_json = json::JSON::Load(list_response_str);
      if (list_response_json.hasKey("collections")) {
        // Return the response as-is
        *output = list_response_str;
        return true;
      } else {
        *output = "{\"error\": \"Invalid response from search worker\"}";
        return false;
      }
    } catch (const std::exception& e) {
      *output = "{\"error\": \"Failed to parse search worker response\"}";
      return false;
    }
  } catch (const std::exception& e) {
    *output = "{\"error\": \"Invalid request\"}";
    return false;
  }
}

bool WorkerImpl::HandleLoadCollection(uint64_t handle_id,
                                      const std::string& input,
                                      std::string* output) {
  try {
    auto json_input = json::JSON::Load(input);

    std::string user_id, roles;
    if (!ValidateRequestToken(json_input, &user_id, &roles, output)) {
      return false;
    }

    auto log_record_prefix = "[id-" + std::to_string(handle_id) + "]";

    // Extract collection_name from input
    if (!json_input.hasKey("collection_name")) {
      *output = "{\"error\": \"Missing collection_name\"}";
      return false;
    }
    std::string collection_name = json_input["collection_name"].ToString();

    // Get preferred search worker address
    std::string search_addr = config_.GetPreferredSearchAddress();
    if (search_addr.empty()) {
      *output = "{\"error\": \"No search worker available\"}";
      return false;
    }

    // Create load request
    hakes::SearchWorkerLoadRequest load_request;
    load_request.collection_name = collection_name;
    load_request.mode = 0;  // Default mode

    std::string load_request_str =
        hakes::encode_search_worker_load_request(load_request);

    // Make request to search worker /load endpoint
    http_.claim();
    std::string load_url = "http://" + search_addr + "/load";
    std::string load_response_str = http_.post(load_url, load_request_str);
    http_.release();

    // Return the response as-is
    *output = load_response_str;
    return true;
  } catch (const std::exception& e) {
    *output = "{\"error\": \"Invalid request\"}";
    return false;
  }
}

bool WorkerImpl::HandleCheckpoint(uint64_t handle_id, const std::string& input,
                                  std::string* output) {
  try {
    auto json_input = json::JSON::Load(input);

    std::string user_id, roles;
    if (!ValidateRequestToken(json_input, &user_id, &roles, output)) {
      return false;
    }

    auto log_record_prefix = "[id-" + std::to_string(handle_id) + "]";

    // Extract collection_name from input
    if (!json_input.hasKey("collection_name")) {
      *output = "{\"error\": \"Missing collection_name\"}";
      return false;
    }
    std::string collection_name = json_input["collection_name"].ToString();

    // Get preferred search worker address
    std::string search_addr = config_.GetPreferredSearchAddress();
    if (search_addr.empty()) {
      *output = "{\"error\": \"No search worker available\"}";
      return false;
    }

    // Create checkpoint request
    hakes::SearchWorkerCheckpointRequest checkpoint_request;
    checkpoint_request.collection_name = collection_name;

    std::string checkpoint_request_str =
        hakes::encode_search_worker_checkpoint_request(checkpoint_request);

    // Make request to search worker /checkpoint endpoint
    http_.claim();
    std::string checkpoint_url = "http://" + search_addr + "/checkpoint";
    std::string checkpoint_response_str =
        http_.post(checkpoint_url, checkpoint_request_str);
    http_.release();

    // Return the response as-is
    *output = checkpoint_response_str;
    return true;
  } catch (const std::exception& e) {
    *output = "{\"error\": \"Invalid request\"}";
    return false;
  }
}

bool WorkerImpl::HandleAdd(uint64_t handle_id, const std::string& input,
                           std::string* output) {
  try {
    auto json_input = json::JSON::Load(input);

    std::string user_id, roles;
    if (!ValidateRequestToken(json_input, &user_id, &roles, output)) {
      return false;
    }

    auto log_record_prefix = "[id-" + std::to_string(handle_id) + "]";

    HakesAddResponse response;
    response.status = false;

    // now only handles add
    HakesAddRequest request;
    auto success = decode_hakes_add_request(input, &request);
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
        std::string embed_response_str =
            http_.post(embed_url, embed_request_str);
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

    // add the key value to hakes-store FIRST (following client logic)
    size_t count = 0;
    auto ids = hakes::decode_hex_int64s(add_request.ids, &count);
    assert(count == 1);

    hakes::KVServiceRequest kv_request;
    kv_request.type = "put";
    kv_request.keys = {std::to_string(ids[0])};
    kv_request.values = {request.data};

    http_.claim();
    std::string kv_resp =
        http_.post("http://" + config_.GetStoreAddr() + "/kv",
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

    auto kv_end = std::chrono::high_resolution_clock::now();
    printf("kv time: %ld us\n",
           std::chrono::duration_cast<std::chrono::microseconds>(kv_end -
                                                                 embed_end)
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
    std::vector<std::string> add_resps =
        multi_http_.post(urls, encoded_requests);
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

    auto search_end = std::chrono::high_resolution_clock::now();
    printf("search time: %ld us\n",
           std::chrono::duration_cast<std::chrono::microseconds>(search_end -
                                                                 kv_end)
               .count());

    response.status = true;
    response.msg = "success";
    encode_hakes_add_response(response, output);
    return true;
  } catch (const std::exception& e) {
    *output = "{\"error\": \"Invalid JSON input\"}";
    return false;
  }
}

bool WorkerImpl::HandleSearch(uint64_t handle_id, const std::string& input,
                              std::string* output) {
  try {
    auto json_input = json::JSON::Load(input);

    std::string user_id, roles;
    if (!ValidateRequestToken(json_input, &user_id, &roles, output)) {
      return false;
    }

    auto log_record_prefix = "[id-" + std::to_string(handle_id) + "]";

    printf("sample request: %s\n", input.c_str());

    HakesSearchResponse response;
    response.status = false;

    auto embed_start = std::chrono::high_resolution_clock::now();

    // now only handles add
    HakeSearchRequest request;
    auto success = decode_hakes_search_request(input, &request);
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
      if (config_.GetEmbedEndpointAddr().empty()) { /* Lines 242-246 omitted */
      }

      if (config_.GetEmbedEndpointType() ==
          "hakes-embed") { /* Lines 249-277 omitted */
      } else {             /* Lines 278-294 omitted */
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
    std::string search_response_str =
        http_.post(search_url, search_request_str);
    http_.release();

    hakes::SearchWorkerSearchResponse search_response;
    if (!hakes::decode_search_worker_search_response(search_response_str,
                                                     &search_response)) {
      std::cout << log_record_prefix << ":failed to decode response\n";
      response.msg = "failed to decode search service search response";
      encode_hakes_search_response(response, output);
      fflush(stdout);
      return false;
    } else if (!search_response.status) { /* Lines 329-334 omitted */
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
      if (!hakes::decode_search_worker_rerank_response(
              rerank_resp, &rerank_response)) { /* Lines 378-383 omitted */
      } else                                    /* Lines 383-389 omitted */
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
      if (ids[i] != -1) { /* Lines 440-441 omitted */
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
    } else if (!kv_response.status) { /* Lines 470-476 omitted */
    }

    response.status = true;
    response.msg = "success";
    response.data = kv_response.values;
    encode_hakes_search_response(response, output);

    printf("last response: %s\n", output->c_str());
    auto kv_end = std::chrono::high_resolution_clock::now();
    printf("kv time: %ld us\n",
           std::chrono::duration_cast<std::chrono::microseconds>(kv_end -
                                                                 merge_end)
               .count());

    // fflush(stdout);
    return true;
  } catch (const std::exception& e) {
    *output = "{\"error\": \"Invalid JSON input\"}";
    return false;
  }
}

bool WorkerImpl::HandleDelete(uint64_t handle_id, const std::string& input,
                              std::string* output) {
  try {
    auto json_input = json::JSON::Load(input);

    std::string user_id, roles;
    if (!ValidateRequestToken(json_input, &user_id, &roles, output)) {
      return false;
    }

    auto log_record_prefix = "[id-" + std::to_string(handle_id) + "]";

    // Extract collection_name and ids from input
    if (!json_input.hasKey("collection_name")) {
      *output = "{\"error\": \"Missing collection_name\"}";
      return false;
    }
    if (!json_input.hasKey("ids")) {
      *output = "{\"error\": \"Missing ids\"}";
      return false;
    }
    std::string collection_name = json_input["collection_name"].ToString();
    std::string ids_hex = json_input["ids"].ToString();

    // Delete from KV store FIRST (following client logic)
    hakes::KVServiceRequest kv_request;
    kv_request.type = "delete";

    // Decode ids to get individual IDs
    size_t count = 0;
    auto ids = hakes::decode_hex_int64s(ids_hex, &count);
    kv_request.keys.reserve(count);
    for (size_t i = 0; i < count; i++) {
      kv_request.keys.push_back(std::to_string(ids[i]));
    }

    http_.claim();
    std::string kv_resp =
        http_.post("http://" + config_.GetStoreAddr() + "/kv",
                   hakes::encode_kvservice_request(kv_request));
    http_.release();

    // Check KV store response
    try {
      hakes::KVServiceResponse kv_response;
      if (!hakes::decode_kvservice_response(kv_resp, &kv_response) ||
          !kv_response.status) {
        *output = "{\"error\": \"Failed to delete from KV store\"}";
        return false;
      }
    } catch (const std::exception& e) {
      *output = "{\"error\": \"Invalid response from KV store\"}";
      return false;
    }

    // Get all search worker addresses
    auto server_count = config_.ServerCount();
    if (server_count == 0) {
      *output = "{\"error\": \"No search workers available\"}";
      return false;
    }

    // Create delete request for search workers
    hakes::SearchWorkerDeleteRequest delete_request;
    delete_request.collection_name = collection_name;
    delete_request.ids = ids_hex;

    std::string delete_request_str =
        hakes::encode_search_worker_delete_request(delete_request);

    // Send delete request to all search workers
    std::vector<std::string> urls;
    std::vector<std::string> requests;
    urls.reserve(server_count);
    requests.reserve(server_count);
    for (int i = 0; i < server_count; i++) {
      urls.push_back("http://" + config_.GetSearchAddressByID(i) + "/delete");
      requests.push_back(delete_request_str);
    }

    multi_http_.claim();
    std::vector<std::string> delete_resps = multi_http_.post(urls, requests);
    multi_http_.release();

    // Check responses from search workers
    for (const auto& resp : delete_resps) {
      try {
        auto resp_json = json::JSON::Load(resp);
        if (!resp_json.hasKey("status") || !resp_json["status"].ToBool()) {
          *output = "{\"error\": \"Failed to delete from search worker\"}";
          return false;
        }
      } catch (const std::exception& e) {
        *output = "{\"error\": \"Invalid response from search worker\"}";
        return false;
      }
    }

    *output = "{\"status\": true, \"message\": \"deleted successfully\"}";
    return true;
  } catch (const std::exception& e) {
    *output = "{\"error\": \"Invalid JSON input\"}";
    return false;
  }
}

bool WorkerImpl::HandleRerank(uint64_t handle_id, const std::string& input,
                              std::string* output) {
  try {
    auto json_input = json::JSON::Load(input);

    std::string user_id, roles;
    if (!ValidateRequestToken(json_input, &user_id, &roles, output)) {
      return false;
    }

    auto log_record_prefix = "[id-" + std::to_string(handle_id) + "]";

    // Extract parameters from input
    if (!json_input.hasKey("collection_name") || !json_input.hasKey("query") ||
        !json_input.hasKey("k") || !json_input.hasKey("input_ids") ||
        !json_input.hasKey("metric_type")) {
      *output = "{\"error\": \"Missing required parameters\"}";
      return false;
    }

    std::string collection_name = json_input["collection_name"].ToString();
    std::string query_hex = json_input["query"].ToString();
    int k = json_input["k"].ToInt();
    std::string input_ids_hex = json_input["input_ids"].ToString();
    std::string metric_type = json_input["metric_type"].ToString();

    // Get preferred search worker address
    std::string search_addr = config_.GetPreferredSearchAddress();
    if (search_addr.empty()) {
      *output = "{\"error\": \"No search worker available\"}";
      return false;
    }

    // Create rerank request
    hakes::SearchWorkerRerankRequest rerank_request;
    rerank_request.collection_name = collection_name;
    rerank_request.k = k;
    rerank_request.vecs = query_hex;
    rerank_request.input_ids = input_ids_hex;
    rerank_request.metric_type = (metric_type == "L2") ? 0 : 1;

    // Calculate d from query_hex (assuming it's float32)
    size_t vecs_size = 0;
    auto vecs_data = hakes::decode_hex_floats(query_hex, &vecs_size);
    rerank_request.d = vecs_size;

    std::string rerank_request_str =
        hakes::encode_search_worker_rerank_request(rerank_request);

    // Make request to search worker /rerank endpoint
    http_.claim();
    std::string rerank_url = "http://" + search_addr + "/rerank";
    std::string rerank_response_str =
        http_.post(rerank_url, rerank_request_str);
    http_.release();

    // Return the response as-is
    *output = rerank_response_str;
    return true;
  } catch (const std::exception& e) {
    *output = "{\"error\": \"Invalid JSON input\"}";
    return false;
  }
}

void WorkerImpl::Close() {}

}  // namespace hakes_worker
