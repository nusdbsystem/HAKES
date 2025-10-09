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

#include <cassert>
#include <cstring>
#include <filesystem>
#include <iostream>

#include "message/searchservice.h"
#include "search-worker/checkpoint.h"
#include "search-worker/index/ext/HakesFlatIndex.h"
#include "search-worker/index/ext/HakesIndex.h"
#include "search-worker/index/ext/IndexFlatL.h"
#include "search-worker/index/ext/IndexIVFPQFastScanL.h"
#include "search-worker/workerImpl.h"
#include "utils/base64.h"
#include "utils/cache.h"
#include "utils/fileutil.h"
#include "utils/hexutil.h"
#include "utils/io.h"

namespace {

std::string encode_hex_floats(const float* vecs, size_t count) {
  return hakes::encode_hex_floats(vecs, count);
}

}  // anonymous namespace

namespace search_worker {

bool WorkerImpl::Initialize(bool keep_pa, int cluster_size, int server_id,
                            const std::string& path) {
  cluster_size_ = cluster_size;
  server_id_ = server_id;
  base_path_ = path;
  return true;
}

bool WorkerImpl::Initialize(const std::string& path) {
  return Initialize(false, 1, 0, path);
}

bool WorkerImpl::IsInitialized() { return true; }

bool WorkerImpl::HasLoadedCollection(const std::string& collection_name) {
  pthread_rwlock_rdlock(&collection_mu_);
  auto loaded = collections.find(collection_name) != collections.end();
  pthread_rwlock_unlock(&collection_mu_);
  return loaded;
}

faiss::HakesCollection* WorkerImpl::GetCollection(
    const std::string& collection_name) {
  pthread_rwlock_rdlock(&collection_mu_);
  auto it = collections.find(collection_name);
  if (it == collections.end() || !it->second->is_loaded()) {
    pthread_rwlock_unlock(&collection_mu_);
    return nullptr;
  }
  auto index = it->second.get();
  pthread_rwlock_unlock(&collection_mu_);
  return index;
}

std::vector<std::string> WorkerImpl::ListCollectionsInternal() {
  if (base_path_.empty()) return {};
  std::vector<std::string> results;
  for (const auto& entry : std::filesystem::directory_iterator(base_path_)) {
    if (entry.is_directory()) {
      for (const auto& subentry :
           std::filesystem::directory_iterator(entry.path())) {
        if (subentry.is_directory() &&
            subentry.path().filename().string().rfind("checkpoint_", 0) == 0) {
          std::string rel_path =
              std::filesystem::relative(entry.path(), base_path_).string();
          results.push_back(rel_path);
          break;
        }
      }
    }
  }
  return results;
}

bool WorkerImpl::ListCollections(char* resp, size_t resp_len) {
  hakes::SearchWorkerListCollectionsReseponse list_resp;
  list_resp.collections = ListCollectionsInternal();
  auto encoded_response =
      hakes::encode_search_worker_list_collections_response(list_resp);
  assert(encoded_response.size() < resp_len);
  memcpy(resp, encoded_response.c_str(), encoded_response.size());
  resp[encoded_response.size()] = '\0';
  return true;
}

bool WorkerImpl::LoadCollectionInternal(const std::string& collection_name,
                                        int mode) {
  if (base_path_.empty()) return false;
  auto loaded = HasLoadedCollection(collection_name);
  faiss::HakesCollection* index = nullptr;
  if (loaded) return true;

  pthread_rwlock_wrlock(&collection_mu_);
  std::cout << "Load collection: " << collection_name << std::endl;
  if (collections.find(collection_name) == collections.end()) {
    collections[collection_name] =
        std::unique_ptr<faiss::HakesCollection>(new faiss::HakesIndex());
  }
  index = collections[collection_name].get();
  pthread_rwlock_unlock(&collection_mu_);
  auto collection_path = base_path_ + "/" + collection_name;
  auto checkpoint = hakes::get_latest_checkpoint_path(collection_path);
  if (!checkpoint.empty()) {
    loaded = index->Initialize(collection_path + "/" + checkpoint, mode, false);
    if (loaded) {
      index->set_loaded();
    }
    auto index_version = hakes::get_checkpoint_no(checkpoint);
    index->index_version_.store(index_version, std::memory_order_relaxed);
  }
  return loaded;
}

bool WorkerImpl::LoadCollection(const char* req, size_t req_len, char* resp,
                                size_t resp_len) {
  // decode the message
  hakes::SearchWorkerLoadRequest load_req;
  hakes::SearchWorkerLoadResponse load_resp;
  if (!hakes::decode_search_worker_load_request(std::string{req, req_len},
                                                &load_req)) {
    return false;
  }

  auto loaded = LoadCollectionInternal(load_req.collection_name, load_req.mode);

  if (!loaded) {
    load_resp.status = false;
    load_resp.msg = "load error: checkpoint not found";
  } else {
    load_resp.status = true;
    auto index = GetCollection(load_req.collection_name);
    load_resp.msg = "load success: " + load_req.collection_name + "{" +
                    index->to_string() + "}";
  }

  // encode response
  std::string encoded_response =
      hakes::encode_search_worker_load_response(load_resp);
  assert(encoded_response.size() < resp_len);
  memcpy(resp, encoded_response.c_str(), encoded_response.size());
  resp[encoded_response.size()] = '\0';

  return load_resp.status;
}

bool WorkerImpl::AddWithIdsInternal(const std::string& collection_name, int n,
                                    int d, const float* vecs,
                                    const faiss::idx_t* ids, bool base_only,
                                    std::string* error_msg) {
  auto index = GetCollection(collection_name);
  if (index == nullptr) {
    error_msg->assign("collection does not exist error");
    return false;
  }

#if DEBUG
  std::cout << "index: " << index->to_string() << std::endl;
#endif  // DEBUG

  std::unique_ptr<faiss::idx_t[]> assign =
      std::unique_ptr<faiss::idx_t[]>(new faiss::idx_t[1]);
  int vecs_t_d = 0;
  std::unique_ptr<float[]> transformed_vecs;

  return (base_only) ? index->AddBase(n, d, vecs, ids)
                     : index->AddWithIds(n, d, vecs, ids, assign.get(),
                                         &vecs_t_d, &transformed_vecs);
}

bool WorkerImpl::AddWithIds(const char* req, size_t req_len, char* resp,
                            size_t resp_len) {
#if DEBUG
  printf("Add request: %s\n", req);
#endif  // DEBUG
  // decode the message
  hakes::SearchWorkerAddRequest add_req;
  hakes::SearchWorkerAddResponse add_resp;
  if (!hakes::decode_search_worker_add_request(std::string{req, req_len},
                                               &add_req)) {
    return false;
  }

  // parse collection_name
  auto collection_name = add_req.collection_name;

  // parse ids and vecs
  size_t parsed_count;
  auto ids = hakes::decode_hex_int64s(add_req.ids, &parsed_count);
  assert(parsed_count == 1);
  auto vecs = hakes::decode_hex_floats(add_req.vecs, &parsed_count);
  assert(parsed_count == add_req.d);

  std::string error_msg;
  auto success =
      AddWithIdsInternal(collection_name, 1, add_req.d, vecs.get(), ids.get(),
                         (ids[0] % cluster_size_ != server_id_), &error_msg);

  if (!success) {
    add_resp.status = false;
    add_resp.msg = "add error" + error_msg;
  } else {
    add_resp.status = true;
    add_resp.msg = "add success";
  }

  // encode response
  std::string encoded_response =
      hakes::encode_search_worker_add_response(add_resp);
  assert(encoded_response.size() < resp_len);
  memcpy(resp, encoded_response.c_str(), encoded_response.size());
  resp[encoded_response.size()] = '\0';

  // printf("-> %ld\n", index->ntotal);
  return add_resp.status;
}

bool WorkerImpl::SearchInternal(const std::string& collection_name, int n,
                                int d, const float* query,
                                const faiss::HakesSearchParams& params,
                                std::unique_ptr<float[]>* scores,
                                std::unique_ptr<faiss::idx_t[]>* ids,
                                std::string* error_msg) {
  auto index = GetCollection(collection_name);
  if (index == nullptr) {
    error_msg->assign("collection does not exist error");
    return false;
  }
#if DEBUG
  std::cout << "index: " << index->to_string() << std::endl;
#endif  // DEBUG
  return index->Search(n, d, query, params, scores, ids);
}

bool WorkerImpl::Search(const char* req, size_t req_len, char* resp,
                        size_t resp_len) {
#if DEBUG
  printf("Search request: %s\n", req);
#endif  // DEBUG

  hakes::SearchWorkerSearchResponse search_resp;
  hakes::SearchWorkerSearchRequest search_req;
  if (!hakes::decode_search_worker_search_request(std::string{req, req_len},
                                                  &search_req)) {
    search_resp.status = false;
    search_resp.msg = "decode search request error";
    std::string encoded_response =
        hakes::encode_search_worker_search_response(search_resp);
    assert(encoded_response.size() < resp_len);
    memcpy(resp, encoded_response.c_str(), encoded_response.size());
    resp[encoded_response.size()] = '\0';
    return false;
  }

  // parse collection_name
  auto collection_name = search_req.collection_name;
  // parse ids and vecs
  size_t parsed_count;
  auto vecs = hakes::decode_hex_floats(search_req.vecs, &parsed_count);
  assert(parsed_count == search_req.d);

  std::unique_ptr<float[]> scores;
  std::unique_ptr<faiss::idx_t[]> ids;
  auto params = faiss::HakesSearchParams{search_req.nprobe, search_req.k,
                                         search_req.k_factor,
                                         faiss::METRIC_INNER_PRODUCT};
  std::string error_msg;
  bool success = SearchInternal(collection_name, 1, search_req.d, vecs.get(),
                                params, &scores, &ids, &error_msg);

  if (!success) {
    search_resp.status = false;
    search_resp.msg = "search error" + error_msg;
    search_resp.ids = "";
    search_resp.scores = "";
  } else {
    search_resp.status = true;
    search_resp.msg = "search success";
    search_resp.ids =
        hakes::encode_hex_int64s(ids.get(), search_req.k * search_req.k_factor);
    search_resp.scores = hakes::encode_hex_floats(
        scores.get(), search_req.k * search_req.k_factor);
  }
  // encode response
  std::string encoded_response =
      hakes::encode_search_worker_search_response(search_resp);
  assert(encoded_response.size() < resp_len);
  memcpy(resp, encoded_response.c_str(), encoded_response.size());
  resp[encoded_response.size()] = '\0';

  return success;
}

bool WorkerImpl::RerankInternal(
    const std::string& collection_name, int n, int d, const float* query, int k,
    faiss::idx_t* k_base_count, faiss::idx_t* base_labels,
    float* base_distances, std::unique_ptr<float[]>* scores,
    std::unique_ptr<faiss::idx_t[]>* ids, std::string* error_msg) {
  auto index = GetCollection(collection_name);
  if (index == nullptr) {
    error_msg->assign("collection does not exist error");
    return false;
  }
#if DEBUG
  std::cout << "index: " << index->to_string() << std::endl;
#endif  // DEBUG

  return index->Rerank(n, d, query, k, k_base_count, base_labels,
                       base_distances, scores, ids);
}

bool WorkerImpl::Rerank(const char* req, size_t req_len, char* resp,
                        size_t resp_len) {
#if DEBUG
  printf("Rerank request: %s\n", req);
#endif  // DEBUG

  hakes::SearchWorkerRerankResponse rerank_resp;
  hakes::SearchWorkerRerankRequest rerank_req;
  if (!hakes::decode_search_worker_rerank_request(std::string{req, req_len},
                                                  &rerank_req)) {
    rerank_resp.status = false;
    rerank_resp.msg = "decode search request error";
    rerank_resp.ids = "";
    rerank_resp.scores = "";
    std::string encoded_response =
        hakes::encode_search_worker_rerank_response(rerank_resp);
    assert(encoded_response.size() < resp_len);
    memcpy(resp, encoded_response.c_str(), encoded_response.size());
    resp[encoded_response.size()] = '\0';
    return false;
  }

  // parse collection_name
  auto collection_name = rerank_req.collection_name;

  // parse ids
  size_t parsed_count;
  auto vecs = hakes::decode_hex_floats(rerank_req.vecs, &parsed_count);
  assert(parsed_count == rerank_req.d);
  auto input_ids =
      hakes::decode_hex_int64s(rerank_req.input_ids, &parsed_count);

  // clean the ids to remove -1
  auto valid_count = 0;
  for (int i = 0; i < parsed_count; i++) {
    if (input_ids[i] != -1 && input_ids[i] % cluster_size_ == server_id_) {
      input_ids[valid_count] = input_ids[i];
      valid_count++;
    }
  }

  std::unique_ptr<float[]> scores;
  std::unique_ptr<faiss::idx_t[]> ids;

#if DEBUG
  // print the rerank request data
  printf("Rerank request: n=%d, d=%d, k=%d, metric_type=%d\n", 1, rerank_req.d,
         rerank_req.k, rerank_req.metric_type);
  for (int j = 0; j < valid_count; j++) {
    printf("base label: %ld", input_ids[j]);
  }
#endif  // DEBUG

  faiss::idx_t k_base_count = (faiss::idx_t)valid_count;
  std::unique_ptr<float[]> base_distances =
      std::unique_ptr<float[]>(new float[k_base_count]);
  std::memset(base_distances.get(), 0, sizeof(float) * k_base_count);
  std::string error_msg;
  auto success = RerankInternal(
      collection_name, 1, rerank_req.d, vecs.get(), rerank_req.k, &k_base_count,
      input_ids.get(), base_distances.get(), &scores, &ids, &error_msg);

  if (!success) {
    rerank_resp.status = false;
    rerank_resp.msg = "rerank error";
    rerank_resp.ids = "";
    rerank_resp.scores = "";
  } else {
    rerank_resp.status = true;
    rerank_resp.msg = "rerank success";
    rerank_resp.ids = hakes::encode_hex_int64s(ids.get(), rerank_req.k);
    rerank_resp.scores = encode_hex_floats(scores.get(), rerank_req.k);
  }
  // encode response
  std::string encoded_response =
      hakes::encode_search_worker_rerank_response(rerank_resp);
  assert(encoded_response.size() < resp_len);
  memcpy(resp, encoded_response.c_str(), encoded_response.size());
  resp[encoded_response.size()] = '\0';

  return success;
}

bool WorkerImpl::Delete(const char* req, size_t req_len, char* resp,
                        size_t resp_len) {
  // decode the message
  hakes::SearchWorkerDeleteRequest delete_req;
  hakes::SearchWorkerDeleteResponse delete_resp;
  if (!hakes::decode_search_worker_delete_request(std::string{req, req_len},
                                                  &delete_req)) {
    return false;
  }

  // parse collection_name
  auto collection_name = delete_req.collection_name;
  auto index = GetCollection(collection_name);
  if (index == nullptr) {
    delete_resp.status = false;
    delete_resp.msg = "collection does not exist error";
    std::string encoded_response =
        hakes::encode_search_worker_delete_response(delete_resp);
    assert(encoded_response.size() < resp_len);
    memcpy(resp, encoded_response.c_str(), encoded_response.size());
    resp[encoded_response.size()] = '\0';
    return false;
  }

  // parse ids and vecs
  size_t parsed_count;
  auto ids = hakes::decode_hex_int64s(delete_req.ids, &parsed_count);
  bool success = index->DeleteWithIds(parsed_count, ids.get());
  if (!success) {
    delete_resp.status = false;
    delete_resp.msg = "delete error";
  } else {
    delete_resp.status = true;
    delete_resp.msg = "delete success";
  }

  // encode response
  std::string encoded_response =
      hakes::encode_search_worker_delete_response(delete_resp);
  assert(encoded_response.size() < resp_len);
  memcpy(resp, encoded_response.c_str(), encoded_response.size());
  resp[encoded_response.size()] = '\0';

  return delete_resp.status;
}

bool WorkerImpl::CheckpointInternal(const std::string& collection_name,
                                    std::string* error_msg) {
  auto index = GetCollection(collection_name);
  bool success = false;
  if (index == nullptr) {
    error_msg->assign("collection does not exist error");
    return false;
  }
  // checkpoint the index
  std::string checkpoint_path =
      base_path_ + "/" + collection_name + "/" +
      hakes::format_checkpoint_path(
          index->index_version_.fetch_add(1, std::memory_order_relaxed) + 1);
  std::filesystem::create_directories(checkpoint_path);
  std::filesystem::permissions(checkpoint_path,
                               std::filesystem::perms::owner_all |
                                   std::filesystem::perms::group_all |
                                   std::filesystem::perms::others_all,
                               std::filesystem::perm_options::add);
  return index->Checkpoint(checkpoint_path);
}

bool WorkerImpl::Checkpoint(const char* req, size_t req_len, char* resp,
                            size_t resp_len) {
  hakes::SearchWorkerCheckpointRequest checkpoint_req;
  hakes::SearchWorkerCheckpointResponse checkpoint_resp;
  if (!hakes::decode_search_worker_checkpoint_request(std::string{req, req_len},
                                                      &checkpoint_req)) {
    return false;
  }

  std::string error_msg;
  auto success = CheckpointInternal(checkpoint_req.collection_name, &error_msg);
  if (!success) {
    checkpoint_resp.status = false;
    checkpoint_resp.msg = "checkpoint error: " + checkpoint_resp.msg;
  } else {
    checkpoint_resp.status = true;
    checkpoint_resp.msg = "checkpoint success";
  }

  // encode response
  std::string encoded_response =
      hakes::encode_search_worker_checkpoint_response(checkpoint_resp);
  assert(encoded_response.size() < resp_len);
  memcpy(resp, encoded_response.c_str(), encoded_response.size());
  resp[encoded_response.size()] = '\0';

  return success;
}

bool WorkerImpl::Close() { return true; }

}  // namespace search_worker
