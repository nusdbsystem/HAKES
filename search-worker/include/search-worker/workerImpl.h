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

#ifndef HAKES_SEARCHWORKER_WORKERIMPL_H_
#define HAKES_SEARCHWORKER_WORKERIMPL_H_

#include <memory>

#include "search-worker/index/Index.h"
#include "search-worker/index/ext/HakesCollection.h"
#include "search-worker/index/ext/IndexFlatL.h"
#include "search-worker/worker.h"

namespace search_worker {

class WorkerImpl : public Worker {
 public:
  WorkerImpl() { pthread_rwlock_init(&collection_mu_, nullptr); };
  virtual ~WorkerImpl() { pthread_rwlock_destroy(&collection_mu_); }

  bool Initialize(bool keep_pa, int cluster_size, int server_id,
                  const std::string& path) override;

  bool Initialize(const std::string& path) override;

  bool IsInitialized() override;

  bool HasLoadedCollection(const std::string& collection_name) override;

  bool ListCollections(char* resp, size_t resp_len) override;

  bool LoadCollection(const char* req, size_t req_len, char* resp,
                      size_t resp_len) override;

  bool AddWithIds(const char* req, size_t req_len, char* resp,
                  size_t resp_len) override;

  bool Search(const char* req, size_t req_len, char* resp,
              size_t resp_len) override;

  bool Rerank(const char* req, size_t req_len, char* resp,
              size_t resp_len) override;

  bool Delete(const char* req, size_t req_len, char* resp,
              size_t resp_len) override;

  bool Checkpoint(const char* req, size_t req_len, char* resp,
                  size_t resp_len) override;

  bool Close() override;

  faiss::HakesCollection* GetCollection(const std::string& collection_name);

  std::vector<std::string> ListCollectionsInternal();

  bool LoadCollectionInternal(const std::string& collection_name, int mode = 0);

  bool AddWithIdsInternal(const std::string& collection_name, int n, int d,
                          const float* vecs, const faiss::idx_t* ids,
                          bool base_only, std::string* error_msg);

  bool SearchInternal(const std::string& collection_name, int n, int d,
                      const float* query,
                      const faiss::HakesSearchParams& params,
                      std::unique_ptr<float[]>* scores,
                      std::unique_ptr<faiss::idx_t[]>* ids,
                      std::string* error_msg);

  bool RerankInternal(const std::string& collection_name, int n, int d,
                      const float* query, int k, faiss::idx_t* k_base_count,
                      faiss::idx_t* base_labels, float* base_distances,
                      std::unique_ptr<float[]>* scores,
                      std::unique_ptr<faiss::idx_t[]>* ids,
                      std::string* error_msg);

  //   bool DeleteInternal(const char* req, size_t req_len, char* resp,
  //               size_t resp_len);

  bool CheckpointInternal(const std::string& collection_name, std::string* error_msg);

 private:
  int cluster_size_;
  int server_id_;
  std::string base_path_;
  mutable pthread_rwlock_t collection_mu_;
  std::unordered_map<std::string, std::unique_ptr<faiss::HakesCollection>>
      collections;
};

}  // namespace search_worker

#endif  // HAKES_SEARCHWORKER_WORKERIMPL_H_
