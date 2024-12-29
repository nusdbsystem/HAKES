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

#ifndef HAKES_SEARCHWORKER_COMMON_WORKERIMPL_H_
#define HAKES_SEARCHWORKER_COMMON_WORKERIMPL_H_

#include <memory>

#include "search-worker/common/worker.h"
#include "search-worker/index/Index.h"
#include "search-worker/index/ext/HakesIndex.h"
#include "search-worker/index/ext/IndexFlatL.h"

namespace search_worker {

class WorkerImpl : public Worker {
 public:
  WorkerImpl() = default;
  virtual ~WorkerImpl() {}

  bool Initialize(hakes::IOReader* ff, hakes::IOReader* rf, hakes::IOReader* uf,
                  bool keep_pa, int cluster_size, int server_id) override;

  bool IsInitialized() override;

  bool AddWithIds(const char* req, size_t req_len, char* resp,
                  size_t resp_len) override;

  bool Search(const char* req, size_t req_len, char* resp,
              size_t resp_len) override;

  bool Rerank(const char* req, size_t req_len, char* resp,
              size_t resp_len) override;

  bool Close() override;

 private:
  int cluster_size_;
  int server_id_;
  std::unique_ptr<faiss::HakesIndex> index_;
};

}  // namespace search_worker

#endif  // HAKES_SEARCHWORKER_COMMON_WORKERIMPL_H_
