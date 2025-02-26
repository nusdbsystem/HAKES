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

#ifndef HAKES_SEARCHWORKER_COMMON_WORKER_H_
#define HAKES_SEARCHWORKER_COMMON_WORKER_H_

#include <cstddef>

#include "utils/io.h"

namespace search_worker {

class Worker {
 public:
  Worker() = default;
  virtual ~Worker() {}

  virtual bool Initialize(std::string collection_name, hakes::IOReader* ff, hakes::IOReader* rf,
                          hakes::IOReader* uf, bool keep_pa, int cluster_size,
                          int server_id) = 0;

  // the worker shall be pre-initialized before installing to a search worker
  virtual bool IsInitialized(std::string collection_name) = 0;

  virtual bool AddWithIds(const char* req, size_t req_len, char* resp,
                          size_t resp_len) = 0;

  virtual bool Search(const char* req, size_t req_len, char* resp,
                      size_t resp_len) = 0;

  virtual bool Rerank(const char* req, size_t req_len, char* resp,
                      size_t resp_len) = 0;

  virtual bool Close() = 0;
};

}  // namespace search_worker

#endif  // HAKES_SEARCHWORKER_COMMON_WORKER_H_
