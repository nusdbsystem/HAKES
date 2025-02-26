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

#ifndef HAKES_SEARCHWORKER_UNTRUSTED_WORKER_U_H_
#define HAKES_SEARCHWORKER_UNTRUSTED_WORKER_U_H_

#include <sgx_eid.h>

#include <cassert>
#include <cstring>
#include <string>

#include "search-worker/common/worker.h"

namespace search_worker {

class WorkerU : public Worker {
 public:
  WorkerU(const std::string& enclave_file_name)
      : initialized_(false), enclave_file_name_(enclave_file_name) {}
  virtual ~WorkerU() {}

  WorkerU(const WorkerU&) = delete;
  WorkerU& operator=(const WorkerU&) = delete;
  WorkerU(WorkerU&&) = delete;
  WorkerU& operator=(WorkerU&&) = delete;

  // bool Initilize() override;
  bool Initialize(const std::string &collection_name, const char* index_data, size_t index_len, int cluster_size, int server_id) override;

  bool IsInitialized(const std::string &collection_name) override;

  bool AddWithIds(const char* req, size_t req_len, char* resp,
                  size_t resp_len) override;

  bool Search(const char* req, size_t req_len, char* resp,
              size_t resp_len) override;

  bool Rerank(const char* req, size_t req_len, char* resp,
              size_t resp_len) override;

  bool Close() override;

 private:
  bool initialized_;
  const std::string enclave_file_name_;
  sgx_enclave_id_t eid_;
};

}  // namespace search_worker

#endif  // HAKES_SEARCHWORKER_UNTRUSTED_WORKER_U_H_
