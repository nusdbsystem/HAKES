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

#ifndef HAKES_EMBEDWORKER_UNTRUSTED_WORKER_H_
#define HAKES_EMBEDWORKER_UNTRUSTED_WORKER_H_

#include <sgx_urts.h>

#include <string>

#include "embed-worker/common/worker.h"
#include "message/embed.h"
#include "store-client/store.h"

namespace embed_worker {

class WorkerU : public Worker {
 public:
  WorkerU(const std::string& enclave_file_name,
          const std::string& remote_store_path,
          const std::string& local_store_path)
      : initialize_(false),
        closed_(false),
        enclave_file_name_(std::move(enclave_file_name)),
        eid_(0),
        model_store_(hakes::OpenFsStore(std::move(remote_store_path),
                                 std::move(local_store_path))) {}

  WorkerU(const std::string& enclave_file_name,
          std::unique_ptr<hakes::Store> model_store)
      : initialize_(false),
        enclave_file_name_(std::move(enclave_file_name)),
        eid_(0),
        model_store_(model_store.release()) {}

  ~WorkerU() { Close(); }

  // delete copy and move constructors and assigment operators
  WorkerU(const WorkerU&) = delete;
  WorkerU& operator=(const WorkerU&) = delete;
  WorkerU(WorkerU&&) = delete;
  WorkerU& operator=(WorkerU&&) = delete;

  bool Initialize() override;

  bool Handle(uint64_t handle_id, const std::string& sample_request,
              std::string* output) override;

  /**
   * @brief execute the inference request in the worker managed enclave.
   *
   * @param request : user request
   * @return int : 0 for success; -1 for failure
   */
  int Execute(const hakes::EmbedWorkerRequest& request, std::string* output) override;

  void Close() override;

 private:
  bool initialize_;
  bool closed_;
  const std::string enclave_file_name_;
  sgx_enclave_id_t eid_;
  hakes::Store* model_store_;
};

}  // namespace embed_worker

#endif  // HAKES_EMBEDWORKER_UNTRUSTED_WORKER_H_
