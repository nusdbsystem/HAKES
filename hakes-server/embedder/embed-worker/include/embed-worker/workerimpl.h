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

#ifndef HAKES_EMBEDWORKER_SRC_WORKERIMPL_H_
#define HAKES_EMBEDWORKER_SRC_WORKERIMPL_H_

#include "embed-worker/worker.h"
#include "embed-worker/inference-runtime/model.h"
#include "store-client/store.h"
#include "utils/cache.h"

namespace embed_worker {

class WorkerImpl : public Worker {
 public:
  WorkerImpl(std::unique_ptr<hakes::Store> model_store)
      : initialize_(false), model_store_(model_store.release()) {}

  ~WorkerImpl() { Close(); }

  // delete copy and move constructors and assigment operators
  WorkerImpl(const WorkerImpl&) = delete;
  WorkerImpl& operator=(const WorkerImpl&) = delete;
  WorkerImpl(WorkerImpl&&) = delete;
  WorkerImpl& operator=(WorkerImpl&&) = delete;

  bool Initialize() override;

  bool Handle(uint64_t handle_id, const std::string& sample_request,
              std::string* output) override;

  int Execute(const hakes::EmbedWorkerRequest& request, std::string* output) override;

  void Close() override;

 private:
  bool initialize_;
  hakes::Store* model_store_;
  hakes::SimpleCache<ModelContext> model_cache{1};
};

}  // namespace embed_worker

#endif  // HAKES_EMBEDWORKER_SRC_WORKERIMPL_H_
