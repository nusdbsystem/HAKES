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

#ifndef HAKES_HAKESWORKER_WORKERIMPL_H_
#define HAKES_HAKESWORKER_WORKERIMPL_H_

#include "hakes-worker/config.h"
#include "hakes-worker/data_manager.h"
#include "hakes-worker/worker.h"
#include "utils/http.h"

namespace hakes_worker {

class WorkerImpl : public Worker {
 public:
  WorkerImpl(HakesWorkerConfig config, std::unique_ptr<DataManager>&& dm)
      : config_(config), dm_(std::move(dm)){};

  ~WorkerImpl() override = default;

  // delete copy and move constructors and assigment operators
  WorkerImpl(const WorkerImpl&) = delete;
  WorkerImpl& operator=(const WorkerImpl&) = delete;
  WorkerImpl(WorkerImpl&&) = delete;
  WorkerImpl& operator=(WorkerImpl&&) = delete;

  bool Initialize() override;

  bool HandleKvOp(uint64_t handle_id, const std::string& sample_request,
                  std::string* output) override;

  bool HandleSearchOp(uint64_t handle_id, const std::string& sample_request,
                      std::string* output) override;

  void Close() override;

 private:
  HakesWorkerConfig config_;
  hakes::HttpClient http_;
  hakes::MultiHttpClient multi_http_;
  std::unique_ptr<DataManager> dm_;
};

}  // namespace hakes_worker

#endif  // HAKES_HAKESWORKER_WORKERIMPL_H_
