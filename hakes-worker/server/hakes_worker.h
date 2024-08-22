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

#ifndef HAKES_HAKESWORKER_SERVER_HAKESWORKER_H_
#define HAKES_HAKESWORKER_SERVER_HAKESWORKER_H_

#include <memory>

#include "hakes-worker/common/worker.h"
#include "server/worker.h"

namespace hakes_worker {

class HakesWorker : public hakes::ServiceWorker {
 public:
  HakesWorker(std::unique_ptr<Worker>&& worker, bool is_ow_action = false)
      : initialized_(false),
        is_ow_action_(is_ow_action),
        worker_(std::move(worker)){};
  ~HakesWorker();

  // delete copy and move
  HakesWorker(const HakesWorker&) = delete;
  HakesWorker(HakesWorker&&) = delete;
  HakesWorker& operator=(const HakesWorker&) = delete;
  HakesWorker& operator=(HakesWorker&&) = delete;

  bool Initialize() override;
  bool Handle(const std::string& url, const std::string& input,
              std::string* output) override;
  void Close() override;

 private:
  bool initialized_;
  bool is_ow_action_;
  std::unique_ptr<Worker> worker_;
};

}  // namespace hakes_worker

#endif  // HAKES_HAKESWORKER_SERVER_HAKESWORKER_H_
