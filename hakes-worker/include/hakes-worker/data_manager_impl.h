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

#ifndef HAKES_HAKESWORKER_DATAMANAGERIMPL_H_
#define HAKES_HAKESWORKER_DATAMANAGERIMPL_H_

#include <string>
#include <vector>

#include "hakes-worker/data_manager.h"
#include "message/searchservice.h"

namespace hakes_worker {

class DataManagerImpl : public DataManager {
 public:
  DataManagerImpl() = default;
  virtual ~DataManagerImpl() = default;

  bool Initialize() override;

  hakes::SearchWorkerRerankResponse MergeSearchResults(
      const std::vector<hakes::SearchWorkerRerankResponse> resps, int k) override;
};
}  // namespace hakes_worker

#endif  // HAKES_HAKESWORKER_DATAMANAGERIMPL_H_
