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

#ifndef HAKES_SEARCHWORKER_SERVER_SEARCHWORKER_H_
#define HAKES_SEARCHWORKER_SERVER_SEARCHWORKER_H_

#include <memory>

#include "search-worker/common/worker.h"
#include "server/worker.h"

namespace search_worker {
class SearchWorker : public hakes::ServiceWorker {
 public:
  SearchWorker(std::unique_ptr<Worker>&& worker) : worker_(std::move(worker)){};
  ~SearchWorker();

  // delete copy and move
  SearchWorker(const SearchWorker&) = delete;
  SearchWorker(SearchWorker&&) = delete;
  SearchWorker& operator=(const SearchWorker&) = delete;
  SearchWorker& operator=(SearchWorker&&) = delete;

  bool Initialize() override;
  bool Handle(const std::string& url, const std::string& input,
              std::string* output) override;
  void Close() override;

 private:
  bool initialized_ = false;
  std::unique_ptr<Worker> worker_;
};

}  // namespace search_worker

#endif  //  HAKES_SEARCHWORKER_SERVER_SEARCHWORKER_H_