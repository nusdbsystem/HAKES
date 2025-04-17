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

#include "search_worker.h"

#include <cassert>
#include <cstring>

namespace search_worker {
SearchWorker::~SearchWorker() {
  if (initialized_) {
    Close();
  }
}

bool SearchWorker::Initialize() {
  if (initialized_) {
    return true;
  }

  if (!worker_->IsInitialized()) {
    return false;
  }

  initialized_ = true;
  return true;
}

bool SearchWorker::Handle(const std::string& url, const std::string& input,
                          std::string* output) {
  assert(initialized_);
  // allocate 1MB buffer
  std::unique_ptr<char[]> buf = std::make_unique<char[]>(1024 * 1024);
  memset(&buf[0], 0, 1024 * 1024);

  if (url == "/add") {
    auto success = worker_->AddWithIds(input.c_str(), input.size(), buf.get(),
                                       1024 * 1024);
    output->assign(&buf[0]);
    return success;
  } else if (url == "/search") {
    auto success = worker_->Search(input.c_str(), input.size(), buf.get(),
                                   1024 * 1024);
    output->assign(&buf[0]);
    return success;
  } else if (url == "/rerank") {
    auto success = worker_->Rerank(input.c_str(), input.size(), buf.get(),
                                   1024 * 1024);
    output->assign(&buf[0]);
    return success;
  } else if (url == "/load") {
    auto success = worker_->LoadCollection(input.c_str(), input.size(), buf.get(),
                                   1024 * 1024);
    output->assign(&buf[0]);
    return success;
  } else if (url == "/delete") {
    auto success = worker_->Delete(input.c_str(), input.size(), buf.get(),
                                   1024 * 1024);
    output->assign(&buf[0]);
    return success;
  }

  return false;
}

void SearchWorker::Close() {
  if (!initialized_) {
    return;
  }

  worker_->Close();
  initialized_ = false;
}

}  // namespace search_worker
