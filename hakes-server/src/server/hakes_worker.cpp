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

#include "server/hakes_worker.h"

#include <cassert>
#include <chrono>
#include <cstring>
#include <sstream>

#include "utils/json.h"
#include "utils/ow_message.h"

namespace hakes_worker {

HakesWorker::~HakesWorker() {
  if (initialized_) {
    Close();
  }
}

bool HakesWorker::Initialize() {
  if (initialized_) {
    return true;
  }
  initialized_ = worker_->Initialize();
  return initialized_;
}

bool HakesWorker::Handle(const std::string& url, const std::string& input,
                         std::string* output) {
  auto start_time = std::chrono::high_resolution_clock::now();
  printf("HakesWorker::Handle starts at %ld\n",
         std::chrono::duration_cast<std::chrono::microseconds>(
             start_time.time_since_epoch())
             .count());

  assert(initialized_);
  // allocate 1MB buffer
  std::unique_ptr<char[]> buf = std::make_unique<char[]>(1024 * 1024);
  memset(&buf[0], 0, 1024 * 1024);

  std::string req = is_ow_action_
                        ? ow_message::extract_ow_input(std::move(input))
                        : std::move(input);

  bool success = false;

  // Handle authentication endpoints
  if (url == "/login") {
    success = worker_->HandleLogin(0, req, output);
  } else if (url == "/logout") {
    success = worker_->HandleLogout(0, req, output);
  }
  // Handle collection management endpoints
  else if (url == "/list") {
    success = worker_->HandleListCollections(0, req, output);
  } else if (url == "/load") {
    success = worker_->HandleLoadCollection(0, req, output);
  } else if (url == "/checkpoint") {
    success = worker_->HandleCheckpoint(0, req, output);
  }
  // Handle data operations
  else if (url == "/add") {
    success = worker_->HandleAdd(0, req, output);
  } else if (url == "/search") {
    success = worker_->HandleSearch(0, req, output);
  } else if (url == "/delete") {
    success = worker_->HandleDelete(0, req, output);
  } else if (url == "/rerank") {
    success = worker_->HandleRerank(0, req, output);
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  printf("HakesWorker::Handle finished at %ld\n",
         std::chrono::duration_cast<std::chrono::microseconds>(
             end_time.time_since_epoch())
             .count());
  printf("HakesWorker::Handle duration: %ld (us)\n",
         std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                               start_time)
             .count());
  fflush(stdout);
  return success;
}

void HakesWorker::Close() {
  if (!initialized_) {
    return;
  }

  worker_->Close();
  initialized_ = false;
}

}  // namespace hakes_worker
