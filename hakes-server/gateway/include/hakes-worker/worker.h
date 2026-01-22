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

#ifndef HAKES_HAKESWORKER_WORKER_H_
#define HAKES_HAKESWORKER_WORKER_H_

#include <string>

namespace hakes_worker {

class Worker {
 public:
  Worker() = default;

  virtual ~Worker() = default;

  // delete copy and move constructors and assigment operators
  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;
  Worker(Worker&&) = delete;
  Worker& operator=(Worker&&) = delete;

  virtual bool Initialize() = 0;

  virtual bool HandleLogin(uint64_t handle_id, const std::string& input,
                           std::string* output) = 0;

  virtual bool HandleLogout(uint64_t handle_id, const std::string& input,
                            std::string* output) = 0;

  virtual bool HandleListCollections(uint64_t handle_id, const std::string& input,
                                     std::string* output) = 0;

  virtual bool HandleLoadCollection(uint64_t handle_id, const std::string& input,
                                    std::string* output) = 0;

  virtual bool HandleCheckpoint(uint64_t handle_id, const std::string& input,
                                std::string* output) = 0;

  virtual bool HandleAdd(uint64_t handle_id, const std::string& input,
                         std::string* output) = 0;

  virtual bool HandleSearch(uint64_t handle_id, const std::string& input,
                            std::string* output) = 0;

  virtual bool HandleDelete(uint64_t handle_id, const std::string& input,
                            std::string* output) = 0;

  virtual bool HandleRerank(uint64_t handle_id, const std::string& input,
                            std::string* output) = 0;

  virtual void Close() = 0;
};

}  // namespace hakes_worker

#endif  // HAKES_HAKESWORKER_WORKER_H_
