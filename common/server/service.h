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

#ifndef HAKES_SERVER_SERVICE_H_
#define HAKES_SERVER_SERVICE_H_

#include <memory>
#include <mutex>
#include <string>

#include "worker.h"

// a further level of abstraction over the original index server to just expose
// a single handle, to be hosted by a server.
namespace hakes {

typedef enum {
  OK,           // 200
  FORBIDDEN,    // 403
  NOT_FOUND,    // 404
  SERVER_ERROR  // 500
} response_code_t;

class Service {
 public:
  Service(std::unique_ptr<ServiceWorker> worker) {
    worker_.reset(worker.release());
  }

  Service() = delete;
  virtual ~Service() = default;

  // delete copy constructor and assignment operator
  Service(const Service&) = delete;
  Service& operator=(const Service&) = delete;
  // delete move constructor and assignment operator
  Service(Service&&) = delete;
  Service& operator=(Service&&) = delete;

  virtual response_code_t Init();

  virtual response_code_t OnWork(const std::string url,
                                 const std::string& request,
                                 std::string* response);

  std::once_flag initialized_;
  std::unique_ptr<ServiceWorker> worker_;
};

}  // namespace hakes

#endif  // HAKES_SERVER_SERVICE_H_
