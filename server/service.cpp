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

#include "service.h"

namespace hakes {

void initialize_once(ServiceWorker* worker, response_code_t* code) {
  // execute logic on the value for initialization
  // decode json request
  // return 404 for decode failure
  printf("init execution\n");
  // execute logic on the value for initialization
  *code = worker->Initialize() ? OK : SERVER_ERROR;
}

response_code_t Service::Init() {
  response_code_t code = SERVER_ERROR;
  std::call_once(initialized_, initialize_once, worker_.get(), &code);
  return code;
}

response_code_t Service::OnWork(const std::string url,
                                const std::string& request,
                                std::string* response) {
  bool ret = worker_->Handle(url, request, response);
  return ret ? response_code_t::OK : response_code_t::SERVER_ERROR;
}

}  // namespace hakes
