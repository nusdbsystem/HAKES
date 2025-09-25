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

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "hakes-worker/config.h"
#include "hakes-worker/data_manager.h"
#include "hakes-worker/worker.h"
#include "hakes-worker/workerimpl.h"
#include "hakes-worker/data_manager_impl.h"
#include "hakes_worker.h"
#include "llhttp.h"
#include "server/server.h"
#include "server/service.h"
#include "utils/fileutil.h"
#include "uv.h"

int main(int argc, char* argv[]) {
  // pick service selection

  if (argc != 4) {
    fprintf(stderr, "Usage: port config_file is_ow_action\n");
    exit(1);
  }
  // parse the port
  auto port = std::stol(argv[1]);
  if (port < 0 || port > 65535) {
    fprintf(stderr, "Invalid port number\n");
    exit(1);
  }
  std::string path = argv[2];
  bool is_ow_action = std::stoi(argv[3]);
  printf("Input arguments: port: %s, config file: %s is ow action: %s\n",
         argv[1], argv[2], (is_ow_action ? "true" : "false"));

  size_t content_len = 0;
  auto content = hakes::ReadFileToCharArray(path.c_str(), &content_len);
  if (content == nullptr) {
    fprintf(stderr, "Failed to load the data\n");
    exit(1);
  }
  hakes_worker::HakesWorkerConfig cfg = hakes_worker::ParseHakesWorkerConfig(
      std::string(content.get(), content_len));

  auto data_manager = std::unique_ptr<hakes_worker::DataManagerImpl>(
      new hakes_worker::DataManagerImpl());
  data_manager->Initialize();
  auto worker = std::unique_ptr<hakes_worker::WorkerImpl>(
      new hakes_worker::WorkerImpl(cfg, std::move(data_manager)));

  hakes::Service s{std::unique_ptr<hakes::ServiceWorker>(
      new hakes_worker::HakesWorker(std::move(worker), is_ow_action))};
  hakes::Server server(port, &s);
  if (!server.Init()) {
    fprintf(stderr, "Failed to initialize the server\n");
    exit(1);
  }
  printf("Service initialized\n");

  printf("Server starting\n");
  server.Start();

  return 0;
}

// UV_THREADPOOL_SIZE=10 ./install/bin/hakes_server 2355 ../data/hakesworker/sample-config.json 0
