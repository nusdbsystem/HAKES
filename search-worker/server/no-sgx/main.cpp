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
#include <filesystem>
#include <string>
#include <thread>

#include "llhttp.h"
#include "search-worker/common/worker.h"
#include "search-worker/common/workerImpl.h"
#include "search_worker.h"
#include "server/server.h"
#include "server/service.h"
#include "utils/fileutil.h"
#include "uv.h"

int main(int argc, char* argv[]) {
  // pick service selection

  if (argc != 5) {
    fprintf(stderr, "Usage: port data_path cluster_size server_id\n");
    exit(1);
  }
  // parse the port
  auto port = std::stol(argv[1]);
  if (port < 0 || port > 65535) {
    fprintf(stderr, "Invalid port number\n");
    exit(1);
  }
  std::string path = argv[2];
  int cluster_size = std::stoi(argv[3]);
  int server_id = std::stoi(argv[4]);

  // check directory exists
  if (!std::filesystem::exists(path)) {
    fprintf(stderr, "Data path does not exist: %s\n", path.c_str());
    exit(1);
  }

  search_worker::WorkerImpl* worker = new search_worker::WorkerImpl();
  worker->Initialize(false, cluster_size, server_id, path);

  hakes::Service s{
      std::unique_ptr<hakes::ServiceWorker>(new search_worker::SearchWorker(
          std::unique_ptr<search_worker::Worker>(worker)))};
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

// ./sample-server 2351 path 1 0
