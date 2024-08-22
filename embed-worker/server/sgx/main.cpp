#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "embed-worker/common/worker.h"
#include "embed-worker/untrusted/worker_u.h"
#include "embed_worker.h"
#include "llhttp.h"
#include "server/server.h"
#include "server/service.h"
#include "uv.h"

int main(int argc, char* argv[]) {
  // pick service selection

  if (argc != 5) {
    fprintf(stderr, "Usage: port data_path is_ow_action enclave_file\n");
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
  std::string enclave_file = argv[4];
  printf("Input arguments: port: %s, model store: %s is ow action: %s\n",
         argv[1], argv[2], (is_ow_action ? "true" : "false"));

  auto model_store = hakes::OpenFsStore(path, path + "/cache", 0);
  hakes::Service s{
      std::unique_ptr<hakes::ServiceWorker>(new embed_worker::EmbedWorker(
          std::unique_ptr<embed_worker::Worker>(new embed_worker::WorkerU(
              enclave_file, std::unique_ptr<hakes::Store>(
                                hakes::OpenFsStore(path, path + "/cache", 0)))),
          is_ow_action))};
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

// UV_THREADPOOL_SIZE=10 SGX_AESM_ADDR=1 ./install/bin/embed_server 2355 tmp 0 ./install/lib/Worker_Enclave.signed.so
