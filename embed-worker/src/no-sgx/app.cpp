#define INFERENCE_COUNT 1
// simulated remote store latency in ms
#define SIM_REMOTE_STORE_LAT 1000
#include <cstring>

#include "embed-worker/no-sgx/workerimpl.h"
#include "utils/fileutil.h"

// for test
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

int main(int argc, char* argv[]) {
  std::cout << "Program starts at: "
            << std::chrono::system_clock::now().time_since_epoch() /
                   std::chrono::microseconds(1)
            << "\n";

  std::string store_path("tmp");
  std::string sample_request_path;
  std::string sample_request;

  // load the local sample request for testing
  if (argc == 3) {
    store_path = argv[1];
    sample_request_path = argv[2];
    if (hakes::ReadFileToString(sample_request_path, &sample_request) == -1) {
      printf("failed to load sample request\n");
      return 1;
    }
  } else if (argc == 2) {
    sample_request = argv[1];
    printf("sample request: %s\n", sample_request.c_str());
  } else {
    fprintf(stderr, "Usage: store_path sample_request_path\n");
    exit(1);
  }

  // test enclave recreate
  // for (int j = 0; j < 2; j++) {
  printf("Starting a worker\n");
  embed_worker::WorkerImpl worker(
      std::unique_ptr<hakes::Store>(hakes::OpenFsStore(
          store_path, store_path + "/cache", SIM_REMOTE_STORE_LAT)));
  auto ret = worker.Initialize();
  if (!ret) {
    printf("failed to initialize worker\n");
    return 1;
  }
  printf("Worker initialized\n");
  std::cout << "worker init done at: "
            << std::chrono::system_clock::now().time_since_epoch() /
                   std::chrono::microseconds(1)
            << "\n";

  // sequential
  // for (uint64_t i = 0; i < INFERENCE_COUNT; i ++) {
  //   worker.Handle(i, store_path, sample_request, &worker);
  // }

  // parallel
  std::vector<std::thread> handlers;
  std::array<std::string, INFERENCE_COUNT> outputs;
  for (uint64_t i = 0; i < INFERENCE_COUNT; i++) {
    handlers.emplace_back(&embed_worker::WorkerImpl::Handle, &worker, i,
                          sample_request, &outputs[i]);
  }
  for (auto& h : handlers) {
    h.join();
  }

  std::cout << "processing " << INFERENCE_COUNT << " requests done at: "
            << std::chrono::system_clock::now().time_since_epoch() /
                   std::chrono::microseconds(1)
            << "\n";

  // print response
  int id = 0;
  for (auto& output : outputs) {
    printf("{\"msg\": \"id-%d, %s\"}", id++, output.c_str());
  }

  // // test subsequent
  // worker.Handle(0, sample_request, &outputs[0]);
  // printf("{\"msg\": \"id-%d, %s\"}", 0, outputs[0].c_str());

  // // test subsequent (see reuse of model rt)
  // worker.Handle(0, sample_request, &outputs[0]);
  // printf("{\"msg\": \"id-%d, %s\"}", 0, outputs[0].c_str());

  // tear down
  printf("worker closing.\n");
  worker.Close();
  // }
  return 0;
}
