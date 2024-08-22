#include "embed-worker/untrusted/worker_u.h"

#include <sgx_urts.h>

#include <cassert>

#include "Enclave_u.h"
#include "message/embed.h"
#include "utils/hexutil.h"

// for test
#include <chrono>
#include <iostream>
#include <mutex>

namespace {
// #ifndef NDEBUG
std::mutex debug_log_mutex;
// #endif // NDEBUG

// model is stored at the untrusted side
char* g_loaded_model = nullptr;
}  // anonymous namespace

void ocall_free_loaded(const char* /*key*/, size_t /*len*/, void* /*store*/) {
  // if (g_loaded_model) free(g_loaded_model);
  if (g_loaded_model) delete[] g_loaded_model;
  g_loaded_model = nullptr;
}

void ocall_load_content(const char* key, size_t len, char** value, size_t* vlen,
                        void* store) {
  {
    std::lock_guard<std::mutex> lg(debug_log_mutex);
    std::cout << "model init start at: "
              << std::chrono::system_clock::now().time_since_epoch() /
                     std::chrono::microseconds(1)
              << "\n";
  }
  if (g_loaded_model) printf("buffer in use error\n");
  std::string model_name{key, len};
  size_t loaded_len = 0;
  std::unique_ptr<char[]> loaded =
      reinterpret_cast<hakes::Store*>(store)->Get(model_name, &loaded_len);
  if (!loaded) {
    printf("Failed to download model\n");
    return;
  }

  if (loaded_len == 0) printf("loaded none\n");
  g_loaded_model = loaded.get();
  loaded.release();

  *value = g_loaded_model;
  *vlen = loaded_len;

  {
    std::lock_guard<std::mutex> lg(debug_log_mutex);
    // printf("bytes loaded %ld\n", loaded.size());
    printf("file name: %s\n", model_name.c_str());
    printf("bytes loaded %ld\n", loaded_len);
    std::cout << "model init done at: "
              << std::chrono::system_clock::now().time_since_epoch() /
                     std::chrono::microseconds(1)
              << "\n";
  }
}

namespace embed_worker {

bool WorkerU::Initialize() {
  std::cout << "init started at: "
            << std::chrono::system_clock::now().time_since_epoch() /
                   std::chrono::microseconds(1)
            << "\n";

  sgx_launch_token_t t;
  int updated = 0;
  memset(t, 0, sizeof(sgx_launch_token_t));
  auto sgxStatus = sgx_create_enclave(
      enclave_file_name_.c_str(), SGX_DEBUG_FLAG, &t, &updated, &eid_, NULL);
  if (sgxStatus != SGX_SUCCESS) {
    printf("Failed to create Enclave : error %d - %#x.\n", sgxStatus,
           sgxStatus);
    return false;
  } else
    printf("Enclave launched.\n");

  std::cout << "init finished at: "
            << std::chrono::system_clock::now().time_since_epoch() /
                   std::chrono::microseconds(1)
            << "\n";

  initialize_ = true;
  return true;
}

int WorkerU::Execute(const hakes::EmbedWorkerRequest& request,
                     std::string* output) {
  if (!initialize_) return -1;
  assert(model_store_);
  sgx_status_t retval;
  size_t output_size;

  std::cout << "exec started at: "
            << std::chrono::system_clock::now().time_since_epoch() /
                   std::chrono::microseconds(1)
            << "\n";
#ifndef NDEBUG
  // printf("encrypted sample size: %ld \n", request.encrypted_sample_.size());
#endif  // NDEBUG
  enc_model_inference(eid_, &retval, request.user_id_.data(),
                      request.user_id_.size(), request.encrypted_sample_.data(),
                      request.encrypted_sample_.size(),
                      request.model_name_.data(), request.model_name_.size(),
                      (void*)model_store_, request.key_service_address_.data(),
                      request.key_service_address_.size(),
                      request.key_service_port_, &output_size);
  if (retval != SGX_SUCCESS) {
    printf("Failed to run model inference : error %d - %#x.\n", retval, retval);
    *output = "failed on model inference " + std::to_string(retval);
    return -1;
  }

  // std::cout << "model inference ecall done at: "
  //   << std::chrono::system_clock::now().time_since_epoch()
  //   / std::chrono::microseconds(1) << "\n";

  uint8_t* output_buf = (uint8_t*)malloc(output_size);
  if (output_buf == NULL) {
    printf("malloc failed for prediction result buffer %ld\n", output_size);
    return -1;
  }

  enc_get_encrypted_prediction(eid_, &retval, output_buf, output_size);
  if (retval != SGX_SUCCESS) {
    printf("Failed to get encrypted result : error %d - %#x.\n", retval,
           retval);
    *output = "failed on encrypted result " + std::to_string(retval);
    return -1;
  }
  *output = std::string((char*)output_buf, output_size);
  free(output_buf);

  // std::cout << "encrypted result retrieved at: "
  //   << std::chrono::system_clock::now().time_since_epoch()
  //   / std::chrono::microseconds(1) << "\n";

  {
    std::lock_guard<std::mutex> lg(debug_log_mutex);
    std::cout << "inference done at: "
              << std::chrono::system_clock::now().time_since_epoch() /
                     std::chrono::microseconds(1)
              << "\n";
  }

  return 0;
}

void WorkerU::Close() {
  if (closed_) return;
  closed_ = true;
  initialize_ = false;
  sgx_status_t ret = enc_clear_exec_context(eid_);
  printf("returned status from close %d\n", ret);
  assert(ret == SGX_SUCCESS);
  ret = sgx_destroy_enclave(eid_);
  assert(ret == SGX_SUCCESS);
  if (model_store_) model_store_->Close();
  delete model_store_;
  model_store_ = nullptr;
}

/**
 * @brief The worker Handle is thread safe
 *
 * 1. decode user request
 * 2. communicate with key service to fetch the key enclave's key
 * 3. ecall for trusted inference
 * 4. return output to the user
 */
bool WorkerU::Handle(uint64_t handle_id, const std::string& sample_request,
                     std::string* output) {
  auto msg_prefix = "[id-" + std::to_string(handle_id) + "]";

#ifndef NDEBUG
  {
    std::lock_guard<std::mutex> lg(debug_log_mutex);
    std::cout << msg_prefix << " request handle start at: "
              << std::chrono::system_clock::now().time_since_epoch() /
                     std::chrono::microseconds(1)
              << "\n";
  }
#endif  // NDEBUG

  hakes::EmbedWorkerRequest request =
      hakes::DecodeEmbedWorkerRequest(sample_request);

  std::string prediction;
  auto ret = Execute(request, &prediction);

#ifndef NDEBUG
  {
    std::lock_guard<std::mutex> lg(debug_log_mutex);

    std::cout << msg_prefix << "prediction done at: "
              << std::chrono::system_clock::now().time_since_epoch() /
                     std::chrono::microseconds(1)
              << "\n";
  }
#endif  // NDEBUG

  // const char* hex_result = hexstring(prediction.data(), prediction.size());
  std::string hex_result =
      hakes::hex_encode(prediction.data(), prediction.size());
  if (ret == 0) {
    hakes::EmbedWorkerResponse resp;
    resp.status = true;
    resp.output = hex_result;
    resp.aux = "embed_aux";
    output->assign(resp.EncodeTo());
  } else {
    hakes::EmbedWorkerResponse resp;
    resp.status = false;
    resp.output = "prediction error";
    output->assign(resp.EncodeTo());
  }

  if (g_loaded_model) free(g_loaded_model);
  g_loaded_model = nullptr;

  return (ret == 0);
}

}  // namespace embed_worker
