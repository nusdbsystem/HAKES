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

#include "search-worker/untrusted/worker_u.h"

#include <sgx_urts.h>

#include "Enclave_u.h"

namespace search_worker {

bool WorkerU::Initialize(const char *index_data, size_t index_len, int cluster_size, int server_id) {
  sgx_launch_token_t t;
  int updated = 0;
  memset(t, 0, sizeof(sgx_launch_token_t));
  sgx_status_t status = sgx_create_enclave(
      enclave_file_name_.c_str(), SGX_DEBUG_FLAG, &t, &updated, &eid_, NULL);
  if (status != SGX_SUCCESS) {
    printf("Failed to create enclave : error %d - %#x\n", status, status);
    return false;
  }

  printf("Enclave created\n");


  // std::string config = "config";
  sgx_status_t exec_status;
  // status = ecall_init(eid_, &exec_status, config.c_str(), config.size());
  status = ecall_init(eid_, &exec_status, index_data, index_len, cluster_size, server_id);
  if ((status != SGX_SUCCESS) || (exec_status != SGX_SUCCESS)) {
    printf("Failed to init : error (%d - %#x, %d - %#x)\n", status, status,
           exec_status, exec_status);
    return false;
  }

  printf("Worker initialized\n");

  initialized_ = true;
  return true;
}

bool WorkerU::IsInitialized() { return initialized_; }

bool WorkerU::AddWithIds(const char* req, size_t req_len, char* resp,
                         size_t resp_len) {
  sgx_status_t exec_status;
  sgx_status_t status =
      ecall_add_with_ids(eid_, &exec_status, req, req_len, resp, resp_len);
  if ((status != SGX_SUCCESS) || (exec_status != SGX_SUCCESS)) {
    printf("Failed to add with ids : error (%d - %#x, %d - %#x)\n", status,
           status, exec_status, exec_status);
    fflush(stdout);
    return false;
  }
  return true;
}

bool WorkerU::Search(const char* req, size_t req_len, char* resp,
                     size_t resp_len) {
  sgx_status_t exec_status;
  sgx_status_t status =
      ecall_search(eid_, &exec_status, req, req_len, resp, resp_len);
  if ((status != SGX_SUCCESS) || (exec_status != SGX_SUCCESS)) {
    printf("Failed to search : error (%d - %#x, %d - %#x)\n", status, status,
           exec_status, exec_status);
    fflush(stdout);
    return false;
  }
  return true;
}

bool WorkerU::Rerank(const char* req, size_t req_len, char* resp,
                     size_t resp_len) {
  sgx_status_t exec_status;
  sgx_status_t status =
      ecall_rerank(eid_, &exec_status, req, req_len, resp, resp_len);
  if ((status != SGX_SUCCESS) || (exec_status != SGX_SUCCESS)) {
    printf("Failed to rerank : error (%d - %#x, %d - %#x)\n", status, status,
           exec_status, exec_status);
    fflush(stdout);
    return false;
  }
  return true;
}

bool WorkerU::Close() {
  if (!initialized_) {
    return true;
  }
  initialized_ = false;
  sgx_status_t status = ecall_clear_exec_context(eid_);
  printf("Clear exec context status: %d\n", status);
  status = sgx_destroy_enclave(eid_);
  printf("Destroy enclave status: %d\n", status);
  return true;
}

}  // namespace search_worker