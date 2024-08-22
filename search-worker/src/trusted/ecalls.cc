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

#include <omp.h>

#include <cassert>
#include <cstring>
#include <string>

#include "Enclave_t.h"
#include "search-worker/common/workerImpl.h"
#include "sgx_trts.h"

search_worker::WorkerImpl worker;

sgx_status_t ecall_init(const char* config, size_t config_len, int cluster_size,
                        int server_id) {
  omp_set_num_threads(4);
  // [TODO] obtain the model(index) owner key and decrypt the index data
  return worker.Initialize(config, config_len, cluster_size, server_id)
             ? SGX_SUCCESS
             : SGX_ERROR_UNEXPECTED;
}

sgx_status_t ecall_add_with_ids(const char* ereq, size_t ereq_len, char* eresp,
                                size_t eresp_len) {
  return worker.AddWithIds(ereq, ereq_len, eresp, eresp_len)
             ? SGX_SUCCESS
             : SGX_ERROR_UNEXPECTED;
}

sgx_status_t ecall_search(const char* ereq, size_t ereq_len, char* eresp,
                          size_t eresp_len) {
  return worker.Search(ereq, ereq_len, eresp, eresp_len) ? SGX_SUCCESS
                                                         : SGX_ERROR_UNEXPECTED;
}

sgx_status_t ecall_rerank(const char* ereq, size_t ereq_len, char* eresp,
                          size_t eresp_len) {
  return worker.Rerank(ereq, ereq_len, eresp, eresp_len) ? SGX_SUCCESS
                                                         : SGX_ERROR_UNEXPECTED;
}

void ecall_clear_exec_context() {}
