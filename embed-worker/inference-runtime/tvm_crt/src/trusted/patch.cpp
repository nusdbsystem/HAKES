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

#include "Enclave_t.h"
#include "common/abstract.h"

void tvm_default_print_string(const char* str) { ocall_tvm_print_string(str); }

void tvm_default_print_time() { ocall_print_time(); }

void tvm_crt_exit(int error_code) { ocall_tvm_crt_exit(error_code); }
