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

/**
 * implement the ocalls defined in worker edl files.
 * packaged into libworker.a
 */

#include <cassert>
#include <climits>
#include <cstdio>
#include <cstring>

#include "Enclave_u.h"  // include this helps to solve the undefined reference error for ocalls. Enclave_u.c is compiled as c.

void ocall_debug_print(const void* s, size_t len) {
  assert(len < INT_MAX);
  printf("DEBUG PRINT: %.*s\n", (int)len, (const char*)s);
}
void ocall_debug_print_string(const char* s) {
  printf("DEBUG PRINT: %s\n", s);
  fflush(stdout);
}
void ocall_debug_print_hexstring(const char* s) {
  printf("DEBUG PRINT (hex): ");
  for (unsigned int i = 0; i < strlen(s); i++) {
    printf("%02hhx", (unsigned char)s[i]);
  }
  printf("\n");
}
void ocall_debug_print_hex(const void* s, size_t len) {
  printf("DEBUG PRINT (hex): ");
  auto it = (const unsigned char*)s;
  for (unsigned int i = 0; i < len; i++) {
    printf("%02hhx", *(it++));
  }
  printf("\n");
}
