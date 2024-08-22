/**
 * implement the ocalls defined in worker edl files.
 * packaged into libworker.a
*/

#include "Enclave_u.h" // include this helps to solve the undefined reference error for ocalls. Enclave_u.c is compiled as c.

#include <cassert>
#include <climits>
#include <cstdio>
#include <cstring>


void ocall_debug_print(const void* s, size_t len) {
  assert(len < INT_MAX);
  printf("DEBUG PRINT: %.*s\n", (int) len, (const char*) s);
}
void ocall_debug_print_string(const char* s) {
  printf("DEBUG PRINT: %s\n", s);
}
void ocall_debug_print_hexstring(const char* s) {
  printf("DEBUG PRINT (hex): ");
  for (unsigned int i = 0; i < strlen(s); i++) {
    printf("%02hhx", (unsigned char) s[i]);
  }
  printf("\n");
}
void ocall_debug_print_hex(const void* s, size_t len) {
  printf("DEBUG PRINT (hex): ");
  auto it = (const unsigned char*) s;
  for (unsigned int i = 0; i < len; i++) {
    printf("%02hhx", *(it++));
  }
  printf("\n");
}
