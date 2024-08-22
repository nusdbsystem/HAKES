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

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

#include "Enclave_u.h"  // include this helps to solve the undefined reference error for ocalls. Enclave_u.c is compiled as c.

int ocall_get_socket(const char* server_addr, size_t addr_len, uint16_t port) {
  // setup socket
  int sockfd;
  if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
    fprintf(stderr, "ERROR: failed to create the socket\n");
    return -1;
  }

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(13571);

  // compensate for the c string passed from enclave
  // otherwise, the server_addr does not terminate proper
  char buf[addr_len + 1];
  memset(buf, 0, addr_len + 1);
  memcpy(buf, server_addr, addr_len);

  if (inet_pton(AF_INET, buf, &addr.sin_addr) != 1) {
    printf("port number: %d\n", port);
    printf("server address: %s..\n", server_addr);
    fprintf(stderr, "ERROR: invalid address");
    return -1;
  }

  if (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
    fprintf(stderr, "ERROR: failed to connect\n");
    return -1;
  }
  return sockfd;
}

void ocall_close_socket(int sockfd) { close(sockfd); }
