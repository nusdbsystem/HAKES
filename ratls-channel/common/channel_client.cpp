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

#include "channel_client.h"

#include <unistd.h>

#include <cassert>
#include <climits>
#include <cstring>

#include "challenger_wolfssl.h"

#ifdef SGXCLIENT
#include "Enclave_t.h"
#include "sgx_error.h"
#include "tattester_wolfssl.h"

namespace {
inline void printf(const char* msg) { ocall_print_string(msg); }
}  // anonymous namespace
#else  // SGXCLIENT
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#endif  // SGXCLIENT

namespace ratls {

int RatlsChannelClient::Initialize() {
  if (initialized_) return 0;
    // setup ssl
#ifdef SGX_DEBUG
  wolfSSL_Debugging_ON();
#else
  wolfSSL_Debugging_OFF();
#endif
  if (wolfSSL_Init() != SSL_SUCCESS) {
    printf("ERROR: failed to init WOLFSSL\n");
    return -1;
  }
  auto method = wolfTLSv1_2_client_method();
  if (!method) {
    printf("wolfTLSv1_2_client_method failure\n");
    return -1;
  }
  ctx_ = wolfSSL_CTX_new(method);
  if (!ctx_) {
    printf("wolfssl_CTX_new failure\n");
    return -1;
  }
  if (SetVerify() == -1) {
    printf("set certificate verify failed\n");
    return -1;
  }

#ifdef SGXCLIENT
  // preapre ra cert and add as extension
  wolfssl_create_key_and_x509_ctx_ecdsa(ctx_);
  // preapre ra cert and add as extension
#endif  // SGXCLIENT

  initialized_ = true;
  return 0;
}

int RatlsChannelClient::Connect() {
  ssl_ = wolfSSL_new(ctx_);
  if (!ssl_) {
    printf("wolfssl_new failure\n");
    return -1;
  }
#ifdef SGXCLIENT
  if (ocall_get_socket(&sockfd_, server_addr_.c_str(), server_addr_.size(),
                       port_) != SGX_SUCCESS)
    return -1;
#else  // SGXCLIENT
  // connect
  if ((sockfd_ = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
    fprintf(stderr, "ERROR: failed to create the socket\n");
    return -1;
  }

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port_);
#ifndef NDEBUG
  printf("port number: %d\n", port_);
  printf("server address: %s\n", server_addr_.c_str());
#endif  // NDEBUG
  if (inet_pton(AF_INET, server_addr_.c_str(), &addr.sin_addr) != 1) {
    perror("ERROR: invalid address");
    return -1;
  }

  if (connect(sockfd_, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
    perror("ERROR: failed to connect");
    return -1;
  }
#endif  // SGXCLIENT

  if (wolfSSL_set_fd(ssl_, sockfd_) != SSL_SUCCESS) {
    printf("wolfSSL_set_fd failure\n");
    return -1;
  }

  if (wolfSSL_connect(ssl_) != SSL_SUCCESS) {
    printf("failed to connect to server with ssl\n");
    return -1;
  }
  connected_ = true;
  return 0;
}

namespace {
int channel_close(int sockfd) {
#ifdef SGXCLIENT
  return (ocall_close_socket(sockfd) == SGX_SUCCESS) ? 0 : -1;
#else   // SGXCLIENT
  return close(sockfd);
#endif  // SGXCLIENT
}
}  // anonymous namespace

int RatlsChannelClient::CloseConnection() {
  int ret{0};
  wolfSSL_free(ssl_);
  ssl_ = nullptr;
  if (connected_) ret = channel_close(sockfd_);
  connected_ = false;
  return ret;
}

int RatlsChannelClient::Close() {
  int ret{0};
  ret = CloseConnection();
  wolfSSL_CTX_free(ctx_);
  ctx_ = nullptr;
  wolfSSL_Cleanup();
  initialized_ = false;
  return ret;
}

int RatlsChannelClient::Read(std::string* output) {
  char buf[RCVBUFSIZE]{0};
  int read_len = wolfSSL_read(ssl_, buf, sizeof(buf));
  if (read_len <= 0) {
    printf("Server failed to read\n");
    return -1;
  }
  *output = std::string(buf, read_len);
  return 0;
}

int RatlsChannelClient::Send(const std::string& msg) {
  assert(msg.size() < INT_MAX);
  if (wolfSSL_write(ssl_, msg.data(), static_cast<int>(msg.size())) !=
      static_cast<int>(msg.size())) {
    printf("Server failed to send\n");
    return -1;
  }
  return 0;
}

int RatlsChannelClient::SetVerify() {
  wolfSSL_CTX_set_verify(ctx_, SSL_VERIFY_PEER, cert_verify_callback);
  return 0;
}

}  // namespace ratls
