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

#include "channel_server.h"

#include <unistd.h>

#include <cassert>
#include <climits>
#include <cstring>
#include <memory>
#include <thread>

#include "Enclave_u.h"
#include "wolfssl/ssl.h"

namespace ratls {

int ChannelServer::Initialize() {
  // setup socket
  if ((sockfd_ = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
    fprintf(stderr, "ERROR: failed to create the socket\n");
    return -1;
  }

  int enable = 1;
  if (setsockopt(sockfd_, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) ==
      -1) {
    fprintf(stderr, "ERROR: failed to setsockopt\n");
    return -1;
  }

  memset(&addr_, 0, sizeof(addr_));
  addr_.sin_family = AF_INET;
  addr_.sin_port = htons(port_);
  addr_.sin_addr.s_addr = INADDR_ANY; /* from anywhere   */

  /* Bind the server socket to our port */
  if (bind(sockfd_, (struct sockaddr*)&addr_, sizeof(addr_)) == -1) {
    fprintf(stderr, "ERROR: failed to bind\n");
    return -1;
  }
  /* set the port for listen, allow 5 pending connections */
  printf("allowed connection count: %d\n", allowed_connection_);
  if (listen(sockfd_, allowed_connection_) == -1) {
    fprintf(stderr, "ERROR: failed to listen\n");
    return -1;
  }
  printf("server socket initialized\n");
  return 0;
}

int RatlsChannelServer::Initialize() {
  if (initialized_) {
    printf("NormalChannelServer reinit skipped\n");
    return 0;
  }
  // call base implementation
  if (ChannelServer::Initialize() == -1) {
    printf("RAtlsChannelServer failed to setup socket\n");
    return -1;
  }

#ifndef NDEBUG
  printf("enclave id: %ld\n", id_);
#endif  // NDEBUG

  int ret = SGX_SUCCESS;
  auto sgxStatus = enc_wolfSSL_Init(id_, &ret);
  if (sgxStatus != SGX_SUCCESS || ret != SSL_SUCCESS) {
    printf("ecall return: %d, sgx return %d\n", sgxStatus, ret);
    printf("wolfSSL_Init failure\n");
    return -1;
  }

#ifdef SGX_DEBUG
  enc_wolfSSL_Debugging_ON(id_);
  ;
#else
  enc_wolfSSL_Debugging_OFF(id_);
#endif
  WOLFSSL_METHOD* method;
  sgxStatus = enc_wolfTLSv1_2_server_method(id_, &method);
  if (sgxStatus != SGX_SUCCESS || method == NULL) {
    printf("wolfTLSv1_2_server_method failure\n");
    return -1;
  }
  sgxStatus = enc_wolfSSL_CTX_new(id_, &ctx_, method);
  if (sgxStatus != SGX_SUCCESS || ctx_ == NULL) {
    printf("wolfSSL_CTX_new failure");
    return -1;
  }

  // verification is set only if this channel is intended to connect
  //  with enclave clients
  if (enclave_client_ && SetVerify() == -1) {
    printf("set certificate verify failed\n");
    return -1;
  }

  // prepare ra cert and add as extension
  sgxStatus = enc_create_key_and_x509_ecdsa(id_, ctx_);
  assert(sgxStatus == SGX_SUCCESS);
  // prepare ra cert and add as extension
  initialized_ = true;
  return 0;
}

namespace {
void HandleConnection(sgx_enclave_id_t eid, int connd, WOLFSSL_CTX* ctx,
                      Service* svc, std::atomic_int* status) {
  // link socket to wolfssl ctx
  WOLFSSL* ssl{nullptr};
  auto sgxStatus = enc_wolfSSL_new(eid, &ssl, ctx);
  if (sgxStatus != SGX_SUCCESS || ssl == nullptr) {
    printf("wolfSSL_new failure");
    status->store(-1, std::memory_order_relaxed);
    return;
  }

  /* Attach wolfSSL to the socket */
  int ret = SSL_SUCCESS;
  sgxStatus = enc_wolfSSL_set_fd(eid, &ret, ssl, connd);
  if (sgxStatus != SGX_SUCCESS || ret != SSL_SUCCESS) {
    printf("wolfSSL_set_fd failure\n");
    enc_wolfSSL_free(eid, ssl);
    status->store(-1, std::memory_order_relaxed);
    return;
  }

  int svc_ret = svc->Handle(ssl);
  if (enc_wolfSSL_free(eid, ssl) != SGX_SUCCESS) {
    status->store(ret, std::memory_order_relaxed);
  } else {
    status->store(((svc_ret == -1) ? -1 : 0), std::memory_order_relaxed);
  }
}

// a quick solution to handle each connection in a thread. A proper thread pool
// will replace this impl later.
struct ConnectionCtx {
  ConnectionCtx() : connd(-1), status(0) {}
  ~ConnectionCtx() { close_conn(); }
  void close_conn() {
    if (t.joinable()) t.join();
    if (connd != -1) close(connd);
    connd = -1;
  }

  std::thread t;
  int connd;
  std::atomic_int status;  // -1 error, 0 idle, 1 busy
};

struct ConnectionPool {
  ConnectionPool(int num) {
    num_ = num;
    pool_ = std::make_unique<ConnectionCtx[]>(num);
    cur_ = pool_.get();
  }

  ConnectionCtx* pick() {
    assert(pool_);
    printf("the first: %d\n", cur_->status.load(std::memory_order_relaxed));
    while (cur_->status.load(std::memory_order_relaxed) == 1) {
      cur_++;
      printf("finding next");
      if (cur_ >= pool_.get() + num_) {
        cur_ = pool_.get();
      }
      printf("next: %d\n", cur_->status.load(std::memory_order_relaxed));
    }
    return cur_;
  }

  void cleanup() {
    if (pool_)
      for (int i = 0; i < num_; i++) pool_.get()[i].close_conn();
  }

 private:
  int num_;
  ConnectionCtx* cur_;
  std::unique_ptr<ConnectionCtx[]> pool_;
};

}  // anonymous namespace

bool RatlsChannelServer::Serve() {
  if (serving_ || (svc_ == nullptr) || (!initialized_)) return false;
  serving_ = true;

  ConnectionPool connections{allowed_connection_};

  while (1) {
    /* Accept client connections */
    int connd;
    struct sockaddr_in client_addr;
    socklen_t size = sizeof(client_addr);
    printf("Waiting for a connction...\n");
    if ((connd = accept(sockfd_, (struct sockaddr*)&client_addr, &size)) ==
        -1) {
      perror("ERROR: failed to accept the connection");
      return -1;
    }
    printf("accepted a connection\n");
    ConnectionCtx* connd_ctx = connections.pick();
    if (connd_ctx->status.load(std::memory_order_relaxed) == -1) break;
    connd_ctx->close_conn();
    connd_ctx->connd = connd;
    connd_ctx->status.store(1, std::memory_order_relaxed);
    connd_ctx->t = std::thread{HandleConnection,    id_, connd, ctx_, svc_,
                               &(connd_ctx->status)};
  }

  connections.cleanup();
  return true;
}

int RatlsChannelServer::Close() {
  int ret = 0;
  if (ctx_ && (enc_wolfSSL_CTX_free(id_, ctx_) != SGX_SUCCESS)) {
    printf("enc_wolfSSL_CTX_free failure");
    ret = -1;
  }
  ctx_ = nullptr;
  int tmp = SGX_SUCCESS;
  if (enc_wolfSSL_Cleanup(id_, &tmp) != SGX_SUCCESS || tmp != SSL_SUCCESS) {
    printf("enc_wolfSSL_Cleanup failure");
    ret = -1;
  }
  initialized_ = false;
  serving_ = false;
  return ret;
}

// needs to switch to enclave driven to ensure verify is set
int RatlsChannelServer::SetVerify() {
  enc_wolfSSL_CTX_set_ratls_verify(id_, ctx_);
  return 0;
}

}  // namespace ratls
