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

#ifndef HAKES_RATLSCHANNEL_CHANNELSERVER_H_
#define HAKES_RATLSCHANNEL_CHANNELSERVER_H_

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sgx_eid.h>
#include <sys/socket.h>
#include <wolfssl/ssl.h>

#include <atomic>
#include <vector>

#include "ratls-channel/common/channel.h"

namespace ratls {

class Service {
 public:
  virtual ~Service() = default;
  virtual int Handle(WOLFSSL* ssl) = 0;
};

class ChannelServer {
 public:
  ChannelServer(uint16_t port, const CheckContext& check_ctx,
                int allowed_connection)
      : initialized_(false),
        serving_(false),
        checker_(check_ctx),
        svc_(nullptr),
        sockfd_(-1),
        port_(port),
        allowed_connection_(allowed_connection) {}
  virtual ~ChannelServer() = default;

  virtual int Initialize();

  inline bool RegisterService(Service* svc) {
    if (svc_) return false;
    svc_ = svc;
    return true;
  }

  virtual bool Serve() = 0;
  virtual int Close() = 0;

  inline bool IsInitialized() const { return initialized_; }

 protected:
  bool initialized_;
  bool serving_;
  const CheckContext& checker_;
  Service* svc_;

  int sockfd_;
  uint16_t port_;
  sockaddr_in addr_;
  int allowed_connection_;
};

class RatlsChannelServer : public ChannelServer {
 public:
  RatlsChannelServer(sgx_enclave_id_t id, uint16_t port,
                     const CheckContext& check_ctx, bool enclave_client,
                     int allowed_connection)
      : ChannelServer(port, check_ctx, allowed_connection),
        id_(id),
        enclave_client_(enclave_client),
        ctx_(nullptr),
        error_flag_(false) {}
  ~RatlsChannelServer() { Close(); }

  RatlsChannelServer(const RatlsChannelServer&) = delete;
  RatlsChannelServer& operator=(const RatlsChannelServer&) = delete;
  RatlsChannelServer(RatlsChannelServer&&) = delete;
  RatlsChannelServer& operator=(RatlsChannelServer&&) = delete;

  int Initialize();
  bool Serve() override;
  int Close() override;

 private:
  int SetVerify();

  sgx_enclave_id_t id_;
  bool enclave_client_;
  WOLFSSL_CTX* ctx_;
  std::atomic_bool error_flag_;
};

}  // namespace ratls

#endif  // HAKES_RATLSCHANNEL_CHANNELSERVER_H_
