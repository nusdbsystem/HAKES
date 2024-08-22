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

#ifndef HAKES_RATLSCHANNEL_COMMON_CHANNELCLIENT_H_
#define HAKES_RATLSCHANNEL_COMMON_CHANNELCLIENT_H_

#include "channel.h"
#include "wolfssl/ssl.h"

namespace ratls {

class RatlsChannelClient : public Channel {
 public:
  RatlsChannelClient(uint16_t port, const std::string& server_addr,
                     const CheckContext& check_ctx)
      : initialized_(false),
        connected_(false),
        sockfd_(-1),
        checker_(check_ctx),
        ctx_(nullptr),
        port_(port),
        server_addr_(server_addr),
        ssl_(nullptr) {}

  ~RatlsChannelClient() { Close(); }

  RatlsChannelClient(const RatlsChannelClient&) = delete;
  RatlsChannelClient& operator=(const RatlsChannelClient&) = delete;
  RatlsChannelClient(RatlsChannelClient&&) = delete;
  RatlsChannelClient& operator=(RatlsChannelClient&&) = delete;

  int Initialize();

  int Connect();

  int CloseConnection() override;

  int Close();

  int Read(std::string* output) override;

  int Send(const std::string& msg) override;

  inline bool IsInitialized() const { return initialized_; }
  inline bool IsConnected() const { return connected_; }

 private:
  int SetVerify();

  bool initialized_;
  bool connected_;
  int sockfd_;
  const CheckContext& checker_;
  WOLFSSL_CTX* ctx_;

  uint16_t port_;
  const std::string server_addr_;
  WOLFSSL* ssl_;
};

}  // namespace ratls

#endif  // HAKES_RATLSCHANNEL_COMMON_CHANNELCLIENT_H_
