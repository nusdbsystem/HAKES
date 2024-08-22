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

#include <cassert>
#include <iostream>
#include <string>

#include "message/keyservice_user.h"
#include "ratls-channel/common/channel_client.h"
#include "utils/fileutil.h"
#include "utils/json.h"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: request_json, config_json");
    exit(1);
  }
  std::string request_json_path{argv[1]};
  std::string config_json_path{argv[2]};

  std::string request_str;
  if (hakes::ReadFileToString(request_json_path, &request_str) == -1) {
    printf("failed to load request json file\n");
    return 1;
  }
  auto request_json = json::JSON::Load(request_str);
  //  set up channel with
  ratls::CheckContext ctx;
  std::string config_json_str;
  if (hakes::ReadFileToString(config_json_path, &config_json_str) == -1) {
    printf("failed to load config json file\n");
    return 1;
  }
  auto config_json = json::JSON::Load(config_json_str);
  assert(config_json.hasKey("server_address"));
  assert(config_json.hasKey("port"));
  auto port_num = config_json["port"].ToInt();
  if (port_num > UINT16_MAX) {
    printf("invalid port number\n");
    return 1;
  }
  ratls::RatlsChannelClient channel{static_cast<uint16_t>(port_num),
                             config_json["server_address"].ToString(), ctx};
  auto status = channel.Initialize();
  if (status == 0) status = channel.Connect();
  if (status == 0) status = channel.Send(request_str);
  std::string reply_msg;
  if (status == 0) status = channel.Read(&reply_msg);
  if (status == 0) {
    auto reply = hakes::DecodeKeyServiceReply(reply_msg);
    std::cout << (reply.success_ ? "Success" : "Failed") << ": "
              << (reply.reply_.empty() ? "no additional info" : reply.reply_)
              << "\n";
  } else
    std::cout << "Program encountered errors\n";
  return 0;
}
