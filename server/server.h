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

#ifndef HAKES_SERVER_SERVER_H_
#define HAKES_SERVER_SERVER_H_

#include "llhttp.h"
#include "service.h"
#include "uv.h"

namespace hakes {
typedef llhttp_t parser_t;
typedef llhttp_settings_t parser_settings_t;

constexpr char server_ready[] = "server-init-done";

struct HttpRequest {
  uint64_t content_len;
  std::string url;
  std::string body;
};

std::string build_response(response_code_t code, const std::string& msg);

class Server {
 public:
  Server(long port, Service* svc) : port_(port), service_(svc) {
    int uv_threadpool_size = 1;

    char* threadpool_size_str = std::getenv("UV_THREADPOOL_SIZE");
    if (threadpool_size_str == nullptr) {
      printf("UV_THREADPOOL_SIZE not set. worker pool size defaulting to %d\n",
             uv_threadpool_size);
    } else {
      uv_threadpool_size = std::atoi(threadpool_size_str);
      printf("worker pool size: %d\n", uv_threadpool_size);
    }
  };
  virtual ~Server();

  // delete copy constructor and assignment operator
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;
  // delete move constructor and assignment operator
  Server(Server&&) = delete;
  Server& operator=(Server&&) = delete;

  static void Handle(uv_stream_t* server_handle, int status);
  static void OnWork(uv_work_t* req);

  virtual bool Init();
  virtual bool Start();
  virtual void Stop();

  long port_;
  uv_tcp_t server_;
  uv_loop_t* loop_;
  parser_settings_t settings_;
  Service* service_;
};

struct client_t {
  uv_tcp_t handle;
  parser_t parser;
  uv_write_t write_req;
  std::string response_scratch;  // buffer for the resbuf
  uv_buf_t resbuf{nullptr, 0};
  bool is_http_req = false;
  bool next_header_value_is_content_len = false;
  HttpRequest req;
  uv_work_t work;  // at a time there is only one work scheduled.
  Server* server;
};

}  // namespace hakes

#endif  // HAKES_SERVER_SERVER_H_