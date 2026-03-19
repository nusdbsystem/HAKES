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

#include "server.h"

#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>

#define NDEBUG 1
namespace hakes {

namespace {

bool check_error(int ret) {
  if (!ret) return false;
  fprintf(stderr, "libuv error: %s\n", uv_strerror(ret));
  return true;
}

void on_close(uv_handle_t* handle) {
#ifndef NDEBUG
  printf("on close\n");
#endif  // NDEBUG
  client_t* client = (client_t*)handle->data;
  client->resbuf.base = nullptr;
  client->resbuf.len = 0;
  delete client;
  client = nullptr;
}

void after_write(uv_write_t* req, int status) {
  check_error(status);
  uv_close((uv_handle_t*)req->handle, on_close);
}

inline void flush_ready() {
  printf("%s\n", server_ready);
  fflush(stdout);
}

void after_work(uv_work_t* req, int status) {
  if (check_error(status)) return;
  client_t* client = (client_t*)req->data;
  uv_write(&client->write_req, (uv_stream_t*)&client->handle, &client->resbuf,
           1, after_write);
}
// concurrent runtime functions

int on_message_begin(parser_t* _) {
  (void)_;
#ifndef NDEBUG
  printf("\n***MESSAGE BEGIN***\n\n");
#endif  // NDEBUG
  return 0;
}

int on_headers_complete(parser_t* _) {
  (void)_;
#ifndef NDEBUG
  printf("\n***HEADERS COMPLETE***\n\n");
#endif  // NDEBUG
  return 0;
}

int on_message_complete(parser_t* parser) {
#ifndef NDEBUG
  printf("\n***MESSAGE COMPLETE***\n\n");
#endif  // NDEBUG
  client_t* client = (client_t*)parser->data;
  uv_queue_work(client->handle.loop, &client->work, Server::OnWork,
                // Server::AfterWork);
                after_work);
  return 0;
}

int on_url(parser_t* parser, const char* at, size_t length) {
#ifndef NDEBUG
  // printf("Url (%d): %.*s\n", (int)length, (int)length, at);
#endif  // NDEBUG
  client_t* client = (client_t*)parser->data;
  client->req.url = std::string(at, length);
  return 0;
}

int on_header_field(parser_t* parser, const char* at, size_t length) {
#ifndef NDEBUG
  // printf("Header field: %.*s\n", (int)length, at);
#endif  // NDEBUG
  if (strncmp(at, "Content-Length",
              std::max(length, strlen("Content-Length"))) == 0) {
    // printf("Header field: %.*s, set next content length\n", (int)length, at);
    client_t* client = (client_t*)parser->data;
    client->next_header_value_is_content_len = true;
  }
  return 0;
}

int on_header_value(parser_t* parser, const char* at, size_t /*length*/) {
#ifndef NDEBUG
  // printf("Header value: %.*s\n", (int)length, at);
#endif  // NDEBUG
  client_t* client = (client_t*)parser->data;
  if (client->next_header_value_is_content_len) {
    client->req.content_len = strtoull(at, NULL, 10);
    client->req.body.reserve(client->req.content_len);
    // printf("reserve body size: %ld\n", client->req.content_len);
    client->next_header_value_is_content_len = false;
  }
  return 0;
}

int on_body(parser_t* parser, const char* at, size_t length) {
  // (void)_;
  client_t* client = (client_t*)parser->data;
  client->req.body.append(at, length);
#ifndef NDEBUG
  // printf("Body: %.*s\n", (int)length, at);
#endif  // NDEBUG
  return 0;
}

// all the callbacks implementation are from http-parser/contrib/parsertrace.c
//  to print something demonstrating the processing of each phase.
// on_message_complete is rewritten to send back response.
void setup_http_parser_settings(parser_settings_t* settings) {
  llhttp_settings_init(settings);
  settings->on_message_begin = on_message_begin;
  settings->on_url = on_url;
  settings->on_header_field = on_header_field;
  settings->on_header_value = on_header_value;
  settings->on_headers_complete = on_headers_complete;
  settings->on_body = on_body;
  settings->on_message_complete = on_message_complete;
}

void on_alloc(uv_handle_t* /*handle*/, size_t suggested_size, uv_buf_t* buf) {
#ifndef NDEBUG
  // printf("on alloc start: %ld\n",
  // std::chrono::system_clock::now().time_since_epoch().count()/1000);
#endif  // NDEBUG
  *buf = uv_buf_init((char*)malloc(suggested_size), suggested_size);
}

void on_read(uv_stream_t* stream, ssize_t nread, const uv_buf_t* buf) {
#ifndef NDEBUG
  // printf("on read\n");
  // auto on_read_start =
  // std::chrono::system_clock::now().time_since_epoch().count()/1000;
  // printf("on read (size %ld) time: %ld\n", nread, on_read_start);
  // std::cout << std::string(buf->base, nread) << std::endl;
#endif  // NDEBUG
  /*do something*/
  if (nread >= 0) {
#ifndef NDEBUG
    // printf("Read:\n%.*s\n", (int) nread, buf->base);
#endif  // NDEBUG
    /*parse http*/
    client_t* client = (client_t*)uv_handle_get_data((uv_handle_t*)stream);
    parser_t* parser = &client->parser;

    if (!client->is_http_req) {
      auto ret = llhttp_execute(parser, buf->base, nread);
      if (ret != HPE_OK) {
        fprintf(stderr, "Parse error: %s %s\n", llhttp_errno_name(ret),
                parser->reason);
        printf("Not a http request, no response sent\n");
        client->server->Stop();
      }
    } else {
      // continuous reading the request. append to the body.
      client->req.body.append(buf->base, nread);
    }

  } else {
    // error
    if (nread != UV_EOF) {
      printf("Read error: %ld\n", nread);
    }
    uv_close((uv_handle_t*)stream, on_close);
  }

  free(buf->base);
}
}  // anonymous namespace

std::string build_response(hakes::response_code_t code,
                           const std::string& msg) {
  auto build = [](const std::string&& code_msg,
                  const std::string&& msg) -> std::string {
    return "HTTP/1.1 " + code_msg + "\r\n" +
           "Content-Type: application/json\r\n" +
           "Content-Length: " + std::to_string(msg.size()) + "\r\n" + "\r\n" +
           std::move(msg);
  };

  switch (code) {
    case hakes::OK:
      return build("200 OK", std::move(msg));
    case hakes::FORBIDDEN:
      return build("403 Forbidden", std::move(msg));
    case hakes::NOT_FOUND:
      return build("404 Not Found", std::move(msg));
    default:
      return build("500 Internal Server Error", std::move(msg));
  }
}

Server::~Server() {
  if (loop_) {
    uv_print_active_handles(loop_, stderr);
    uv_walk(
        loop_,
        [](uv_handle_t* handle, void* /*arg*/) {
          // if (uv_is_closing(handle)) return;
          uv_close(handle, [](uv_handle_t* handle) {
            printf("closing handle\n");
            if (handle != nullptr) {
              delete handle;
              handle = nullptr;
            }
          });
        },
        nullptr);
    uv_run(loop_, UV_RUN_DEFAULT);
    uv_loop_delete(loop_);
  }
};

bool Server::Init() {
  setup_http_parser_settings(&settings_);
  loop_ = (uv_loop_t*)malloc(sizeof(uv_loop_t));
  uv_loop_init(loop_);
  uv_tcp_init(loop_, &server_);
  struct sockaddr_in address;
  uv_ip4_addr("0.0.0.0", port_, &address);
  server_.data = this;

  int ret = uv_tcp_bind(&server_, (const struct sockaddr*)&address, 0);
  if (check_error(ret)) return false;

  return service_->Init() == hakes::OK;
}

void Server::OnWork(uv_work_t* req) {
  client_t* client = (client_t*)req->data;
  hakes::response_code_t code;
  std::string msg;
#ifndef NDEBUG
  // printf("\nRequest body before processing:\n%s\n",
  // client->req.body.c_str());
#endif  // NDEBUG
  code =
      client->server->service_->OnWork(client->req.url, client->req.body, &msg);
  client->response_scratch = build_response(code, std::move(msg));
  client->resbuf =
      uv_buf_init(const_cast<char*>(client->response_scratch.c_str()),
                  client->response_scratch.size());
}

void Server::Handle(uv_stream_t* server_handle, int status) {
  Server* server = (Server*)server_handle->data;
  assert(server_handle == (uv_stream_t*)&(server->server_));
#ifndef NDEBUG
  printf("\non_connection\n");
#endif  // NDEBUG

  if (check_error(status)) return;

  // allocate http parser and a handle for each connection
  client_t* client = new client_t;
  client->server = server;

  // init
  llhttp_init(&client->parser, HTTP_BOTH, &server->settings_);

  auto ret = uv_tcp_init(server_handle->loop, &client->handle);
  // let the data pointer of handle to point to the client struct,
  //  so we can access http parser.
  uv_handle_set_data((uv_handle_t*)&client->handle, client);
  // let the data pointer of parser to point to the client struct,
  //  so we can access handle.
  client->parser.data = client;
  uv_req_set_data((uv_req_t*)&client->work, client);

  check_error(ret);
  ret = uv_accept(server_handle, (uv_stream_t*)&client->handle);
  if (check_error(ret)) {
    uv_close((uv_handle_t*)&client->handle, on_close);
  } else {
    ret = uv_read_start((uv_stream_t*)&client->handle, on_alloc, on_read);
    check_error(ret);
  }
}

bool Server::Start() {
  int ret = uv_listen((uv_stream_t*)&server_, 128, Server::Handle);
  if (check_error(ret)) return false;
  printf("server is listening on port %ld\n", port_);

  flush_ready();
  uv_run(loop_, UV_RUN_DEFAULT);
  return true;
}

void Server::Stop() {
  auto tcp_close_ret = uv_tcp_close_reset(
      &server_, [](uv_handle_t* /*handle*/) { printf("server is closed\n"); });
  assert(tcp_close_ret == 0);
  uv_stop(loop_);
  printf("server is stopped\n");
}

}  // namespace hakes
