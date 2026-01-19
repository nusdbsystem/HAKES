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

#ifndef HAKES_UTILS_HTTP_H_
#define HAKES_UTILS_HTTP_H_

#include <curl/curl.h>

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

// curl handles are expected to be used in a single thread. So the atomic flag
// is used to ensure that only one thread can use the handle at a time.

namespace hakes {

namespace {
std::once_flag curl_init_flag;
}  // anonymous namespace

inline void setup_curl() {
  std::call_once(curl_init_flag, []() { curl_global_init(CURL_GLOBAL_ALL); });
}

inline void cleanup_curl() { curl_global_cleanup(); }

class HttpClient {
 public:
  HttpClient() {
    // Initialize curl
    curl = curl_easy_init();
  }

  virtual ~HttpClient();

  inline bool claim() { return !in_use_.test_and_set(); }
  inline void release() { in_use_.clear(); }

  virtual std::string get(const std::string& url);

  virtual std::string post(const std::string& url, const std::string& data, const std::string& additional_header = "");

 private:
  std::atomic_flag in_use_;
  CURL* curl;
};

class MultiHttpClient {
 public:
  MultiHttpClient() {
    // Initialize curl
    mch_ = curl_multi_init();
  }

  virtual ~MultiHttpClient();

  inline bool claim() { return !in_use_.test_and_set(); }
  inline void release() { in_use_.clear(); }

  virtual std::string get(const std::string& url);

  // the urls should be unique
  // return success or the first invalid message
  virtual std::vector<std::string> get(const std::vector<std::string>& urls);

  virtual std::string post(const std::string& url, const std::string& data);

  // the urls should be unique
  // return success or the first invalid message
  virtual std::vector<std::string> post(
      const std::vector<std::string>& urls,
      const std::vector<std::string>& url_data);

 private:
  std::atomic_flag in_use_;
  CURLM* mch_;
  std::unordered_map<std::string, CURL*> chs_;
};

}  // namespace hakes

#endif  // HAKES_UTILS_HTTP_H_
