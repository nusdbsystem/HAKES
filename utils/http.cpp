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

#include "http.h"

#include <chrono>
#include <iostream>

namespace hakes {
namespace {
size_t writeCallback(char* data, size_t size, size_t nmemb,
                     std::string* buffer) {
  // Append the data to the buffer
  buffer->append(data, size * nmemb);
  return size * nmemb;
}
}  // anonymous namespace

HttpClient::~HttpClient() {
  std::cout << "HttpClient destructor called\n";
  // Cleanup curl
  curl_easy_cleanup(curl);
}

std::string HttpClient::get(const std::string& url) {
  if (curl) {

    // Set the URL to send the GET request to
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

    struct curl_slist* slist1 = NULL;
    slist1 = curl_slist_append(slist1, "Expect:");
    slist1 = curl_slist_append(slist1, "Content-Type: application/json");
    slist1 = curl_slist_append(slist1, "Connection: keep-alive");
    slist1 = curl_slist_append(slist1, "Accept: */*");
    slist1 = curl_slist_append(slist1, "User-Agent: libcurl/7.71.1");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, slist1);

    // Store response in a string
    std::string response;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    // Perform the GET request
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
      std::cerr << "Failed to perform GET request: " << curl_easy_strerror(res)
                << "\n";
    }
    curl_easy_reset(curl);
    curl_slist_free_all(slist1);

    return response;
  }

  return "";
}

std::string HttpClient::post(const std::string& url, const std::string& data,
                             const std::string& additional_header) {
  if (curl) {

    std::cout << "post url: " << url << "\n";
    // Set the URL to send the GET request to
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

    // set headers
    struct curl_slist* slist1 = NULL;
    slist1 = curl_slist_append(slist1, "Expect:");
    slist1 = curl_slist_append(slist1, "Content-Type: application/json");
    slist1 = curl_slist_append(slist1, "Connection: keep-alive");
    slist1 = curl_slist_append(slist1, "Accept: */*");
    slist1 = curl_slist_append(slist1, "User-Agent: libcurl/7.71.1");
    if (!additional_header.empty()) {
      slist1 = curl_slist_append(slist1, additional_header.c_str());
    }
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, slist1);

    // set the post data
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());

    // Store response in a string
    std::string response;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    // Perform the GET request
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
      std::cerr << "Failed to perform POST request: " << curl_easy_strerror(res)
                << "\n";
    }
    curl_easy_reset(curl);
    curl_slist_free_all(slist1);

    return response;
  }

  return "";
}

MultiHttpClient::~MultiHttpClient() {
  std::cout << "MultiHttpClient destructor called\n";
  // Clean up curl
  for (auto& ch : chs_) {
    curl_easy_cleanup(ch.second);
  }
  curl_multi_cleanup(mch_);
}

std::string MultiHttpClient::get(const std::string& url) {
  if (url.empty()) {
    std::cerr << "Empty input\n";
    return "";
  }

  // Set the URL to send the GET request to
  CURL* ch;
  if (chs_.find(url) != chs_.end()) {
    ch = chs_[url];
  } else {
    ch = curl_easy_init();
    chs_[url] = ch;
  }
  curl_easy_setopt(ch, CURLOPT_URL, url.c_str());

  // set headers
  struct curl_slist* slist1 = NULL;
  slist1 = curl_slist_append(slist1, "Expect:");
  slist1 = curl_slist_append(slist1, "Content-Type: application/json");
  slist1 = curl_slist_append(slist1, "Connection: keep-alive");
  slist1 = curl_slist_append(slist1, "Accept: */*");
  slist1 = curl_slist_append(slist1, "User-Agent: libcurl/7.71.1");
  curl_easy_setopt(ch, CURLOPT_HTTPHEADER, slist1);

  curl_easy_setopt(ch, CURLOPT_WRITEFUNCTION, writeCallback);
  std::string response;
  curl_easy_setopt(ch, CURLOPT_WRITEDATA, &response);

  // Perform the GET request
  CURLcode res = curl_easy_perform(ch);
  if (res != CURLE_OK) {
    std::cerr << "Failed to perform GET request: " << curl_easy_strerror(res)
              << "\n";
  }

  // Clean up
  curl_easy_reset(ch);
  curl_slist_free_all(slist1);

  return response;
}

std::vector<std::string> MultiHttpClient::get(
    const std::vector<std::string>& urls) {
  if (urls.empty()) {
    std::cerr << "Empty input\n";
    return {};
  }

  // set headers
  struct curl_slist* slist1 = NULL;
  slist1 = curl_slist_append(slist1, "Expect:");
  slist1 = curl_slist_append(slist1, "Content-Type: application/json");
  slist1 = curl_slist_append(slist1, "Connection: keep-alive");
  slist1 = curl_slist_append(slist1, "Accept: */*");
  slist1 = curl_slist_append(slist1, "User-Agent: libcurl/7.71.1");

  // Set the URLs to send the GET requests to
  std::vector<std::string> responses{urls.size()};

  int idx = 0;
  for (const auto& url : urls) {
    CURL* ch;
    if (chs_.find(url) != chs_.end()) {
      ch = chs_[url];
    } else {
      ch = curl_easy_init();
      chs_[url] = ch;
    }
    curl_easy_setopt(ch, CURLOPT_URL, url.c_str());

    curl_easy_setopt(ch, CURLOPT_HTTPHEADER, slist1);

    curl_easy_setopt(ch, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(ch, CURLOPT_WRITEDATA, &responses[idx++]);
    curl_multi_add_handle(mch_, ch);
  }

  // Perform the GET requests
  int running_handles = 1;
  // Wait for the requests to complete
  while (running_handles) {
    auto res = curl_multi_perform(mch_, &running_handles);
    if (running_handles) {
      // res = curl_multi_poll(mch_, nullptr, 0, 1000, nullptr);
      res = curl_multi_wait(mch_, nullptr, 0, 1000, nullptr);
    }
    if (res != CURLM_OK) {
      std::cerr << "Failed to perform GET requests: "
                << curl_multi_strerror(res) << "\n";
    }
  }

  // Clean up
  curl_slist_free_all(slist1);
  for (const auto& url : urls) {
    curl_multi_remove_handle(mch_, chs_[url]);
    curl_easy_reset(chs_[url]);
  }

  return responses;
}

std::string MultiHttpClient::post(const std::string& url,
                                  const std::string& data) {
  auto start = std::chrono::high_resolution_clock::now();

  if (url.empty() || data.empty()) {
    std::cerr << "Empty input\n";
    return "";
  }

  // Set the URL to send the GET request to
  CURL* ch;
  if (chs_.find(url) != chs_.end()) {
    ch = chs_[url];
  } else {
    ch = curl_easy_init();
    chs_[url] = ch;
  }
  curl_easy_setopt(ch, CURLOPT_URL, url.c_str());
  // set headers
  struct curl_slist* slist1 = NULL;
  slist1 = curl_slist_append(slist1, "Expect:");
  slist1 = curl_slist_append(slist1, "Content-Type: application/json");
  slist1 = curl_slist_append(slist1, "Connection: keep-alive");
  slist1 = curl_slist_append(slist1, "Accept: */*");
  slist1 = curl_slist_append(slist1, "User-Agent: libcurl/7.71.1");
  curl_easy_setopt(ch, CURLOPT_HTTPHEADER, slist1);

  curl_easy_setopt(ch, CURLOPT_POSTFIELDSIZE, data.size());
  curl_easy_setopt(ch, CURLOPT_POSTFIELDS, data.c_str());
  curl_easy_setopt(ch, CURLOPT_WRITEFUNCTION, writeCallback);
  std::string response;
  curl_easy_setopt(ch, CURLOPT_WRITEDATA, &response);

  auto prepare_handle_end = std::chrono::high_resolution_clock::now();

  // Perform the POST request
  CURLcode res = curl_easy_perform(ch);
  if (res != CURLE_OK) {
    std::cerr << "Failed to perform POST request: " << curl_easy_strerror(res)
              << "\n";
  }

  auto perform_end = std::chrono::high_resolution_clock::now();

  // Clean up
  curl_easy_reset(ch);
  curl_slist_free_all(slist1);
  auto cleanup_end = std::chrono::high_resolution_clock::now();

  printf("prepare_handle: %ld\n",
         std::chrono::duration_cast<std::chrono::microseconds>(
             prepare_handle_end - start)
             .count());
  printf("perform: %ld\n",
         std::chrono::duration_cast<std::chrono::microseconds>(
             perform_end - prepare_handle_end)
             .count());
  printf("cleanup: %ld\n",
         std::chrono::duration_cast<std::chrono::microseconds>(cleanup_end -
                                                               perform_end)
             .count());

  return response;
}

std::vector<std::string> MultiHttpClient::post(
    const std::vector<std::string>& urls,
    const std::vector<std::string>& url_data) {
  if (url_data.empty()) {
    std::cerr << "Empty input\n";
    return {};
  }
  if (urls.size() != url_data.size()) {
    std::cerr << "URLs and data sizes do not match\n";
    return {};
  }

  // Set the URLs to send the GET requests to
  std::vector<std::string> responses{url_data.size()};

  // set headers
  struct curl_slist* slist1 = NULL;
  slist1 = curl_slist_append(slist1, "Expect:");
  slist1 = curl_slist_append(slist1, "Content-Type: application/json");
  slist1 = curl_slist_append(slist1, "Connection: keep-alive");
  slist1 = curl_slist_append(slist1, "Accept: */*");
  slist1 = curl_slist_append(slist1, "User-Agent: libcurl/7.71.1");

  int idx = 0;
  for (int i = 0; i < urls.size(); i++) {
    auto& url = urls[i];
    auto& data = url_data[i];
    CURL* ch;
    if (chs_.find(url) != chs_.end()) {
      ch = chs_[url];
    } else {
      ch = curl_easy_init();
      chs_[url] = ch;
    }
    curl_easy_setopt(ch, CURLOPT_URL, url.c_str());
    curl_easy_setopt(ch, CURLOPT_HTTPHEADER, slist1);

    curl_easy_setopt(ch, CURLOPT_HTTPHEADER, slist1);

    curl_easy_setopt(ch, CURLOPT_POSTFIELDS, data.c_str());
    curl_easy_setopt(ch, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(ch, CURLOPT_WRITEDATA, &responses[idx++]);
    curl_multi_add_handle(mch_, ch);
  }

  // Perform the POST requests
  int running_handles;
  CURLMcode res = curl_multi_perform(mch_, &running_handles);
  if (res != CURLM_OK) {
    std::cerr << "Failed to perform POST requests: " << curl_multi_strerror(res)
              << "\n";
  }

  // Wait for the requests to complete
  while (running_handles) {
    res = curl_multi_perform(mch_, &running_handles);
    if (res != CURLM_OK) {
      std::cerr << "Failed to perform POST requests: "
                << curl_multi_strerror(res) << "\n";
    }
  }

  // Clean up
  curl_slist_free_all(slist1);
  for (int i = 0; i < urls.size(); i++) {
    curl_multi_remove_handle(mch_, chs_[urls[i]]);
    curl_easy_reset(chs_[urls[i]]);
  }

  return responses;
}

}  // namespace hakes