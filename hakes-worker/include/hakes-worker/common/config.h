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

#ifndef HAKES_HAKESWORKER_COMMON_CONFIG_H_
#define HAKES_HAKESWORKER_COMMON_CONFIG_H_

#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// overall hakes service configuration

namespace hakes_worker {

// for now search service only support mod placement.
class HakesWorkerConfig {
 public:
  HakesWorkerConfig() = default;
  HakesWorkerConfig(const std::string& embed_endpoint_type,
                    const std::string& embed_endpoint_config,
                    const std::string& embed_endpoint_addr,
                    const std::vector<std::string>& search_worker_addrs,
                    const int preferred_search_worker,
                    const std::string& store_addr)
      : embed_endpoint_type_(embed_endpoint_type),
        embed_endpoint_config_(embed_endpoint_config),
        embed_endpoint_addr_(embed_endpoint_addr),
        search_worker_addrs_(search_worker_addrs),
        preferred_search_worker_(preferred_search_worker),
        store_addr_(store_addr) {}

  ~HakesWorkerConfig() = default;
  // copy constructor and assignment operator
  HakesWorkerConfig(const HakesWorkerConfig&) = default;
  HakesWorkerConfig& operator=(const HakesWorkerConfig&) = default;
  // move constructor
  HakesWorkerConfig(HakesWorkerConfig&&) = default;
  // move assignment operator
  HakesWorkerConfig& operator=(HakesWorkerConfig&&) = default;

  inline std::string GetEmbedEndpointType() const {
    return embed_endpoint_type_;
  }

  inline std::string GetEmbedEndpointConfig() const {
    return embed_endpoint_config_;
  }

  inline std::string GetEmbedEndpointAddr() const {
    return embed_endpoint_addr_;
  }

  inline std::string GetServerAddress(uint64_t item_id) {
    auto id = item_id % search_worker_addrs_.size();
    return search_worker_addrs_[id];
  }

  inline int GetServerId(uint64_t item_id) {
    return (int)(item_id % search_worker_addrs_.size());
  }

  inline std::string GetSearchAddressByID(int server_id) const {
    int n = search_worker_addrs_.size();
    if (server_id < 0 || server_id >= n) {
      printf("Error: server_id: %d, n: %d\n", server_id, n);
      return "";
    }
    return search_worker_addrs_[server_id];
  }

  inline std::string GetPreferredSearchAddress() const {
    return search_worker_addrs_[preferred_search_worker_];
  }

  inline int ServerCount() const {
    return static_cast<int>(search_worker_addrs_.size());
  }

  inline std::string GetStoreAddr() const { return store_addr_; }

  inline std::string to_string() const {
    std::string s;
    s.reserve(400);
    s += "embed_endpoint_type: " + embed_endpoint_type_;
    s += "embed_endpoint_addr: " + embed_endpoint_addr_;
    s += "\nsearch_worker_addrs: ";
    for (const auto& addr : search_worker_addrs_) {
      s += addr + " ";
    }
    s += "\npreferred_search_worker: " +
         std::to_string(preferred_search_worker_);
    s += "\nstore_addr: ";
    s += store_addr_;
    s += "\n";
    return s;
  }

 private:
  std::string embed_endpoint_type_;
  std::string embed_endpoint_config_;
  std::string embed_endpoint_addr_;
  std::vector<std::string> search_worker_addrs_;
  int preferred_search_worker_;
  std::string store_addr_;
};

HakesWorkerConfig ParseHakesWorkerConfig(const std::string& cfg);

}  // namespace hakes_worker

#endif  // HAKES_HAKESWORKER_COMMON_CONFIG_H_
