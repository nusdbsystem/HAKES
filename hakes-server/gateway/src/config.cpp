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

#include "hakes-worker/config.h"

#include <stdexcept>

#include "utils/json.h"

namespace hakes_worker {

HakesWorkerConfig ParseHakesWorkerConfig(const std::string& cfg) {
  auto j = json::JSON::Load(cfg);

  // get HAKES-embed config
  std::string embed_endpoint_type;
  if (!j.hasKey("embed_endpoint_type")) {
    throw std::invalid_argument(
        "embed_endpoint_type is not found in the config");
  }
  embed_endpoint_type = j["embed_endpoint_type"].ToString();

  std::string embed_endpoint_config;
  if (!j.hasKey("embed_endpoint_config")) {
    throw std::invalid_argument(
        "embed_endpoint_config is not found in the config");
  }
  embed_endpoint_config = j["embed_endpoint_config"].ToString();

  std::string embed_endpoint_addr;
  if (!j.hasKey("embed_endpoint_addr")) {
    throw std::invalid_argument(
        "embed_endpoint_addr is not found in the config");
  }
  embed_endpoint_addr = j["embed_endpoint_addr"].ToString();

  // get HAKES-search config
  std::vector<std::string> search_worker_addrs;
  search_worker_addrs.reserve(10);
  // input validation
  if (!j.hasKey("search_worker_addrs")) {
    throw std::invalid_argument(
        "search_worker_addrs is not found in the config");
  }

  int n = 0;
  for (auto& addr : j["search_worker_addrs"].ArrayRange()) {
    search_worker_addrs.emplace_back(addr.ToString());
    n++;
  }

  int preference = -1;
  if (j.hasKey("preferred_search_worker")) {
    preference = j["preferred_search_worker"].ToInt();
  }

  // get HAKES-store config

  std::string store_addr;
  if (j.hasKey("store_addr")) {
    store_addr = j["store_addr"].ToString();
  }

  return HakesWorkerConfig(embed_endpoint_type, embed_endpoint_config,
                           embed_endpoint_addr, search_worker_addrs, preference,
                           store_addr);
}

}  // namespace hakes_worker