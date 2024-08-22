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

#ifndef HAKES_HAKESWORKER_UNTRUSTED_DATAMANAGERU_H_
#define HAKES_HAKESWORKER_UNTRUSTED_DATAMANAGERU_H_

#include <sgx_eid.h>

#include <string>
#include <vector>

#include "hakes-worker/common/data_manager.h"
#include "message/searchservice.h"

namespace hakes_worker {

class DataManagerU : public DataManager {
 public:
  DataManagerU(const std::string& enclave_file_name)
      : initialized_(false), enclave_file_name_(enclave_file_name) {}
  virtual ~DataManagerU() = default;

  bool Initialize() override;

  hakes::SearchWorkerRerankResponse MergeSearchResults(
      const std::vector<hakes::SearchWorkerRerankResponse> resps, int k,
      const std::string user_id = "", const std::string ks_addr = "",
      const uint16_t ks_port = 0) override;

 private:
  bool initialized_;
  const std::string enclave_file_name_;
  sgx_enclave_id_t eid_;
};
}  // namespace hakes_worker

#endif  // HAKES_HAKESWORKER_UNTRUSTED_DATAMANAGERU_H_
