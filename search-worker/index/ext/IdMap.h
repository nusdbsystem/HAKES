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

#ifndef HAKES_SEARCHWORKER_INDEX_EXT_IDMAP_H_
#define HAKES_SEARCHWORKER_INDEX_EXT_IDMAP_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "search-worker/index/MetricType.h"
#include "utils/io.h"

namespace faiss {

class IDMap {
 public:
  virtual ~IDMap() = default;

  virtual std::vector<faiss::idx_t> add_ids(int n, const faiss::idx_t* ids) = 0;

  // return the offset for the given ids
  // inplace update to vals
  virtual bool get_val_for_ids(int n, const faiss::idx_t* query,
                               faiss::idx_t* vals) const = 0;
  // return the ids for the given offsets
  // inplace update to keys
  virtual bool get_keys_for_ids(int n, const faiss::idx_t* query,
                                faiss::idx_t* keys) const = 0;

  virtual size_t size() const = 0;

  virtual bool reset() = 0;

  virtual bool load(hakes::IOReader* f, int io_flags = 0) = 0;

  virtual bool save(hakes::IOWriter* f) const = 0;

  virtual std::string to_string() const { return "IDMap"; };
};

class IDMapImpl : public IDMap {
 public:
  IDMapImpl() : ntotal_(0) {}

  // delete copy constructor and assignment operator
  IDMapImpl(const IDMapImpl&) = delete;
  IDMapImpl& operator=(const IDMapImpl&) = delete;
  // move constructor and assignment operator
  IDMapImpl(IDMapImpl&&) = default;
  IDMapImpl& operator=(IDMapImpl&&) = default;

  std::vector<faiss::idx_t> add_ids(int n, const faiss::idx_t* ids) override;

  bool get_val_for_ids(int n, const faiss::idx_t* query,
                       faiss::idx_t* vals) const override;
  bool get_keys_for_ids(int n, const faiss::idx_t* query,
                        faiss::idx_t* keys) const override;

  inline size_t size() const override { return idx_to_off_.size(); }

  bool reset() override;

  bool load(hakes::IOReader* f, int io_flags = 0) override;
  bool save(hakes::IOWriter* f) const override;

  std::string to_string() const override;

 private:
  std::unordered_map<idx_t, idx_t> idx_to_off_;
  std::unordered_map<idx_t, idx_t> off_to_idx_;
  faiss::idx_t ntotal_;
};
}  // namespace faiss

#endif  // HAKES_SEARCHWORKER_INDEX_EXT_IDMAP_H_
