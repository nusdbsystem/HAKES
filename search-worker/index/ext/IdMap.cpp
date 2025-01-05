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

#include "search-worker/index/ext/IdMap.h"

#include <cassert>
#include <cstring>

#include "search-worker/index/impl/io_macros.h"

namespace faiss {

std::vector<faiss::idx_t> IDMapImpl::add_ids(int n, const faiss::idx_t* ids) {
  std::vector<faiss::idx_t> ret(n);
  for (int i = 0; i < n; i++) {
    idx_to_off_[ids[i]] = ntotal_;
    off_to_idx_[ntotal_] = ids[i];
    ret[i] = ntotal_;
    ntotal_++;
  }
  return ret;
}

bool IDMapImpl::get_val_for_ids(int n, const faiss::idx_t* query,
                                faiss::idx_t* vals) const {
  assert(vals != nullptr);
  for (int i = 0; i < n; i++) {
    assert(query[i] >= -1);
    if (query[i] >= 0) {
      vals[i] = idx_to_off_.at(query[i]);
    }
  }
  return true;
}
bool IDMapImpl::get_keys_for_ids(int n, const faiss::idx_t* query,
                                 faiss::idx_t* keys) const {
  assert(keys != nullptr);
  for (int i = 0; i < n; i++) {
    if (query[i] >= 0) {
      keys[i] = off_to_idx_.at(query[i]);
    }
  }
  return true;
}

bool IDMapImpl::reset() {
  idx_to_off_.clear();
  off_to_idx_.clear();
  ntotal_ = 0;
  return true;
}

namespace {

void write_map(const std::unordered_map<idx_t, idx_t>& m, hakes::IOWriter* f) {
  std::vector<std::pair<idx_t, idx_t>> v;
  v.resize(m.size());
  std::copy(m.begin(), m.end(), v.begin());
  WRITEVECTOR(v);
}

void read_map(std::unordered_map<idx_t, idx_t>* m, hakes::IOReader* f) {
  std::vector<std::pair<idx_t, idx_t>> v;
  READVECTOR(v);
  m->clear();
  m->reserve(v.size());
  for (auto& p : v) {
    (*m)[p.first] = p.second;
  }
}

}  // anonymous namespace

bool IDMapImpl::load(hakes::IOReader* f, int io_flags) {
  read_map(&off_to_idx_, f);
  read_map(&idx_to_off_, f);
  ntotal_ = off_to_idx_.size();
  return true;
}

bool IDMapImpl::save(hakes::IOWriter* f) const {
  write_map(off_to_idx_, f);
  write_map(idx_to_off_, f);
  return true;
}

std::string IDMapImpl::to_string() const {
  // serialize off_to_idx and idx_to_off maps
  std::string str;
  str.reserve(4000);
  str.append("IDMapImpl: ");
  str.append("\nntotal=");
  str.append(std::to_string(ntotal_));
  str.append("\noff_to_idx=");
  for (auto& p : off_to_idx_) {
    str.append(" ");
    str.append(std::to_string(p.first));
    str.append(":");
    str.append(std::to_string(p.second));
  }
  str.append("\nidx_to_off=");
  for (auto& p : idx_to_off_) {
    str.append(" ");
    str.append(std::to_string(p.first));
    str.append(":");
    str.append(std::to_string(p.second));
  }
  str.append("\n");
  return str;
}
}  // namespace faiss
