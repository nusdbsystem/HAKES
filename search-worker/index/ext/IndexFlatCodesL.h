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

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAKES_SEARCHWORKER_INDEX_EXT_INDEXFLATCODESL_H_
#define HAKES_SEARCHWORKER_INDEX_EXT_INDEXFLATCODESL_H_

#include "search-worker/index/Index.h"
#include "search-worker/index/impl/DistanceComputer.h"

// #include <shared_mutex>
#include <pthread.h>

#include <vector>

namespace faiss {

struct CodePacker;

struct IndexFlatCodesL : Index {
  size_t code_size;
  std::vector<uint8_t> codes;

  // mutable std::shared_mutex codes_mutex;
  mutable pthread_rwlock_t codes_mutex;

  bool get_is_trained() const override {
    // std::shared_lock lock(this->codes_mutex);
    // return is_trained;
    pthread_rwlock_rdlock(&this->codes_mutex);
    auto ret = this->is_trained;
    pthread_rwlock_unlock(&this->codes_mutex);
    return ret;
  }

  idx_t get_ntotal() const override {
    // std::shared_lock lock(this->codes_mutex);
    // return ntotal;
    pthread_rwlock_rdlock(&this->codes_mutex);
    auto ret = this->ntotal;
    pthread_rwlock_unlock(&this->codes_mutex);
    return ret;
  }

  IndexFlatCodesL() { pthread_rwlock_init(&this->codes_mutex, NULL); };
  IndexFlatCodesL(size_t code_size, idx_t d, MetricType metric)
      : Index(d, metric), code_size(code_size) {
    pthread_rwlock_init(&this->codes_mutex, NULL);
  }

  ~IndexFlatCodesL() override { pthread_rwlock_destroy(&this->codes_mutex); }

  // copy constructors
  IndexFlatCodesL(const IndexFlatCodesL& other) = delete;
  IndexFlatCodesL& operator=(const IndexFlatCodesL& other) = delete;
  // move constructors
  IndexFlatCodesL(IndexFlatCodesL&& other) = default;
  IndexFlatCodesL& operator=(IndexFlatCodesL&& other) = default;

  void add(idx_t n, const float* x) override;

  // assume that xids is in ascending order
  void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

  void reset() override;

  void reserve(idx_t n);

  void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

  void reconstruct(idx_t key, float* recons) const override;

  size_t sa_code_size() const override;

  /** remove some ids. NB that because of the structure of the
   * index, the semantics of this operation are
   * different from the usual ones: the new ids are shifted */
  size_t remove_ids(const IDSelector& sel) override;

  /** a FlatCodesDistanceComputer offers a distance_to_code method */
  virtual FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const;

  DistanceComputer* get_distance_computer() const override {
    return get_FlatCodesDistanceComputer();
  }

  // returns a new instance of a CodePacker
  CodePacker* get_CodePacker() const;

  void check_compatible_for_merge(const Index& otherIndex) const override;

  virtual void merge_from(Index& otherIndex, idx_t add_id = 0) override;

  // permute_entries. perm of size ntotal maps new to old positions
  void permute_entries(const idx_t* perm);
};

}  // namespace faiss

#endif  // HAKES_SEARCHWORKER_INDEX_EXT_INDEXFLATCODESL_H_