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

#ifndef HAKES_SEARCHWORKER_INDEX_EXT_INDEXFLATL_H
#define HAKES_SEARCHWORKER_INDEX_EXT_INDEXFLATL_H

#include <vector>

#include "search-worker/index/ext/IndexFlatCodesL.h"

namespace faiss {

/** Index that stores the full vectors and performs exhaustive search */
struct IndexFlatL : IndexFlatCodesL {
  explicit IndexFlatL(idx_t d,  ///< dimensionality of the input vectors
                      MetricType metric = METRIC_L2);

  void search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;

  void range_search(idx_t n, const float* x, float radius,
                    RangeSearchResult* result,
                    const SearchParameters* params = nullptr) const override;

  void reconstruct(idx_t key, float* recons) const override;

  /** compute distance with a subset of vectors
   *
   * @param x       query vectors, size n * d
   * @param labels  indices of the vectors that should be compared
   *                for each query vector, size n * k
   * @param distances
   *                corresponding output distances, size n * k
   */
  void compute_distance_subset(idx_t n, const float* x, idx_t k,
                               float* distances, const idx_t* labels) const;

  // it will obtain shared lock, so call release_xb() after use
  const float* get_xb() const {
    // this->codes_mutex.lock_shared();
    pthread_rwlock_rdlock(&this->codes_mutex);
    return (const float*)codes.data();
  }

  void release_xb() const {
    // this->codes_mutex.unlock_shared();
    pthread_rwlock_unlock(&this->codes_mutex);
  }

  IndexFlatL() {}

  FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

  /* The stanadlone codec interface (just memcopies in this case) */
  void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

  void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
};

struct IndexFlatLIP : IndexFlatL {
  explicit IndexFlatLIP(idx_t d) : IndexFlatL(d, METRIC_INNER_PRODUCT) {}
  IndexFlatLIP() {}
};

struct IndexFlatLL2 : IndexFlatL {
  /**
   * @param d dimensionality of the input vectors
   */
  explicit IndexFlatLL2(idx_t d) : IndexFlatL(d, METRIC_L2) {}
  IndexFlatLL2() {}

  // override for l2 norms cache.
  FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;
};

}  // namespace faiss

#endif  // HAKES_SEARCHWORKER_INDEX_EXT_INDEXFLATL_H