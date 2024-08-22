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

#ifndef HAKES_SEARCHWORKER_INDEX_INDEXIVFPQFASTSCAN_H_
#define HAKES_SEARCHWORKER_INDEX_INDEXIVFPQFASTSCAN_H_

#include <memory>

#include "search-worker/index/IndexIVFPQ.h"
#include "search-worker/index/ext/IndexIVFFastScanL.h"
#include "search-worker/index/impl/ProductQuantizer.h"
#include "search-worker/index/utils/AlignedTable.h"

namespace faiss {

/** Fast scan version of IVFPQ. Works for 4-bit PQ for now.
 *
 * The codes in the inverted lists are not stored sequentially but
 * grouped in blocks of size bbs. This makes it possible to very quickly
 * compute distances with SIMD instructions.
 *
 * Implementations (implem):
 * 0: auto-select implementation (default)
 * 1: orig's search, re-implemented
 * 2: orig's search, re-ordered by invlist
 * 10: optimizer int16 search, collect results in heap, no qbs
 * 11: idem, collect results in reservoir
 * 12: optimizer int16 search, collect results in heap, uses qbs
 * 13: idem, collect results in reservoir
 */

struct IndexIVFPQFastScanL : IndexIVFFastScanL {
  ProductQuantizer pq;  ///< produces the codes

  /// precomputed tables management
  int use_precomputed_table = 0;
  /// if use_precompute_table size (nlist, pq.M, pq.ksub)
  AlignedTable<float> precomputed_table;

  IndexIVFPQFastScanL(Index* quantizer, size_t d, size_t nlist, size_t M,
                      size_t nbits, MetricType metric = METRIC_L2,
                      int bbs = 32);

  IndexIVFPQFastScanL(Index* quantizer, size_t d, size_t nlist, size_t M,
                      size_t nbits, bool balanced_assign = false,
                      int balance_k = 1, MetricType metric = METRIC_L2,
                      int bbs = 32);

  IndexIVFPQFastScanL() {
    by_residual = false;
    bbs = 0;
    M2 = 0;
  };

  // built from an IndexIVFPQ
  explicit IndexIVFPQFastScanL(const IndexIVFPQ& orig, int bbs = 32);

  void train_encoder(idx_t n, const float* x, const idx_t* assign) override;

  idx_t train_encoder_num_vectors() const override;

  /// build precomputed table, possibly updating use_precomputed_table
  void precompute_table();

  /// same as the regular IVFPQ encoder. The codes are not reorganized by
  /// blocks a that point
  void encode_vectors(idx_t n, const float* x, const idx_t* list_nos,
                      uint8_t* codes,
                      bool include_listno = false) const override;

  // prepare look-up tables

  bool lookup_table_is_3d() const override;

  void compute_LUT(size_t n, const float* x, const CoarseQuantized& cq,
                   AlignedTable<float>& dis_tables,
                   AlignedTable<float>& biases) const override;

  void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
};

}  // namespace faiss

#endif  // HAKES_SEARCHWORKER_INDEX_INDEXIVFPQFASTSCAN_H_
