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

// -*- c++ -*-

#ifndef FAISS_INDEX_SCALAR_QUANTIZER_H
#define FAISS_INDEX_SCALAR_QUANTIZER_H

// #include <faiss/IndexFlatCodes.h>
#include <search-worker/index/IndexFlatCodes.h>
// #include <faiss/IndexIVF.h>
#include <search-worker/index/IndexIVF.h>
// #include <faiss/ext/IndexFlatCodesL.h>
#include <search-worker/index/ext/IndexFlatCodesL.h>
// #include <faiss/impl/ScalarQuantizer.h>
#include <search-worker/index/impl/ScalarQuantizer.h>
#include <stdint.h>

#include <vector>

namespace faiss {

/**
 * Flat index built on a scalar quantizer.
 */
struct IndexScalarQuantizerL : IndexFlatCodesL {
  /// Used to encode the vectors
  ScalarQuantizer sq;

  /** Constructor.
   *
   * @param d      dimensionality of the input vectors
   * @param M      number of subquantizers
   * @param nbits  number of bit per subvector index
   */
  IndexScalarQuantizerL(int d, ScalarQuantizer::QuantizerType qtype,
                        MetricType metric = METRIC_L2);

  IndexScalarQuantizerL();

  void train(idx_t n, const float* x) override;

  void search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;

  //   FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const
  //   override;

  // it will obtain shared lock, so call release_xb() after use
  const uint8_t* get_xb() const {
    // this->codes_mutex.lock_shared();
    pthread_rwlock_rdlock(&this->codes_mutex);
    return codes.data();
  }

  // void release_xb() const { this->codes_mutex.unlock_shared(); }
  void release_xb() const { pthread_rwlock_unlock(&this->codes_mutex); }

  /* standalone codec interface */
  void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

  void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

  void compute_distance_subset(
      idx_t n, const float* x, idx_t k, float* distances, const idx_t* labels,
      const SearchParameters* params = nullptr) const override;
};

// /** An IVF implementation where the components of the residuals are
//  * encoded with a scalar quantizer. All distance computations
//  * are asymmetric, so the encoded vectors are decoded and approximate
//  * distances are computed.
//  */

// struct IndexIVFScalarQuantizer : IndexIVF {
//     ScalarQuantizer sq;

//     IndexIVFScalarQuantizer(
//             Index* quantizer,
//             size_t d,
//             size_t nlist,
//             ScalarQuantizer::QuantizerType qtype,
//             MetricType metric = METRIC_L2,
//             bool by_residual = true);

//     IndexIVFScalarQuantizer();

//     void train_encoder(idx_t n, const float* x, const idx_t* assign)
//     override;

//     idx_t train_encoder_num_vectors() const override;

//     void encode_vectors(
//             idx_t n,
//             const float* x,
//             const idx_t* list_nos,
//             uint8_t* codes,
//             bool include_listnos = false) const override;

//     void add_core(
//             idx_t n,
//             const float* x,
//             const idx_t* xids,
//             const idx_t* precomputed_idx) override;

//     InvertedListScanner* get_InvertedListScanner(
//             bool store_pairs,
//             const IDSelector* sel) const override;

//     void reconstruct_from_offset(int64_t list_no, int64_t offset, float*
//     recons)
//             const override;

//     /* standalone codec interface */
//     void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
// };

}  // namespace faiss

#endif
