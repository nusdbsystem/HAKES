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

#ifndef HAKES_SEARCHWORKER_INDEX_EXT_INDEXIVFFASTSCANL_H_
#define HAKES_SEARCHWORKER_INDEX_EXT_INDEXIVFFASTSCANL_H_

#include <pthread.h>

#include <memory>

#include "search-worker/index/ext/IndexIVFL.h"
#include "search-worker/index/utils/AlignedTable.h"

namespace faiss {

struct NormTableScaler;
struct SIMDResultHandlerToFloat;

/** Fast scan version of IVFPQ and IVFAQ. Works for 4-bit PQ/AQ for now.
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
 * 14: internally multithreaded implem over nq * nprobe
 * 15: same with reservoir
 *
 * For range search, only 10 and 12 are supported.
 * add 100 to the implem to force single-thread scanning (the coarse quantizer
 * may still use multiple threads).
 */

struct IndexIVFFastScanL : IndexIVFL {
  // size of the kernel
  int bbs;  // set at build time

  size_t M;
  size_t nbits;
  size_t ksub;

  // M rounded up to a multiple of 2
  size_t M2;

  // search-time implementation
  int implem = 0;
  // skip some parts of the computation (for timing)
  int skip = 0;

  // batching factors at search time (0 = default)
  int qbs = 0;
  size_t qbs2 = 0;

  bool use_balanced_assign_ = false;
  int balance_k_ = 1;

  IndexIVFFastScanL(Index* quantizer, size_t d, size_t nlist, size_t code_size,
                    MetricType metric = METRIC_L2);

  IndexIVFFastScanL(Index* quantizer, size_t d, size_t nlist, size_t code_size,
                    bool balanced_assign = false, int balance_k = 1,
                    MetricType metric = METRIC_L2);

  IndexIVFFastScanL() {
    bbs = 0;
    M2 = 0;
    is_trained = false;
    by_residual = false;
  };

  void init_fastscan(size_t M, size_t nbits, size_t nlist, MetricType metric,
                     int bbs);

  // initialize the CodePacker in the InvertedLists
  void init_code_packer();

  ~IndexIVFFastScanL() override;

  /// orig's inverted lists (for debugging)
  InvertedLists* orig_invlists = nullptr;

  void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

  // if assigned is true, then assign is assumed to contain the assignment
  // if assigned is false, then assign is assumed to be pre-allocated and will
  // hold the assignment generated
  void add_with_ids(idx_t n, const float* x, const idx_t* xids, bool assigned,
                    idx_t* assign);

  // assign should have the size of n.
  void get_add_assign(idx_t n, const float* x, idx_t* assign);

  // prepare look-up tables

  virtual bool lookup_table_is_3d() const = 0;

  // compact way of conveying coarse quantization results
  struct CoarseQuantized {
    size_t nprobe;
    const float* dis = nullptr;
    const idx_t* ids = nullptr;
  };

  virtual void compute_LUT(size_t n, const float* x, const CoarseQuantized& cq,
                           AlignedTable<float>& dis_tables,
                           AlignedTable<float>& biases) const = 0;

  void compute_LUT_uint8(size_t n, const float* x, const CoarseQuantized& cq,
                         AlignedTable<uint8_t>& dis_tables,
                         AlignedTable<uint16_t>& biases,
                         float* normalizers) const;

  void search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;

  void search_preassigned(idx_t n, const float* x, idx_t k, const idx_t* assign,
                          const float* centroid_dis, float* distances,
                          idx_t* labels, bool store_pairs,
                          const IVFSearchParameters* params = nullptr,
                          IndexIVFStats* stats = nullptr) const override;

  void range_search(idx_t n, const float* x, float radius,
                    RangeSearchResult* result,
                    const SearchParameters* params = nullptr) const override;

  // internal search funcs

  // dispatch to implementations and parallelize
  void search_dispatch_implem(idx_t n, const float* x, idx_t k,
                              float* distances, idx_t* labels,
                              const CoarseQuantized& cq,
                              const NormTableScaler* scaler) const;

  void range_search_dispatch_implem(idx_t n, const float* x, float radius,
                                    RangeSearchResult& rres,
                                    const CoarseQuantized& cq_in,
                                    const NormTableScaler* scaler) const;

  // // impl 1 and 2 are just for verification
  // template <class C>
  // void search_implem_1(idx_t n, const float* x, idx_t k, float* distances,
  //                      idx_t* labels, const CoarseQuantized& cq,
  //                      const NormTableScaler* scaler) const;

  // template <class C>
  // void search_implem_2(idx_t n, const float* x, idx_t k, float* distances,
  //                      idx_t* labels, const CoarseQuantized& cq,
  //                      const NormTableScaler* scaler) const;

  // implem 10 and 12 are not multithreaded internally, so
  // export search stats
  void search_implem_10(idx_t n, const float* x,
                        SIMDResultHandlerToFloat& handler,
                        const CoarseQuantized& cq, size_t* ndis_out,
                        size_t* nlist_out, const NormTableScaler* scaler) const;

  void search_implem_12(idx_t n, const float* x,
                        SIMDResultHandlerToFloat& handler,
                        const CoarseQuantized& cq, size_t* ndis_out,
                        size_t* nlist_out, const NormTableScaler* scaler) const;

  // implem 14 is multithreaded internally across nprobes and queries
  void search_implem_14(idx_t n, const float* x, idx_t k, float* distances,
                        idx_t* labels, const CoarseQuantized& cq, int impl,
                        const NormTableScaler* scaler) const;

  // reconstruct vectors from packed invlists
  void reconstruct_from_offset(int64_t list_no, int64_t offset,
                               float* recons) const override;

  CodePacker* get_CodePacker() const override;

  // reconstruct orig invlists (for debugging)
  void reconstruct_orig_invlists();
};

// struct IVFFastScanStatsL {
//   uint64_t times[10];
//   uint64_t t_compute_distance_tables, t_round;
//   uint64_t t_copy_pack, t_scan, t_to_flat;
//   uint64_t reservoir_times[4];
//   double t_aq_encode;
//   double t_aq_norm_encode;
//   // std::shared_mutex mu_;  // lock this to update the stats
//   mutable pthread_rwlock_t mu_;

//   double Mcy_at(int i) { return times[i] / (1000 * 1000.0); }

//   double Mcy_reservoir_at(int i) {
//     return reservoir_times[i] / (1000 * 1000.0);
//   }
//   IVFFastScanStatsL() { reset(); }
//   // void reset() { memset(this, 0, sizeof(*this)); }
//   inline void reset() {
//     memset(times, 0, sizeof(times));
//     t_compute_distance_tables = 0;
//     t_round = 0;
//     t_copy_pack = 0;
//     t_scan = 0;
//     t_to_flat = 0;
//     memset(reservoir_times, 0, sizeof(reservoir_times));
//     t_aq_encode = 0;
//     t_aq_norm_encode = 0;
//   }
// };

// // FAISS_API extern IVFFastScanStatsL IVFFastScan_statsL;
// extern IVFFastScanStatsL IVFFastScan_statsL;

}  // namespace faiss

#endif  // HAKES_SEARCHWORKER_INDEX_EXT_INDEXIVFFASTSCANL_H_