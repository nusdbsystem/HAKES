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

// #include <faiss/ext/IndexScalarQuantizerL.h>
#include <search-worker/index/ext/IndexScalarQuantizerL.h>
// #include <faiss/impl/AuxIndexStructures.h>
// #include <faiss/impl/FaissAssert.h>
// #include <faiss/impl/IDSelector.h>
// #include <faiss/impl/ScalarQuantizer.h>
// #include <faiss/utils/utils.h>
#include <omp.h>

#include <algorithm>
#include <cstdio>

namespace faiss {

/*******************************************************************
 * IndexScalarQuantizerL implementation
 ********************************************************************/

IndexScalarQuantizerL::IndexScalarQuantizerL(
    int d, ScalarQuantizer::QuantizerType qtype, MetricType metric)
    : IndexFlatCodesL(0, d, metric), sq(d, qtype) {
  is_trained = qtype == ScalarQuantizer::QT_fp16 ||
               qtype == ScalarQuantizer::QT_8bit_direct;
  code_size = sq.code_size;
}

IndexScalarQuantizerL::IndexScalarQuantizerL()
    : IndexScalarQuantizerL(0, ScalarQuantizer::QT_8bit) {}

void IndexScalarQuantizerL::train(idx_t n, const float* x) {
  sq.train(n, x);
  is_trained = true;
}

void IndexScalarQuantizerL::search(idx_t n, const float* x, idx_t k,
                                   float* distances, idx_t* labels,
                                   const SearchParameters* params) const {
  const uint8_t* xb = get_xb();
  const IDSelector* sel = params ? params->sel : nullptr;

  // FAISS_THROW_IF_NOT(k > 0);
  // FAISS_THROW_IF_NOT(is_trained);
  // FAISS_THROW_IF_NOT(metric_type == METRIC_L2 ||
  //                    metric_type == METRIC_INNER_PRODUCT);
  assert(k > 0);
  assert(is_trained);
  assert(metric_type == METRIC_L2 || metric_type == METRIC_INNER_PRODUCT);

  // #pragma omp parallel
  {
    std::unique_ptr<InvertedListScanner> scanner(
        sq.select_InvertedListScanner(metric_type, nullptr, true, sel));

    scanner->list_no = 0;  // directly the list number

    // #pragma omp for
    for (idx_t i = 0; i < n; i++) {
      float* D = distances + k * i;
      idx_t* I = labels + k * i;
      // re-order heap
      if (metric_type == METRIC_L2) {
        maxheap_heapify(k, D, I);
      } else {
        minheap_heapify(k, D, I);
      }
      //   for (int j = 0; j < k; ++j) {
      //     printf(" %ld (%.4f)", I[j], D[j]);
      //   }
      //   auto scan_start_time = std::chrono::high_resolution_clock::now();
      scanner->set_query(x + i * d);
      scanner->scan_codes(ntotal, xb, nullptr, D, I, k);
      //   auto scan_end_time = std::chrono::high_resolution_clock::now();

      //   auto heap_start_time = std::chrono::high_resolution_clock::now();
      // re-order heap
      if (metric_type == METRIC_L2) {
        maxheap_reorder(k, D, I);
      } else {
        minheap_reorder(k, D, I);
      }
      //   for (int j = 0; j < k; ++j) {
      //     printf(" %ld (%.4f)", I[j], D[j]);
      //   }
      //   auto heap_end_time = std::chrono::high_resolution_clock::now();
      //   if (print_stats) {
      //     printf("scan time: %.6f, heap time: %.6f\n",
      //            std::chrono::duration<double>(scan_end_time -
      //            scan_start_time)
      //                .count(),
      //            std::chrono::duration<double>(heap_end_time -
      //            heap_start_time)
      //                .count());
      //   }
    }
  }
}

// FlatCodesDistanceComputer*
// IndexScalarQuantizerL::get_FlatCodesDistanceComputer()
//         const {
//     ScalarQuantizer::SQDistanceComputer* dc =
//             sq.get_distance_computer(metric_type);
//     dc->code_size = sq.code_size;
//     dc->codes = codes.data();
//     return dc;
// }

/* Codec interface */

void IndexScalarQuantizerL::sa_encode(idx_t n, const float* x,
                                      uint8_t* bytes) const {
  // FAISS_THROW_IF_NOT(is_trained);
  assert(is_trained);
  sq.compute_codes(x, bytes, n);
}

void IndexScalarQuantizerL::sa_decode(idx_t n, const uint8_t* bytes,
                                      float* x) const {
  // FAISS_THROW_IF_NOT(is_trained);
  assert(is_trained);
  sq.decode(bytes, x, n);
}

void IndexScalarQuantizerL::compute_distance_subset(
    idx_t n, const float* x, idx_t k, float* distances, const idx_t* labels,
    const SearchParameters* params) const {
  const IDSelector* sel = params ? params->sel : nullptr;

  // FAISS_THROW_IF_NOT(k > 0);
  // FAISS_THROW_IF_NOT(is_trained);
  // FAISS_THROW_IF_NOT(metric_type == METRIC_L2 ||
  //                    metric_type == METRIC_INNER_PRODUCT);
  assert(k > 0);
  assert(is_trained);
  assert(metric_type == METRIC_L2 || metric_type == METRIC_INNER_PRODUCT);

  std::unique_ptr<InvertedListScanner> scanner(
      sq.select_InvertedListScanner(metric_type, nullptr, true, sel));
  scanner->set_query(x);
  //   auto code_size = scanner->code_size;

  const uint8_t* xb = get_xb();
  for (idx_t i = 0; i < k; i++) {
    if (labels[i] < 0) {
      break;
    }
    distances[i] = scanner->distance_to_code(xb + labels[i] * code_size);
  }
  release_xb();
}

// /*******************************************************************
//  * IndexIVFScalarQuantizer implementation
//  ********************************************************************/

// IndexIVFScalarQuantizer::IndexIVFScalarQuantizer(
//         Index* quantizer,
//         size_t d,
//         size_t nlist,
//         ScalarQuantizer::QuantizerType qtype,
//         MetricType metric,
//         bool by_residual)
//         : IndexIVF(quantizer, d, nlist, 0, metric), sq(d, qtype) {
//     code_size = sq.code_size;
//     this->by_residual = by_residual;
//     // was not known at construction time
//     invlists->code_size = code_size;
//     is_trained = false;
// }

// IndexIVFScalarQuantizer::IndexIVFScalarQuantizer() : IndexIVF() {
//     by_residual = true;
// }

// void IndexIVFScalarQuantizer::train_encoder(
//         idx_t n,
//         const float* x,
//         const idx_t* assign) {
//     sq.train(n, x);
// }

// idx_t IndexIVFScalarQuantizer::train_encoder_num_vectors() const {
//     return 100000;
// }

// void IndexIVFScalarQuantizer::encode_vectors(
//         idx_t n,
//         const float* x,
//         const idx_t* list_nos,
//         uint8_t* codes,
//         bool include_listnos) const {
//     std::unique_ptr<ScalarQuantizer::SQuantizer>
//     squant(sq.select_quantizer()); size_t coarse_size = include_listnos ?
//     coarse_code_size() : 0; memset(codes, 0, (code_size + coarse_size) * n);

// #pragma omp parallel if (n > 1000)
//     {
//         std::vector<float> residual(d);

// #pragma omp for
//         for (idx_t i = 0; i < n; i++) {
//             int64_t list_no = list_nos[i];
//             if (list_no >= 0) {
//                 const float* xi = x + i * d;
//                 uint8_t* code = codes + i * (code_size + coarse_size);
//                 if (by_residual) {
//                     quantizer->compute_residual(xi, residual.data(),
//                     list_no); xi = residual.data();
//                 }
//                 if (coarse_size) {
//                     encode_listno(list_no, code);
//                 }
//                 squant->encode_vector(xi, code + coarse_size);
//             }
//         }
//     }
// }

// void IndexIVFScalarQuantizer::sa_decode(idx_t n, const uint8_t* codes, float*
// x)
//         const {
//     std::unique_ptr<ScalarQuantizer::SQuantizer>
//     squant(sq.select_quantizer()); size_t coarse_size = coarse_code_size();

// #pragma omp parallel if (n > 1000)
//     {
//         std::vector<float> residual(d);

// #pragma omp for
//         for (idx_t i = 0; i < n; i++) {
//             const uint8_t* code = codes + i * (code_size + coarse_size);
//             int64_t list_no = decode_listno(code);
//             float* xi = x + i * d;
//             squant->decode_vector(code + coarse_size, xi);
//             if (by_residual) {
//                 quantizer->reconstruct(list_no, residual.data());
//                 for (size_t j = 0; j < d; j++) {
//                     xi[j] += residual[j];
//                 }
//             }
//         }
//     }
// }

// void IndexIVFScalarQuantizer::add_core(
//         idx_t n,
//         const float* x,
//         const idx_t* xids,
//         const idx_t* coarse_idx) {
//     FAISS_THROW_IF_NOT(is_trained);

//     size_t nadd = 0;
//     std::unique_ptr<ScalarQuantizer::SQuantizer>
//     squant(sq.select_quantizer());

//     DirectMapAdd dm_add(direct_map, n, xids);

// #pragma omp parallel reduction(+ : nadd)
//     {
//         std::vector<float> residual(d);
//         std::vector<uint8_t> one_code(code_size);
//         int nt = omp_get_num_threads();
//         int rank = omp_get_thread_num();

//         // each thread takes care of a subset of lists
//         for (size_t i = 0; i < n; i++) {
//             int64_t list_no = coarse_idx[i];
//             if (list_no >= 0 && list_no % nt == rank) {
//                 int64_t id = xids ? xids[i] : ntotal + i;

//                 const float* xi = x + i * d;
//                 if (by_residual) {
//                     quantizer->compute_residual(xi, residual.data(),
//                     list_no); xi = residual.data();
//                 }

//                 memset(one_code.data(), 0, code_size);
//                 squant->encode_vector(xi, one_code.data());

//                 size_t ofs = invlists->add_entry(list_no, id,
//                 one_code.data());

//                 dm_add.add(i, list_no, ofs);
//                 nadd++;

//             } else if (rank == 0 && list_no == -1) {
//                 dm_add.add(i, -1, 0);
//             }
//         }
//     }

//     ntotal += n;
// }

// InvertedListScanner* IndexIVFScalarQuantizer::get_InvertedListScanner(
//         bool store_pairs,
//         const IDSelector* sel) const {
//     return sq.select_InvertedListScanner(
//             metric_type, quantizer, store_pairs, sel, by_residual);
// }

// void IndexIVFScalarQuantizer::reconstruct_from_offset(
//         int64_t list_no,
//         int64_t offset,
//         float* recons) const {
//     const uint8_t* code = invlists->get_single_code(list_no, offset);

//     if (by_residual) {
//         std::vector<float> centroid(d);
//         quantizer->reconstruct(list_no, centroid.data());

//         sq.decode(code, recons, 1);
//         for (int i = 0; i < d; ++i) {
//             recons[i] += centroid[i];
//         }
//     } else {
//         sq.decode(code, recons, 1);
//     }
// }

}  // namespace faiss
