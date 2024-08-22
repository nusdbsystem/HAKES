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

#include "search-worker/index/ext/IndexRefineL.h"

#include "search-worker/index/ext/IndexFlatL.h"
#include "search-worker/index/ext/utils.h"
#include "search-worker/index/impl/AuxIndexStructures.h"
#include "search-worker/index/utils/Heap.h"
#include "search-worker/index/utils/distances.h"
#include "search-worker/index/utils/utils.h"

#include <iostream>

namespace faiss {

/***************************************************
 * IndexRefineL
 ***************************************************/

IndexRefineL::IndexRefineL(Index* base_index, Index* refine_index)
    : Index(base_index->d, base_index->metric_type),
      base_index(base_index),
      refine_index(refine_index) {
  own_fields = own_refine_index = false;
  if (refine_index != nullptr) {
    assert(base_index->d == refine_index->d);
    assert(base_index->metric_type == refine_index->metric_type);
    is_trained = base_index->get_is_trained() && refine_index->get_is_trained();
    assert(base_index->get_ntotal() == refine_index->get_ntotal());
  }  // other case is useful only to construct an IndexRefineFlatL
  ntotal = base_index->get_ntotal();
}

void IndexRefineL::train(idx_t n, const float* x) {
  // std::unique_lock lock(mu);
  pthread_rwlock_wrlock(&mu);
  base_index->train(n, x);
  refine_index->train(n, x);
  is_trained = true;
  pthread_rwlock_unlock(&mu);
}

void IndexRefineL::add(idx_t n, const float* x) {
  // FAISS_THROW_MSG("add not implemented use add_with_ids");
  assert(!"add not implemented use add_with_ids");
}

void IndexRefineL::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
  // std::chrono::high_resolution_clock::time_point begin =
  //     std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> obtain_lock, convert, add_refine;
  {
    // std::unique_lock lock(mu);
    pthread_rwlock_wrlock(&mu);
    // obtain_lock = std::chrono::high_resolution_clock::now() - begin;
    assert(is_trained);
    for (idx_t i = 0; i < n; i++) {
      idx_t key = ntotal + i;
      idx_to_off[xids[i]] = key;
      off_to_idx[key] = xids[i];
    }
    // convert = std::chrono::high_resolution_clock::now() - begin;
    // auto add_refine_begin = std::chrono::high_resolution_clock::now();
    refine_index->add(n, x);
    // add_refine = std::chrono::high_resolution_clock::now() -
    // add_refine_begin;
    ntotal = refine_index->ntotal;
    pthread_rwlock_unlock(&mu);
  }
  base_index->add_with_ids(n, x, xids);
  // std::chrono::duration<double> total =
  //     std::chrono::high_resolution_clock::now() - begin;
}

void IndexRefineL::reset() {
  // std::unique_lock lock(mu);
  pthread_rwlock_wrlock(&mu);
  idx_to_off.clear();
  off_to_idx.clear();
  base_index->reset();
  refine_index->reset();
  ntotal = 0;
  pthread_rwlock_unlock(&mu);
}

void IndexRefineL::search(idx_t n, const float* x, idx_t k, float* distances,
                          idx_t* labels,
                          const SearchParameters* params_in) const {
  const IndexRefineLSearchParameters* params = nullptr;
  if (params_in) {
    params = dynamic_cast<const IndexRefineLSearchParameters*>(params_in);
    // FAISS_THROW_IF_NOT_MSG(params, "IndexRefineL params have incorrect
    // type");
    assert(params);
  }

  idx_t k_base =
      (params != nullptr) ? idx_t(k * params->k_factor) : idx_t(k * k_factor);
  SearchParameters* base_index_params =
      (params != nullptr) ? params->base_index_params : nullptr;

  assert(k_base >= k);

  assert(base_index);
  assert(refine_index);

  assert(k > 0);
  assert(get_is_trained());
  idx_t* base_labels = labels;
  float* base_distances = distances;
  std::unique_ptr<idx_t[]> del1;
  std::unique_ptr<float[]> del2;

  if (k != k_base) {
    base_labels = new idx_t[n * k_base];
    del1.reset(base_labels);
    base_distances = new float[n * k_base];
    del2.reset(base_distances);
  }

  {
    // obtain shared access to prevent concurrent add
    // std::shared_lock lock(mu);
    pthread_rwlock_rdlock(&mu);

    base_index->search(n, x, k_base, base_distances, base_labels,
                       base_index_params);

    for (int i = 0; i < n * k_base; i++) {
      // assert(base_labels[i] >= -1 && base_labels[i] < ntotal);
      assert(base_labels[i] >= -1);
      if (base_labels[i] >= 0) {  // base label can take -1.
        base_labels[i] = idx_to_off.at(base_labels[i]);
      }
    }

    // parallelize over queries
#pragma omp parallel if (n > 1)
    {
      std::unique_ptr<DistanceComputer> dc(
          refine_index->get_distance_computer());
#pragma omp for
      for (idx_t i = 0; i < n; i++) {
        dc->set_query(x + i * d);
        idx_t ij = i * k_base;
        for (idx_t j = 0; j < k_base; j++) {
          idx_t idx = base_labels[ij];
          if (idx < 0) break;
          base_distances[ij] = (*dc)(idx);
          ij++;
        }
      }
    }
    pthread_rwlock_unlock(&mu);
  }

  // sort and store result
  if (metric_type == METRIC_L2) {
    typedef CMax<float, idx_t> C;
    reorder_2_heaps<C>(n, k, labels, distances, k_base, base_labels,
                       base_distances);

  } else if (metric_type == METRIC_INNER_PRODUCT) {
    typedef CMin<float, idx_t> C;
    reorder_2_heaps<C>(n, k, labels, distances, k_base, base_labels,
                       base_distances);
  } else {
    // FAISS_THROW_MSG("Metric type not supported");
    assert(!"Metric type not supported");
  }

  {
    // std::shared_lock lock(mu);
    pthread_rwlock_rdlock(&mu);
    // transform refine index xid back to original xid
    for (int i = 0; i < n * k; i++) {
      if (labels[i] >= 0) {
        labels[i] = off_to_idx.at(labels[i]);
      }
    }
    pthread_rwlock_unlock(&mu);
  }
}

void IndexRefineL::reconstruct(idx_t key, float* recons) const {
  // std::unique_lock lock(mu);
  pthread_rwlock_wrlock(&mu);
  refine_index->reconstruct(key, recons);
  pthread_rwlock_unlock(&mu);
}

size_t IndexRefineL::sa_code_size() const {
  return base_index->sa_code_size() + refine_index->sa_code_size();
}

void IndexRefineL::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
  size_t cs1 = base_index->sa_code_size(), cs2 = refine_index->sa_code_size();
  std::unique_ptr<uint8_t[]> tmp1(new uint8_t[n * cs1]);
  base_index->sa_encode(n, x, tmp1.get());
  std::unique_ptr<uint8_t[]> tmp2(new uint8_t[n * cs2]);
  refine_index->sa_encode(n, x, tmp2.get());
  for (size_t i = 0; i < n; i++) {
    uint8_t* b = bytes + i * (cs1 + cs2);
    memcpy(b, tmp1.get() + cs1 * i, cs1);
    memcpy(b + cs1, tmp2.get() + cs2 * i, cs2);
  }
}

void IndexRefineL::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
  size_t cs1 = base_index->sa_code_size(), cs2 = refine_index->sa_code_size();
  std::unique_ptr<uint8_t[]> tmp2(
      new uint8_t[n * refine_index->sa_code_size()]);
  for (size_t i = 0; i < n; i++) {
    memcpy(tmp2.get() + i * cs2, bytes + i * (cs1 + cs2), cs2);
  }

  refine_index->sa_decode(n, tmp2.get(), x);
}

IndexRefineL::~IndexRefineL() {
  if (own_fields) delete base_index;
  if (own_refine_index) delete refine_index;
}

/***************************************************
 * IndexRefineFlatL
 ***************************************************/

IndexRefineFlatL::IndexRefineFlatL(Index* base_index)
    : IndexRefineL(base_index,
                   new IndexFlatL(base_index->d, base_index->metric_type)) {
  //   is_trained = base_index->is_trained;
  own_refine_index = true;
  // FAISS_THROW_IF_NOT_MSG(base_index->get_ntotal() == 0,
  //                        "base_index should be empty in the beginning");
  assert(base_index->get_ntotal() == 0);
}

void IndexRefineFlatL::search(idx_t n, const float* x, idx_t k,
                              float* distances, idx_t* labels,
                              const SearchParameters* params_in) const {
  const IndexRefineLSearchParameters* params = nullptr;
  if (params_in) {
    params = dynamic_cast<const IndexRefineLSearchParameters*>(params_in);
    // FAISS_THROW_IF_NOT_MSG(params,
    //                        "IndexRefineFlatL params have incorrect type");
    assert(params);
  }

  idx_t k_base =
      (params != nullptr) ? idx_t(k * params->k_factor) : idx_t(k * k_factor);
  SearchParameters* base_index_params =
      (params != nullptr) ? params->base_index_params : nullptr;

  assert(k_base >= k);

  assert(base_index);
  assert(refine_index);

  assert(k > 0);
  assert(get_is_trained());
  idx_t* base_labels = labels;
  float* base_distances = distances;
  std::unique_ptr<idx_t[]> del1;
  std::unique_ptr<float[]> del2;

  if (k != k_base) {
    base_labels = new idx_t[n * k_base];
    del1.reset(base_labels);
    base_distances = new float[n * k_base];
    del2.reset(base_distances);
  }

  // std::chrono::high_resolution_clock::time_point begin =
  //     std::chrono::high_resolution_clock::now();

  base_index->search(n, x, k_base, base_distances, base_labels,
                     base_index_params);
  {
    // obtain shared access for id translation to prevent concurrent add
    // std::shared_lock lock(mu);
    pthread_rwlock_rdlock(&mu);
    for (int i = 0; i < n * k_base; i++) {
      // assert(base_labels[i] >= -1 && base_labels[i] < ntotal);
      assert(base_labels[i] >= -1);
      if (base_labels[i] >= 0) {  // base label can take -1.
        try {
          base_labels[i] = idx_to_off.at(base_labels[i]);
        } catch (const std::exception& e) {
          // std::cerr << e.what() << '\n';
          // std::cerr << "base_labels[i]: " << base_labels[i] << std::endl;
          assert(!"base_labels[i] not found in idx_to_off");
        }
      }
    }
    pthread_rwlock_unlock(&mu);
  }
  // auto base_search_end = std::chrono::high_resolution_clock::now();

  // compute refined distances
  auto rf = dynamic_cast<const IndexFlatL*>(refine_index);
  assert(rf);

  rf->compute_distance_subset(n, x, k_base, base_distances, base_labels);
  // std::chrono::high_resolution_clock::time_point refine_search_end =
  //     std::chrono::high_resolution_clock::now();

  // sort and store result
  if (metric_type == METRIC_L2) {
    typedef CMax<float, idx_t> C;
    reorder_2_heaps<C>(n, k, labels, distances, k_base, base_labels,
                       base_distances);

  } else if (metric_type == METRIC_INNER_PRODUCT) {
    typedef CMin<float, idx_t> C;
    reorder_2_heaps<C>(n, k, labels, distances, k_base, base_labels,
                       base_distances);
  } else {
    // FAISS_THROW_MSG("Metric type not supported");
    assert(!"Metric type not supported");
  }
  // std::chrono::high_resolution_clock::time_point reorder_end =
  //     std::chrono::high_resolution_clock::now();

  {
    // std::shared_lock lock(mu);
    pthread_rwlock_rdlock(&mu);

    // transform refine index xid back to original xid
    for (int i = 0; i < n * k; i++) {
      if (labels[i] >= 0) {
        labels[i] = off_to_idx.at(labels[i]);
      }
    }
    pthread_rwlock_unlock(&mu);
  }

  // std::chrono::high_resolution_clock::time_point end =
  //     std::chrono::high_resolution_clock::now();
}

void IndexRefineFlatL::reserve(idx_t n) {
  // cast the refine index to IndexFlatL
  auto rf = dynamic_cast<IndexFlatL*>(refine_index);
  rf->reserve(n);
}

}  // namespace faiss
