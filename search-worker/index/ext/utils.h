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

#ifndef HAKES_SEARCHWORKER_INDEX_EXT_UTILS_H_
#define HAKES_SEARCHWORKER_INDEX_EXT_UTILS_H_

#include <unistd.h>

#include "search-worker/index/MetricType.h"
#include "search-worker/index/utils/Heap.h"

namespace faiss {
template <class C>
inline void reorder_2_heaps(idx_t n, idx_t k, idx_t* labels, float* distances,
                            idx_t k_base, const idx_t* base_labels,
                            const float* base_distances) {
  // #pragma omp parallel for
  for (idx_t i = 0; i < n; i++) {
    idx_t* idxo = labels + i * k;
    float* diso = distances + i * k;
    const idx_t* idxi = base_labels + i * k_base;
    const float* disi = base_distances + i * k_base;

    heap_heapify<C>(k, diso, idxo, disi, idxi, k);
    if (k_base != k) {  // add remaining elements
      heap_addn<C>(k, diso, idxo, disi + k, idxi + k, k_base - k);
    }
    heap_reorder<C>(k, diso, idxo);
  }
}

inline size_t getCurrentRSS() {
  // /* Linux ---------------------------------------------------- */
  // long rss = 0L;
  // FILE *fp = NULL;
  // if ((fp = fopen("/proc/self/statm", "r")) == NULL)
  //   return (size_t)0L; /* Can't open? */
  // if (fscanf(fp, "%*s%ld", &rss) != 1) {
  //   fclose(fp);
  //   return (size_t)0L; /* Can't read? */
  // }
  // fclose(fp);
  // return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);
  return 0;
}

}  // namespace faiss

#endif  // HAKES_SEARCHWORKER_INDEX_EXT_UTILS_H_
