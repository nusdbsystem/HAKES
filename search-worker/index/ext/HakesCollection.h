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

#ifndef HAKES_SEARCHWORKER_INDEX_EXT_HAKESCOLLECTION_H_
#define HAKES_SEARCHWORKER_INDEX_EXT_HAKESCOLLECTION_H_

#include "search-worker/index/VectorTransform.h"
#include "search-worker/index/ext/IdMap.h"
#include "search-worker/index/ext/IndexFlatL.h"
#include "search-worker/index/ext/IndexIVFPQFastScanL.h"

namespace faiss {

struct HakesSearchParams {
  int nprobe;
  int k;
  int k_factor;
  faiss::MetricType metric_type;
};

class HakesCollection {
 public:
  HakesCollection() = default;
  virtual ~HakesCollection() {}

  // mode = 0: load full, 1: load filter index, 2: load refine index
  virtual bool Initialize(const std::string& path, int mode = 0,
                          bool keep_pa = false) = 0;

  virtual void UpdateIndex(const HakesCollection* other) = 0;

  // it is assumed that receiving engine shall store the full vecs of all
  // inputs.
  virtual bool AddWithIds(int n, int d, const float* vecs,
                          const faiss::idx_t* ids, faiss::idx_t* assign,
                          int* vecs_t_d, std::unique_ptr<float[]>* vecs_t) = 0;

  virtual bool AddBase(int n, int d, const float* vecs,
                       const faiss::idx_t* ids) = 0;

  virtual bool AddRefine(int n, int d, const float* vecs,
                         const faiss::idx_t* ids) = 0;

  virtual bool Search(int n, int d, const float* query,
                      const HakesSearchParams& params,
                      std::unique_ptr<float[]>* distances,
                      std::unique_ptr<faiss::idx_t[]>* labels) = 0;

  virtual bool Rerank(int n, int d, const float* query, int k,
                      faiss::idx_t* k_base_count, faiss::idx_t* base_labels,
                      float* base_distances,
                      std::unique_ptr<float[]>* distances,
                      std::unique_ptr<faiss::idx_t[]>* labels) = 0;

  virtual bool Checkpoint(const std::string& checkpoint_path) const = 0;

  virtual std::string GetParams() const = 0;

  virtual bool UpdateParams(const std::string& params) = 0;

  virtual bool DeleteWithIds(int n, const faiss::idx_t* ids) = 0;

  virtual std::string to_string() const = 0;
};

}  // namespace faiss

#endif  // HAKES_SEARCHWORKER_INDEX_EXT_HAKESCOLLECTION_H_
