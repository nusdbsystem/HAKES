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

#ifndef HAKES_SEARCHWORKER_INDEX_EXT_HAKESINDEX_H_
#define HAKES_SEARCHWORKER_INDEX_EXT_HAKESINDEX_H_

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

class HakesIndex {
 public:
  HakesIndex() = default;
  ~HakesIndex() {
    for (auto vt : vts_) {
      delete vt;
    }
    for (auto vt : ivf_vts_) {
      delete vt;
    }
  }

  // delete copy constructors and assignment operators
  HakesIndex(const HakesIndex&) = delete;
  HakesIndex& operator=(const HakesIndex&) = delete;
  // delete move constructors and assignment operators
  HakesIndex(HakesIndex&&) = delete;
  HakesIndex& operator=(HakesIndex&&) = delete;

  // bool Initialize(const std::string& path);
  bool Initialize(IOReader* f, bool params_only = true);

  void UpdateIndex(const HakesIndex& other);

  bool AddWithIds(int n, int d, const float* vecs, const faiss::idx_t* ids) {
    // printf(
    //     "EngineV3::AddWithIds need to return results of base index "
    //     "assignment\n");
    return false;
  }

  // it is assumed that receiving engine shall store the full vecs of all
  // inputs.
  bool AddWithIds(int n, int d, const float* vecs, const faiss::idx_t* ids,
                  faiss::idx_t* assign, int* vecs_t_d,
                  std::unique_ptr<float[]>* vecs_t);

  bool AddBase(int n, int d, const float* vecs, const faiss::idx_t* ids);

  bool AddBasePreassigned(int n, int d, const float* vecs,
                          const faiss::idx_t* ids, const faiss::idx_t* assign);

  bool Search(int n, int d, const float* query, const HakesSearchParams& params,
              std::unique_ptr<float[]>* distances,
              std::unique_ptr<faiss::idx_t[]>* labels);

  bool Rerank(int n, int d, const float* query, int k,
              faiss::idx_t* k_base_count, faiss::idx_t* base_labels,
              float* base_distances, std::unique_ptr<float[]>* distances,
              std::unique_ptr<faiss::idx_t[]>* labels);

  inline bool has_ivf_vts() const { return ivf_vts_.size() > 0; }

  bool Checkpoint(IOWriter* f);

  std::string to_string() const;

  //  private:
 public:
  std::string index_path_;
  bool share_vt_;
  std::vector<faiss::VectorTransform*> vts_;
  std::vector<faiss::VectorTransform*> ivf_vts_;
  faiss::Index* cq_;
  std::unique_ptr<faiss::IndexIVFPQFastScanL> base_index_;
  mutable pthread_rwlock_t mapping_mu_;
  std::unique_ptr<faiss::IDMap> mapping_;
  std::unique_ptr<faiss::IndexFlatL> refine_index_;
};

}  // namespace faiss

#endif  // HAKES_SEARCHWORKER_INDEX_EXT_HAKESINDEX_H_
