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

#ifndef HAKES_SEARCHWORKER_INDEX_EXT_HAKESFILTERINDEX_H_
#define HAKES_SEARCHWORKER_INDEX_EXT_HAKESFILTERINDEX_H_

#include <vector>
#include <queue>
#include <mutex>
#include <memory>
#include <random>

#include "search-worker/index/ext/HakesCollection.h"
#include "search-worker/index/ext/IndexFlatL.h"
#include "search-worker/index/ext/TagChecker.h"
#include "search-worker/index/impl/IDSelector.h"

namespace faiss {

typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template <typename dist_t>
struct FavorCompareByFirst {
    constexpr bool operator()(const std::pair<dist_t, tableint>& a,
                              const std::pair<dist_t, tableint>& b) const noexcept {
        return a.first < b.first;
    }
};

class HakesFilterIndex : public HakesCollection {
 public:
  HakesFilterIndex();
  ~HakesFilterIndex() override;

  HakesFilterIndex(const HakesFilterIndex&) = delete;
  HakesFilterIndex& operator=(const HakesFilterIndex&) = delete;
  HakesFilterIndex(HakesFilterIndex&&) = delete;
  HakesFilterIndex& operator=(HakesFilterIndex&&) = delete;

  bool Initialize(const std::string& path, int mode = 0,
                  bool keep_pa = false) override;

  void UpdateIndex(const HakesCollection* other) override;

  bool AddWithIds(int n, int d, const float* vecs, const faiss::idx_t* ids,
                  faiss::idx_t* assign, int* vecs_t_d,
                  std::unique_ptr<float[]>* vecs_t) override;

  bool AddBase(int n, int d, const float* vecs,
               const faiss::idx_t* ids) override;

  bool AddRefine(int n, int d, const float* vecs,
                 const faiss::idx_t* ids) override {
    return AddWithIds(n, d, vecs, ids, nullptr, nullptr, nullptr);
  }

  bool Search(int n, int d, const float* query, const HakesSearchParams& params,
              std::unique_ptr<float[]>* distances,
              std::unique_ptr<faiss::idx_t[]>* labels) override;

  bool Rerank(int n, int d, const float* query, int k,
              faiss::idx_t* k_base_count, faiss::idx_t* base_labels,
              float* base_distances,
              std::unique_ptr<float[]>* distances,
              std::unique_ptr<faiss::idx_t[]>* labels) override;

  bool Checkpoint(const std::string& checkpoint_path) const override;

  std::string GetParams() const override;

  bool UpdateParams(const std::string& path) override { return true; }

  inline bool DeleteWithIds(int n, const idx_t* ids) override {
    if (del_checker_) {
      del_checker_->set(n, ids);
    }
    for (int i = 0; i < n; i++) {
      MarkDeleted(ids[i]);
    }
    return true;
  }

  std::string to_string() const override;

  void InitFavorParams(size_t max_elements, size_t M = 16, 
                       size_t ef_construction = 200, size_t random_seed = 100);
  
  bool BuildFilterIndex(int n, int d, const float* vecs, const idx_t* ids);
  
  bool SearchWithFilter(int n, int d, const float* query, int k,
                        const faiss::IDSelector* id_selector,
                        float* distances, idx_t* labels);
  
  bool SaveFilterIndex(const std::string& path) const;
  bool LoadFilterIndex(const std::string& path);
  
  float SelectivityEstimator(const faiss::IDSelector* id_selector) const;
  
  float DistFilter(float p) const {
    return (1.0f - p) * (static_cast<float>(ef_) - p) * delta_d_ / (2.0f * p);
  }
  
  void MarkDeleted(idx_t label);
  bool IsMarkedDeleted(tableint internal_id) const;
  
  tableint GetInternalId(idx_t label) const;
  idx_t GetExternalLabel(tableint internal_id) const;
  const float* GetDataByInternalId(tableint internal_id) const;
  
  void SearchBaseLayerSTFilter(tableint ep_id, const float* query_data, 
                               size_t ef, size_t k, float e_distance,
                               const faiss::IDSelector* id_selector,
                               std::vector<std::pair<float, tableint>>& results) const;
  
  void SearchGraphFilter(const float* query_data, size_t k, float p,
                         const faiss::IDSelector* id_selector,
                         std::vector<std::pair<float, idx_t>>& results) const;
  
  void SearchBruteForceFilter(const float* query_data, size_t k,
                              const faiss::IDSelector* id_selector,
                              std::vector<std::pair<float, idx_t>>& results) const;

  std::unique_ptr<faiss::IndexFlatL> refine_index_;
  std::unique_ptr<TagChecker<idx_t>> del_checker_;
  
  int d_ = 0;
  faiss::MetricType metric_type_ = METRIC_L2;
  faiss::idx_t ntotal_ = 0;
  
  size_t max_elements_ = 0;
  size_t cur_element_count_ = 0;
  size_t M_ = 16;
  size_t maxM_ = 16;
  size_t maxM0_ = 32;
  size_t ef_construction_ = 200;
  size_t ef_ = 100;
  float delta_d_ = 0.0f;
  const float LARGE_DIST = 100000.0f;
  
  std::vector<char> data_level0_memory_;
  std::vector<char*> link_lists_;
  std::vector<int> element_levels_;
  std::unordered_map<idx_t, tableint> label_lookup_;
  std::unordered_set<tableint> deleted_elements_;
  
  size_t size_links_level0_ = 0;
  size_t size_data_per_element_ = 0;
  size_t offset_data_ = 0;
  size_t label_offset_ = 0;
  size_t offset_level0_ = 0;
  size_t size_links_per_element_ = 0;
  
  std::mt19937 level_generator_;
  std::mt19937 update_probability_generator_;
  float mult_ = 0.0f;
  tableint enterpoint_node_ = 0;
  int maxlevel_ = -1;
  
  mutable std::mutex label_lookup_lock_;
  mutable std::vector<std::unique_ptr<std::mutex>> link_list_locks_;
  mutable std::mutex deleted_elements_lock_;
  mutable std::mutex delta_mutex_;
  
  mutable std::vector<std::vector<bool>> visited_pool_;
  mutable std::mutex visited_pool_mutex_;
};

}  // namespace faiss

#endif  // HAKES_SEARCHWORKER_INDEX_EXT_HAKESFILTERINDEX_H_
