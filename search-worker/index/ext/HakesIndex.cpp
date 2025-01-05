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

#include "search-worker/index/ext/HakesIndex.h"

#include <vector>

#include "search-worker/index/ext/IndexIVFFastScanL.h"
#include "search-worker/index/ext/IndexScalarQuantizerL.h"
#include "search-worker/index/ext/index_io_ext.h"
#include "search-worker/index/ext/utils.h"
#include "utils/io.h"

namespace faiss {

bool HakesIndex::Initialize(hakes::IOReader* ff, hakes::IOReader* rf,
                            hakes::IOReader* uf, bool keep_pa) {
  bool success = load_hakes_index(ff, rf, this, keep_pa);
  if (uf != nullptr) {
    // load query index
    faiss::HakesIndex update_index;
    success = load_hakes_index(uf, nullptr, &update_index, keep_pa);
    this->UpdateIndex(update_index);
  } else if (use_ivf_sq_) {
    // no query index provided, apply sq to the ivf centroids
    auto tmp = new IndexScalarQuantizerL(
        base_index_->d, ScalarQuantizer::QuantizerType::QT_8bit_direct,
        faiss::MetricType::METRIC_L2);
    int nlist = base_index_->quantizer->get_ntotal();
    auto scaled_codes = std::vector<float>(nlist * base_index_->d);
    const float* to_scale =
        (const float*)static_cast<IndexFlatL*>(base_index_->quantizer)
            ->codes.data();
    for (int i = 0; i < nlist * base_index_->d; i++) {
      scaled_codes[i] = to_scale[i] * 127 + 128;
    }
    tmp->add(nlist, scaled_codes.data());
    cq_ = tmp;
  } else {
    // no query index provided, just access the ivf centroids
    cq_ = base_index_->quantizer;
  }
  return success;
}

void HakesIndex::UpdateIndex(const HakesIndex& update_index) {
  assert(update_index.base_index_);
  assert(update_index.base_index_->metric_type == base_index_->metric_type);
  assert(update_index.cq_->ntotal == update_index.cq_->ntotal);
  assert(update_index.vts_.back()->d_out == base_index_->d);
  assert(update_index.base_index_->by_residual == base_index_->by_residual);
  assert(update_index.base_index_->code_size == base_index_->code_size);
  assert(update_index.base_index_->ksub == base_index_->ksub);
  assert(update_index.base_index_->nbits == base_index_->nbits);
  assert(update_index.base_index_->M == base_index_->M);
  assert(update_index.base_index_->M2 == base_index_->M2);
  assert(update_index.base_index_->implem == base_index_->implem);
  assert(update_index.base_index_->qbs2 == base_index_->qbs2);
  assert(update_index.base_index_->bbs == base_index_->bbs);

  // new vts
  std::vector<VectorTransform*> new_vts_;
  new_vts_.reserve(update_index.vts_.size());
  for (auto vt : update_index.vts_) {
    auto lt = dynamic_cast<LinearTransform*>(vt);
    LinearTransform* new_vt = new LinearTransform(vt->d_in, vt->d_out);
    new_vt->A = lt->A;
    new_vt->b = lt->b;
    new_vt->have_bias = lt->have_bias;
    new_vt->is_trained = lt->is_trained;
    new_vts_.push_back(new_vt);
  }
  // default behavior is to always install the updated index as query index
  has_q_index_ = true;

  if (has_q_index_) {
    // clear old q_vts_
    for (auto vt : q_vts_) {
      delete vt;
    }
    q_vts_ = new_vts_;
  } else {
    // release the old ones
    for (auto vt : vts_) {
      delete vt;
    }
    vts_ = new_vts_;
  }

  // sq the update index ivf centroids
  if (use_ivf_sq_) {
    // apply sq to the update index ivf centroids
    auto tmp = new IndexScalarQuantizerL(
        update_index.base_index_->d,
        ScalarQuantizer::QuantizerType::QT_8bit_direct,
        faiss::MetricType::METRIC_L2);
    int nlist = update_index.base_index_->quantizer->get_ntotal();
    auto scaled_codes = std::vector<float>(nlist * update_index.base_index_->d);
    const float* to_scale = (const float*)static_cast<IndexFlatL*>(
                                update_index.base_index_->quantizer)
                                ->codes.data();
    for (int i = 0; i < nlist * update_index.base_index_->d; i++) {
      scaled_codes[i] = to_scale[i] * 127 + 128;
    }
    tmp->add(nlist, scaled_codes.data());
    if (has_q_index_) {
      delete q_cq_;
      q_cq_ = tmp;
    } else {
      delete cq_;
      // install new ones
      cq_ = tmp;
    }
  } else {
    // new quantizer
    IndexFlatL* new_quantizer =
        new IndexFlatL(update_index.base_index_->quantizer->d,
                       update_index.base_index_->quantizer->metric_type);
    new_quantizer->ntotal = update_index.base_index_->quantizer->ntotal;
    new_quantizer->is_trained = update_index.base_index_->quantizer->is_trained;
    new_quantizer->codes =
        dynamic_cast<IndexFlatL*>(update_index.base_index_->quantizer)->codes;
    if (has_q_index_) {
      delete q_cq_;
      q_cq_ = new_quantizer;
    } else {
      delete cq_;
      // install new ones
      base_index_->quantizer = new_quantizer;
      cq_ = base_index_->quantizer;
    }
  }

  // new base index pq
  assert(update_index.base_index_->pq.M == base_index_->pq.M);
  assert(update_index.base_index_->pq.nbits == base_index_->pq.nbits);
  if (has_q_index_) {
    base_index_->has_q_pq = true;
    base_index_->q_pq = update_index.base_index_->pq;
  } else {
    base_index_->pq.centroids = update_index.base_index_->pq.centroids;
  }
}

bool HakesIndex::AddWithIds(int n, int d, const float* vecs,
                            const faiss::idx_t* xids, faiss::idx_t* assign,
                            int* vecs_t_d,
                            std::unique_ptr<float[]>* transformed_vecs) {
  auto start = std::chrono::high_resolution_clock::now();

  {
    // std::unique_lock lock(mapping_mu_);
    pthread_rwlock_wrlock(&mapping_mu_);
    mapping_->add_ids(n, xids);
    refine_index_->add(n, vecs);
    pthread_rwlock_unlock(&mapping_mu_);
  }

  auto refine_add_end = std::chrono::high_resolution_clock::now();

  if (vts_.empty()) {
    base_index_->add_with_ids(n, vecs, xids, false, assign);
    return true;
  }

  // get assignment with cq first
  std::vector<float> vecs_t(n * d);
  bool assigned = false;

  vecs_t.resize(n * d);
  std::memcpy(vecs_t.data(), vecs, n * d * sizeof(float));
  std::vector<float> tmp;
  for (auto vt : vts_) {
    tmp.resize(n * vt->d_out);
    std::memset(tmp.data(), 0, tmp.size() * sizeof(float));
    vt->apply_noalloc(n, vecs_t.data(), tmp.data());
    vecs_t.resize(tmp.size());
    std::memcpy(vecs_t.data(), tmp.data(), tmp.size() * sizeof(float));
  }

  // return the transformed vecs.
  *vecs_t_d = vecs_t.size() / n;
  transformed_vecs->reset(new float[vecs_t.size()]);
  std::memcpy(transformed_vecs->get(), vecs_t.data(),
              vecs_t.size() * sizeof(float));

  auto vt_add_end = std::chrono::high_resolution_clock::now();
  printf("transformed vecs\n");

  base_index_->get_add_assign(n, vecs_t.data(), assign);
  assigned = true;

  auto cq_assign_end = std::chrono::high_resolution_clock::now();
  printf("cq assigned\n");

  assert(vecs_t.size() == n * base_index_->d);
  // base_index_->add_with_ids(n, vecs_t.data(), xids, false, assign);
  base_index_->add_with_ids(n, vecs_t.data(), xids, assigned, assign);

  auto base_add_end = std::chrono::high_resolution_clock::now();

  printf(
      "refine add time: %f, vt add time: %f, cq assign time: %f, base add "
      "time: %f\n",
      std::chrono::duration<double>(refine_add_end - start).count(),
      std::chrono::duration<double>(vt_add_end - refine_add_end).count(),
      std::chrono::duration<double>(cq_assign_end - vt_add_end).count(),
      std::chrono::duration<double>(base_add_end - cq_assign_end).count());
  return true;
}

bool HakesIndex::AddBase(int n, int d, const float* vecs,
                         const faiss::idx_t* xids) {
  if (vts_.empty()) {
    base_index_->add_with_ids(n, vecs, xids, false, nullptr);
    return true;
  }

  faiss::idx_t assign[n];

  // get assignment with cq first
  std::vector<float> vecs_t(n * d);
  bool assigned = false;

  vecs_t.resize(n * d);
  std::memcpy(vecs_t.data(), vecs, n * d * sizeof(float));
  std::vector<float> tmp;
  for (auto vt : vts_) {
    tmp.resize(n * vt->d_out);
    std::memset(tmp.data(), 0, tmp.size() * sizeof(float));
    vt->apply_noalloc(n, vecs_t.data(), tmp.data());
    vecs_t.resize(tmp.size());
    std::memcpy(vecs_t.data(), tmp.data(), tmp.size() * sizeof(float));
  }

  base_index_->get_add_assign(n, vecs_t.data(), assign);
  assigned = true;

  // auto vt_add_end = std::chrono::high_resolution_clock::now();

  assert(vecs_t.size() == n * base_index_->d);
  // base_index_->add_with_ids(n, vecs_t.data(), xids, false, assign);
  base_index_->add_with_ids(n, vecs_t.data(), xids, assigned, assign);

  // auto base_add_end = std::chrono::high_resolution_clock::now();
  return true;
}

bool HakesIndex::Search(int n, int d, const float* query,
                        const HakesSearchParams& params,
                        std::unique_ptr<float[]>* distances,
                        std::unique_ptr<faiss::idx_t[]>* labels) {
  if (!distances || !labels) {
    throw std::runtime_error("distances and labels must not be nullptr");
  }

  if (n <= 0 || params.k <= 0 || params.k_factor <= 0 || params.nprobe <= 0) {
    // printf(
    //     "n, k, k_factor and nprobe must > 0: n %d, k: %d, k_factor: %d, "
    //     "nprobe: %d\n",
    //     n, params.k, params.k_factor, params.nprobe);
    return false;
  }

  // auto opq_vt_start = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> opq_alloc_time;
  // std::chrono::duration<double> opq_apply_time;

  auto q_vts = has_q_index_ ? q_vts_ : vts_;
  std::vector<float> query_t(n * d);
  std::memcpy(query_t.data(), query, n * d * sizeof(float));
  {
    std::vector<float> tmp;
    for (auto vt : q_vts) {
      tmp.resize(n * vt->d_out);
      std::memset(tmp.data(), 0, tmp.size() * sizeof(float));
      // opq_alloc_time = std::chrono::high_resolution_clock::now() -
      // opq_vt_start;
      vt->apply_noalloc(n, query_t.data(), tmp.data());
      // opq_apply_time = std::chrono::high_resolution_clock::now() -
      // opq_vt_start;
      query_t.resize(tmp.size());
      std::memcpy(query_t.data(), tmp.data(), tmp.size() * sizeof(float));
    }
  }
  // auto opq_vt_end = std::chrono::high_resolution_clock::now();

  // step 1: find out the invlists for the query
  std::unique_ptr<faiss::idx_t[]> cq_ids =
      // std::make_unique<faiss::idx_t[]>(params.nprobe * n);
      std::unique_ptr<faiss::idx_t[]>(new faiss::idx_t[params.nprobe * n]);
  std::unique_ptr<float[]> cq_dist =
      // std::make_unique<float[]>(params.nprobe * n);
      std::unique_ptr<float[]>(new float[params.nprobe * n]);
  auto q_cq = has_q_index_ ? q_cq_ : cq_;

  assert(q_cq->d == query_t.size() / n);
  q_cq->search(n, query_t.data(), params.nprobe, cq_dist.get(), cq_ids.get());
  // auto cq_end = std::chrono::high_resolution_clock::now();

  // step 2: search the base index
  faiss::idx_t k_base = params.k_factor * params.k;
  std::unique_ptr<faiss::idx_t[]> base_lab(new faiss::idx_t[n * k_base]);
  std::unique_ptr<float[]> base_dist(new float[n * k_base]);
  faiss::IVFSearchParameters base_params;
  base_params.nprobe = params.nprobe;
  for (int i = 0; i < n; i++) {
    base_index_->search_preassigned_new(
        1, query_t.data() + i * d, k_base, cq_ids.get() + i * params.nprobe,
        cq_dist.get() + i * params.nprobe, base_dist.get() + i * k_base,
        base_lab.get() + i * k_base, false, &base_params);
  }

  // step 3: return the base search results
  distances->reset(base_dist.release());
  labels->reset(base_lab.release());

  // auto base_end = std::chrono::high_resolution_clock::now();
  return true;
}

// Each Rerank for a vector can have different sizes of base search found
// candidate points for this node.
bool HakesIndex::Rerank(int n, int d, const float* query, int k,
                        faiss::idx_t* k_base_count, faiss::idx_t* base_labels,
                        float* base_distances,
                        std::unique_ptr<float[]>* distances,
                        std::unique_ptr<faiss::idx_t[]>* labels) {
  if (!base_labels || !base_distances) {
    throw std::runtime_error(
        "base distances and base labels must not be nullptr");
  }

  if (!distances || !labels) {
    throw std::runtime_error("distances and labels must not be nullptr");
  }
  faiss::idx_t total_labels = 0;
  std::vector<faiss::idx_t> base_label_start(n);
  for (int i = 0; i < n; i++) {
    base_label_start[i] = total_labels;
    total_labels += k_base_count[i];
  }

  // step 1(step 3 in V2): id translation
  {
    // std::shared_lock lock(mapping_mu_);
    pthread_rwlock_rdlock(&mapping_mu_);
    mapping_->get_val_for_ids(total_labels, base_labels, base_labels);
    pthread_rwlock_unlock(&mapping_mu_);
  }

  std::unique_ptr<float[]> dist(new float[n * k]);
  std::memset(dist.get(), 0, n * k * sizeof(float));
  std::unique_ptr<faiss::idx_t[]> lab(new faiss::idx_t[n * k]);
  std::memset(lab.get(), 0, n * k * sizeof(faiss::idx_t));

  // #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    // step 2 (step 4 in V2): search the refine index
    refine_index_->compute_distance_subset(1, query + i * d, k_base_count[i],
                                           base_distances + base_label_start[i],
                                           base_labels + base_label_start[i]);

    if (k >= k_base_count[i]) {
      std::memcpy(lab.get() + k * i, base_labels + base_label_start[i],
                  k_base_count[i] * sizeof(faiss::idx_t));
      std::memcpy(dist.get() + k * i, base_distances + base_label_start[i],
                  k_base_count[i] * sizeof(float));
      for (int j = k_base_count[i]; j < k; j++) {
        (lab.get() + k * i)[j] = -1;
      }
    }

    // step 3 (step 5 in V2): sort and store the result
    if (refine_index_->metric_type) {
      typedef faiss::CMax<float, faiss::idx_t> C;
      faiss::reorder_2_heaps<C>(1, (k > k_base_count[i]) ? k_base_count[i] : k,
                                lab.get() + k * i, dist.get() + k * i,
                                k_base_count[i],
                                base_labels + base_label_start[i],
                                base_distances + base_label_start[i]);

    } else if (refine_index_->metric_type == faiss::METRIC_INNER_PRODUCT) {
      typedef faiss::CMin<float, faiss::idx_t> C;
      faiss::reorder_2_heaps<C>(1, (k > k_base_count[i]) ? k_base_count[i] : k,
                                lab.get() + k * i, dist.get() + k * i,
                                k_base_count[i],
                                base_labels + base_label_start[i],
                                base_distances + base_label_start[i]);
    } else {
      // FAISS_THROW_MSG("Metric type not supported");
      assert(!"Metric type not supported");
    }
  }

  // step 4 (step 6 in V2): id translation
  {
    // std::shared_lock lock(mapping_mu_);
    pthread_rwlock_rdlock(&mapping_mu_);
    mapping_->get_keys_for_ids(n * k, lab.get(), lab.get());
    pthread_rwlock_unlock(&mapping_mu_);
  }

  // step 5 (step 7 in V2): return the result
  distances->reset(dist.release());
  labels->reset(lab.release());
  return true;
}

// same as EngineV2
bool HakesIndex::Checkpoint(hakes::IOWriter* ff, hakes::IOWriter* rf) const {
  save_hakes_index(ff, rf, this);
  return true;
}

bool HakesIndex::GetParams(hakes::IOWriter* pf) const {
  save_hakes_params(pf, this);
  return true;
}

bool HakesIndex::UpdateParams(hakes::IOReader* pf) {
  HakesIndex loaded;
  bool success = load_hakes_params(pf, &loaded);
  if (!success) {
    return false;
  }
  UpdateIndex(loaded);
  return true;
}

std::string HakesIndex::to_string() const {
  std::string ret = "HakesIndex:\nopq vt size " + std::to_string(vts_.size());
  for (auto vt : vts_) {
    ret += "\n  d_in " + std::to_string(vt->d_in) + ", d_out " +
           std::to_string(vt->d_out);
  }
  ret =
      ret + "\nbase_index n " + std::to_string(base_index_->ntotal) +
      ", nlist " + std::to_string(base_index_->nlist) + ", d " +
      std::to_string(base_index_->d) + ", pq m " +
      std::to_string(base_index_->pq.M) + ", nbits " +
      std::to_string(base_index_->pq.nbits) + ", metric: " +
      (base_index_->metric_type == faiss::METRIC_INNER_PRODUCT ? "ip" : "l2") +
      "\nmapping size " + std::to_string(mapping_->size()) +
      "\nrefine_index n " + std::to_string(refine_index_->ntotal) + ", d " +
      std::to_string(refine_index_->d) + ", metric: " +
      (refine_index_->metric_type == faiss::METRIC_INNER_PRODUCT ? "ip" : "l2");
  return ret;
};

}  // namespace faiss
