#include "search-worker/index/ext/HakesFilterIndex.h"

#include <cstring>
#include <stdexcept>

#include "search-worker/index/ext/index_io_ext.h"
#include "search-worker/index/ext/utils.h"
#include "utils/fileutil.h"

#define FILTER_RINDEX_NAME "rindex.bin"

namespace faiss {

bool HakesFilterIndex::Initialize(const std::string& path, int mode,
                                  bool keep_pa) {
  std::string rindex_path = path + "/" + FILTER_RINDEX_NAME;
  std::unique_ptr<hakes::FileIOReader> rf =
      hakes::IsFileExist(rindex_path)
          ? std::unique_ptr<hakes::FileIOReader>(
                new hakes::FileIOReader(rindex_path.c_str()))
          : nullptr;
  return load_hakes_filterindex(rf.get(), this);
}

bool HakesFilterIndex::AddWithIds(int n, int d, const float* vecs,
                                  const faiss::idx_t* ids, faiss::idx_t* assign,
                                  int* vecs_t_d,
                                  std::unique_ptr<float[]>* vecs_t) {
  if (refine_index_) {
    refine_index_->add_with_ids(n, vecs, ids);
  }
  return true;
}

bool HakesFilterIndex::Search(int n, int d, const float* query,
                              const HakesSearchParams& params,
                              std::unique_ptr<float[]>* distances,
                              std::unique_ptr<faiss::idx_t[]>* labels) {
  if (!distances || !labels) {
    throw std::runtime_error("distances and labels must not be nullptr");
  }

  if (n <= 0 || params.k <= 0) {
    return false;
  }

  distances->reset(new float[n * params.k]);
  labels->reset(new idx_t[n * params.k]);

  if (params.id_selector) {
    faiss::SearchParameters search_params;
    search_params.sel = const_cast<faiss::IDSelector*>(params.id_selector);
    refine_index_->search(n, query, params.k, distances->get(), labels->get(),
                          &search_params);
  } else {
    refine_index_->search(n, query, params.k, distances->get(), labels->get());
  }

  return true;
}

bool HakesFilterIndex::SearchWithIds(int n, int d, const float* query,
                                     const HakesSearchParams& params,
                                     std::unique_ptr<float[]>* distances,
                                     std::unique_ptr<faiss::idx_t[]>* labels) {
  return Search(n, d, query, params, distances, labels);
}

bool HakesFilterIndex::Rerank(int n, int d, const float* query, int k,
                              faiss::idx_t* k_base_count,
                              faiss::idx_t* base_labels, float* base_distances,
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

  std::unique_ptr<float[]> dist(new float[n * k]);
  std::memset(dist.get(), 0, n * k * sizeof(float));
  std::unique_ptr<faiss::idx_t[]> lab(new faiss::idx_t[n * k]);
  std::memset(lab.get(), 0, n * k * sizeof(faiss::idx_t));

  for (int i = 0; i < n; i++) {
    refine_index_->compute_distance_subset(
        1, query + i * d, k_base_count[i], base_distances + base_label_start[i],
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

    if (refine_index_->metric_type == faiss::METRIC_L2) {
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
      throw std::runtime_error("Metric type not supported");
    }
  }

  distances->reset(dist.release());
  labels->reset(lab.release());
  return true;
}

bool HakesFilterIndex::Checkpoint(const std::string& checkpoint_path) const {
  std::string rindex_path = checkpoint_path + "/" + FILTER_RINDEX_NAME;
  std::unique_ptr<hakes::FileIOWriter> rf =
      std::unique_ptr<hakes::FileIOWriter>(
          new hakes::FileIOWriter(rindex_path.c_str()));
  save_hakes_filterindex(rf.get(), this);
  return true;
}

std::string HakesFilterIndex::to_string() const {
  std::string ret;
  if (refine_index_) {
    ret = "HakesFilterIndex: refine_index n " +
          std::to_string(refine_index_->ntotal) + ", d " +
          std::to_string(refine_index_->d) + ", metric: " +
          (refine_index_->metric_type == faiss::METRIC_INNER_PRODUCT ? "ip"
                                                                      : "l2");
  } else {
    ret = "HakesFilterIndex: no refine_index loaded";
  }
  return ret;
}

}  // namespace faiss
