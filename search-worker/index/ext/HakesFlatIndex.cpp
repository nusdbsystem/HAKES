#include "search-worker/index/ext/HakesFlatIndex.h"

#include "search-worker/index/ext/index_io_ext.h"

namespace faiss {

bool HakesFlatIndex::Initialize(hakes::IOReader* ff, hakes::IOReader* rf,
                                hakes::IOReader* uf, bool keep_pa) {
  return load_hakes_flatindex(rf, this);
}

bool HakesFlatIndex::AddWithIds(int n, int d, const float* vecs,
                                const faiss::idx_t* ids, faiss::idx_t* assign,
                                int* vecs_t_d,
                                std::unique_ptr<float[]>* vecs_t) {
  refine_index_->add_with_ids(n, vecs, ids);
  return true;
}

bool HakesFlatIndex::Search(int n, int d, const float* query,
                            const HakesSearchParams& params,
                            std::unique_ptr<float[]>* distances,
                            std::unique_ptr<faiss::idx_t[]>* labels) {
  distances->reset(new float[n * params.k]);
  labels->reset(new idx_t[n * params.k]);
  refine_index_->search(n, query, params.k, distances->get(), labels->get());
  return true;
}

bool HakesFlatIndex::Checkpoint(hakes::IOWriter* ff,
                                hakes::IOWriter* rf) const {
  save_hakes_flatindex(rf, this);
  return true;
}

std::string HakesFlatIndex::to_string() const {
  std::string ret =
      "refine_index n " + std::to_string(refine_index_->ntotal) + ", d " +
      std::to_string(refine_index_->d) + ", metric: " +
      (refine_index_->metric_type == faiss::METRIC_INNER_PRODUCT ? "ip" : "l2");
  return ret;
}

}  // namespace faiss