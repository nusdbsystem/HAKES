#include "search-worker/index/ext/HakesFlatIndex.h"

#include "search-worker/index/ext/index_io_ext.h"
#include "utils/fileutil.h"

#define FLAT_RINDEX_NAME "rindex.bin"

namespace faiss {

bool HakesFlatIndex::Initialize(const std::string& path, int mode,
                                bool keep_pa) {
  std::string rindex_path = path + "/" + FLAT_RINDEX_NAME;
  std::unique_ptr<hakes::FileIOReader> rf =
      hakes::IsFileExist(rindex_path)
          ? std::unique_ptr<hakes::FileIOReader>(
                new hakes::FileIOReader(rindex_path.c_str()))
          : nullptr;
  return load_hakes_flatindex(rf.get(), this);
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

bool HakesFlatIndex::Checkpoint(const std::string& checkpoint_path) const {
  std::string rindex_path = checkpoint_path + "/" + FLAT_RINDEX_NAME;
  std::unique_ptr<hakes::FileIOWriter> rf =
      std::unique_ptr<hakes::FileIOWriter>(
          new hakes::FileIOWriter(rindex_path.c_str()));
  save_hakes_flatindex(rf.get(), this);
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