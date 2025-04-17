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

#include "search-worker/index/ext/index_io_ext.h"

#include "search-worker/index/IndexFlat.h"
#include "search-worker/index/ext/BlockInvertedListsL.h"
#include "search-worker/index/ext/IndexFlatL.h"
#include "search-worker/index/ext/IndexIVFPQFastScanL.h"
#include "search-worker/index/ext/IndexRefineL.h"
#include "search-worker/index/ext/utils.h"
#include "search-worker/index/impl/CodePacker.h"
#include "search-worker/index/impl/io.h"
#include "search-worker/index/impl/io_macros.h"
#include "search-worker/index/impl/pq4_fast_scan.h"

namespace faiss {

/**
 * streamlined functions from index_read.cpp and index_write.cpp
 */

bool read_hakes_pretransform(hakes::IOReader* f,
                             std::vector<VectorTransform*>* vts) {
  // open pretransform file
  int32_t num_vt;
  READ1(num_vt);
  vts->reserve(num_vt);
  for (int i = 0; i < num_vt; i++) {
    int32_t d_out, d_in;
    READ1(d_out);
    READ1(d_in);
    size_t A_size = d_out * d_in;
    std::vector<float> A(d_out * d_in);
    READANDCHECK(A.data(), A_size);
    size_t b_size = d_out;
    std::vector<float> b(d_out);
    READANDCHECK(b.data(), b_size);
    LinearTransform* lt = new LinearTransform(d_in, d_out, true);
    lt->A = std::move(A);
    lt->b = std::move(b);
    lt->have_bias = true;
    lt->is_trained = true;
    vts->emplace_back(lt);
  }
  printf("read_hakes_pretransform: num_vt: %d\n", num_vt);
  return true;
}

bool write_hakes_pretransform(hakes::IOWriter* f,
                              const std::vector<VectorTransform*>* vts) {
  int32_t num_vt = vts->size();
  WRITE1(num_vt);
  for (int i = 0; i < num_vt; i++) {
    LinearTransform* lt = dynamic_cast<LinearTransform*>((*vts)[i]);
    if (lt == nullptr) {
      // printf("write_hakes_pretransform: Only LinearTransform is
      // supported\n");
      return false;
    }
    int32_t d_out = lt->d_out;
    int32_t d_in = lt->d_in;
    WRITE1(d_out);
    WRITE1(d_in);
    size_t A_size = d_out * d_in;
    WRITEANDCHECK(lt->A.data(), A_size);
    size_t b_size = d_out;
    if (!lt->have_bias) {
      auto zero_bias = std::vector<float>(d_out, 0);
      WRITEANDCHECK(zero_bias.data(), b_size);
    } else {
      WRITEANDCHECK(lt->b.data(), b_size);
    }
  }
  return true;
}

void write_hakes_ivf(hakes::IOWriter* f, const HakesIndex* idx,
                     bool q_ivf = false) {
  int32_t d = idx->base_index_->d;
  uint64_t ntotal = idx->base_index_->ntotal;
  uint8_t metric_type = (idx->base_index_->metric_type == METRIC_L2) ? 0 : 1;
  int32_t nlist = idx->base_index_->nlist;
  WRITE1(d);
  WRITE1(ntotal);
  WRITE1(metric_type);
  WRITE1(nlist);

  size_t code_size =
      idx->base_index_->nlist * idx->base_index_->d * sizeof(float);
  if (q_ivf) {
    WRITEANDCHECK(static_cast<IndexFlatL*>(idx->q_quantizer_)->codes.data(),
                  code_size);
  } else {
    WRITEANDCHECK(
        static_cast<IndexFlatL*>(idx->base_index_->quantizer)->codes.data(),
        code_size);
  }
  printf("write_hakes_ivf: d: %d, ntotal: %ld, nlist: %ld\n",
         idx->base_index_->d, idx->base_index_->ntotal,
         idx->base_index_->nlist);
}

bool read_hakes_ivf(hakes::IOReader* f, HakesIndex* idx) {
  int32_t d;
  uint64_t ntotal;
  uint8_t metric_type;
  int32_t nlist;
  READ1(d);
  READ1(ntotal);
  READ1(metric_type);
  READ1(nlist);
  idx->base_index_->d = d;
  idx->base_index_->ntotal = ntotal;
  faiss::MetricType metric =
      (metric_type == 0) ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
  idx->base_index_->metric_type = metric;
  idx->base_index_->nlist = nlist;
  printf("read_hakes_ivf: d: %d, ntotal: %ld, nlist: %d\n", d, ntotal, nlist);

  IndexFlatL* quantizer = new IndexFlatL(d, metric);
  size_t code_size = nlist * d * sizeof(float);
  quantizer->codes.resize(code_size);
  READANDCHECK(quantizer->codes.data(), code_size);
  quantizer->is_trained = true;
  quantizer->ntotal = nlist;
  idx->base_index_->quantizer = quantizer;
  return true;
}

void write_hakes_pq(hakes::IOWriter* f, const HakesIndex* idx,
                    bool q_pq = false) {
  ProductQuantizer* pq =
      (q_pq) ? &idx->base_index_->q_pq : &idx->base_index_->pq;
  int32_t d = pq->d;
  int32_t M = pq->M;
  int32_t nbits = pq->nbits;
  WRITE1(d);
  WRITE1(M);
  WRITE1(nbits);
  size_t centroids_size = pq->M * pq->ksub * pq->dsub;
  WRITEANDCHECK(pq->centroids.data(), centroids_size);
}

bool read_hakes_pq(hakes::IOReader* f, HakesIndex* idx) {
  int32_t d, M, nbits;
  READ1(d);
  READ1(M);
  READ1(nbits);
  printf("read_hakes_pq: d: %d, M: %d, nbits: %d\n", d, M, nbits);
  ProductQuantizer* pq = &idx->base_index_->pq;
  pq->d = d;
  pq->M = M;
  pq->nbits = nbits;
  pq->set_derived_values();
  pq->train_type = ProductQuantizer::Train_hot_start;
  size_t centroids_size = pq->M * pq->ksub * pq->dsub;
  pq->centroids.resize(centroids_size);
  printf("centroids_size: %ld\n", centroids_size);
  READANDCHECK(pq->centroids.data(), centroids_size);
  return true;
}

void write_hakes_compressed_vecs(hakes::IOWriter* f, const HakesIndex* idx) {
  if (idx->base_index_->ntotal == 0) {
    return;
  }
  // write the compressed vectors
  const BlockInvertedListsL* il =
      dynamic_cast<const BlockInvertedListsL*>(idx->base_index_->invlists);
  for (size_t i = 0; i < il->nlist; i++) {
    il->lists_[i].write(f);
  }
}

bool read_hakes_compressed_vecs(hakes::IOReader* f, HakesIndex* idx) {
  int32_t nlist = idx->base_index_->nlist;
  printf("read_hakes_compressed_vecs: nlist: %d\n", nlist);

  auto packer = CodePackerPQ4(idx->base_index_->pq.M, 32);
  BlockInvertedListsL* il =
      new BlockInvertedListsL(nlist, packer.nvec, packer.block_size);

  if (idx->base_index_->ntotal != 0) {
    for (size_t i = 0; i < nlist; i++) {
      il->lists_[i].read(f);
    }
  }

  il->init(nullptr);
  idx->base_index_->invlists = il;
  idx->base_index_->own_invlists = true;
  return true;
}

void write_hakes_full_vecs(hakes::IOWriter* f, const HakesIndex* idx) {
  int32_t d = idx->refine_index_->d;
  uint64_t ntotal = idx->refine_index_->ntotal;
  uint8_t metric_type = (idx->refine_index_->metric_type == METRIC_L2) ? 0 : 1;
  WRITE1(d);
  WRITE1(ntotal);
  WRITE1(metric_type);
  WRITEXBVECTOR(idx->refine_index_->codes);
}

void save_hakes_flatindex(hakes::IOWriter* f, const HakesFlatIndex* idx) {
  uint32_t d = idx->refine_index_->d;
  uint64_t ntotal = idx->refine_index_->ntotal;
  uint8_t metric_type = (idx->refine_index_->metric_type == METRIC_L2) ? 0 : 1;
  WRITE1(d);
  WRITE1(ntotal);
  WRITE1(metric_type);
  WRITEXBVECTOR(idx->refine_index_->codes);
}

bool read_hakes_full_vecs(hakes::IOReader* f, HakesIndex* idx) {
  int32_t d;
  uint64_t ntotal;
  uint8_t metric_type;
  READ1(d);
  READ1(ntotal);
  READ1(metric_type);
  if (metric_type == 0) {
    idx->refine_index_.reset(new faiss::IndexFlatLL2(d));
  } else if (metric_type == 1) {
    idx->refine_index_.reset(new faiss::IndexFlatLIP(d));
  } else {
    // printf("read_hakes_full_vecs: metric type not supported\n");
    return false;
  }
  idx->refine_index_->ntotal = ntotal;
  idx->refine_index_->is_trained = true;
  idx->refine_index_->code_size = d * sizeof(float);
  idx->refine_index_->codes.reserve(ntotal * idx->refine_index_->code_size * 2);
  READXBVECTOR(idx->refine_index_->codes);
  assert(idx->refine_index_->codes.size() ==
         idx->refine_index_->ntotal * idx->refine_index_->code_size);
  return true;
}

bool load_hakes_flatindex(hakes::IOReader* f, HakesFlatIndex* idx) {
  uint32_t d;
  uint64_t ntotal;
  uint8_t metric_type;
  READ1(d);
  READ1(ntotal);
  READ1(metric_type);
  if (metric_type == 0) {
    idx->refine_index_.reset(new faiss::IndexFlatLL2(d));
  } else if (metric_type == 1) {
    idx->refine_index_.reset(new faiss::IndexFlatLIP(d));
  } else {
    // printf("read_hakes_full_vecs: metric type not supported\n");
    return false;
  }
  idx->refine_index_->ntotal = ntotal;
  idx->refine_index_->is_trained = true;
  idx->refine_index_->code_size = d * sizeof(float);
  idx->refine_index_->codes.reserve(ntotal * idx->refine_index_->code_size * 2);
  READXBVECTOR(idx->refine_index_->codes);
  assert(idx->refine_index_->codes.size() ==
         idx->refine_index_->ntotal * idx->refine_index_->code_size);
  return true;
}

void save_hakes_findex(hakes::IOWriter* ff, const HakesIndex* idx) {
  write_hakes_pretransform(ff, &idx->vts_);
  printf("write_hakes_pretransform\n");
  write_hakes_ivf(ff, idx);
  printf("write_hakes_ivf\n");
  write_hakes_pq(ff, idx);
  printf("write_hakes_pq\n");
  write_hakes_compressed_vecs(ff, idx);
  printf("write_hakes_compressed_vecs\n");
}

void save_hakes_rindex(hakes::IOWriter* rf, const HakesIndex* idx) {
  idx->mapping_->save(rf);
  write_hakes_full_vecs(rf, idx);
}

void save_hakes_uindex(hakes::IOWriter* uf, const HakesIndex* idx) {
  if (!idx->has_q_index_) {
    return;
  }
  write_hakes_pretransform(uf, &idx->q_vts_);
  printf("write_hakes_pretransform\n");
  write_hakes_ivf(uf, idx, true);
  printf("write_hakes_ivf\n");
  write_hakes_pq(uf, idx, true);
  printf("write_hakes_pq\n");
}

void save_hakes_index(hakes::IOWriter* ff, hakes::IOWriter* rf,
                      const HakesIndex* idx) {
  if (idx->base_index_) {
    save_hakes_findex(ff, idx);
  }
  if (idx->refine_index_) {
    save_hakes_rindex(rf, idx);
  }
}

bool load_hakes_findex(hakes::IOReader* ff, HakesIndex* idx) {
  if (!read_hakes_pretransform(ff, &idx->vts_)) {
    return false;
  }
  idx->base_index_.reset(new faiss::IndexIVFPQFastScanL());
  bool success = read_hakes_ivf(ff, idx) && read_hakes_pq(ff, idx) &&
                 read_hakes_compressed_vecs(ff, idx);
  if (!success) {
    return false;
  }
  // default field settings
  idx->base_index_->by_residual = false;
  idx->base_index_->nprobe = 1;
  idx->base_index_->own_fields = true;
  idx->base_index_->is_trained = true;
  assert(idx->base_index_->d == idx->vts_.back()->d_out);
  idx->base_index_->bbs = 32;
  idx->base_index_->M = idx->base_index_->pq.M;
  idx->base_index_->M2 = (idx->base_index_->M + 1) / 2 * 2;
  idx->base_index_->implem = 0;
  idx->base_index_->qbs2 = 0;
  idx->base_index_->nbits = idx->base_index_->pq.nbits;
  idx->base_index_->ksub = 1 << idx->base_index_->pq.nbits;
  idx->base_index_->code_size = idx->base_index_->pq.code_size;
  idx->base_index_->init_code_packer();
  idx->base_index_->precompute_table();
  printf("code size: %ld\n", idx->base_index_->code_size);
  printf("bbs: %d\n", idx->base_index_->bbs);
  printf("M2: %ld\n", idx->base_index_->M2);
  printf("implem: %d\n", idx->base_index_->implem);
  printf("qbs2: %ld\n", idx->base_index_->qbs2);
  printf("nbits: %ld\n", idx->base_index_->nbits);
  printf("ksub: %ld\n", idx->base_index_->ksub);
  printf("nlist: %ld\n", idx->base_index_->nlist);
  printf("ntotal: %ld\n", idx->base_index_->ntotal);
  printf("d: %d\n", idx->base_index_->d);

  return true;
}

bool load_hakes_rindex(hakes::IOReader* rf, HakesIndex* idx) {
  idx->mapping_.reset(new faiss::IDMapImpl());
  printf("load_hakes_rindex: load mapping\n");
  idx->mapping_->load(rf);
  printf("load_hakes_rindex: mapping size: %ld\n", idx->mapping_->size());
  return read_hakes_full_vecs(rf, idx);
}

bool load_hakes_index(hakes::IOReader* ff, hakes::IOReader* rf, HakesIndex* idx,
                      int mode) {
  if (mode < 2) {
    // filter index shall be loaded
    printf("load_hakes_index: load filter index\n");
    if (!load_hakes_findex(ff, idx)) {
      return false;
    }
    if (mode == 1) {
      // only filter index is needed
      return true;
    }
  }

  if (rf) {
    printf("load_hakes_index: load refine index\n");
    return load_hakes_rindex(rf, idx);
  } else {
    printf("load_hakes_index: no refine index\n");
    idx->mapping_.reset(new faiss::IDMapImpl());
    int refine_d =
        (idx->vts_.empty()) ? idx->base_index_->d : idx->vts_.front()->d_in;
    idx->refine_index_.reset(
        new faiss::IndexFlatL(refine_d, idx->base_index_->metric_type));
    return true;
  }
}

bool load_hakes_params(hakes::IOReader* f, HakesIndex* idx) {
  if (!read_hakes_pretransform(f, &idx->vts_)) {
    return false;
  }
  idx->base_index_.reset(new faiss::IndexIVFPQFastScanL());
  bool success = read_hakes_ivf(f, idx) && read_hakes_pq(f, idx);
  if (!success) {
    return false;
  }
  // default field settings
  idx->base_index_->by_residual = false;
  idx->base_index_->nprobe = 1;
  idx->base_index_->own_fields = true;
  idx->base_index_->is_trained = true;
  assert(idx->base_index_->d == idx->vts_.back()->d_out);
  idx->base_index_->bbs = 32;
  idx->base_index_->M = idx->base_index_->pq.M;
  idx->base_index_->M2 = (idx->base_index_->M + 1) / 2 * 2;
  idx->base_index_->implem = 0;
  idx->base_index_->qbs2 = 0;
  idx->base_index_->nbits = idx->base_index_->pq.nbits;
  idx->base_index_->ksub = 1 << idx->base_index_->pq.nbits;
  idx->base_index_->code_size = idx->base_index_->pq.code_size;
  idx->base_index_->precompute_table();

  auto code_packer = idx->base_index_->get_CodePacker();
  auto il = new BlockInvertedListsL(idx->base_index_->nlist, code_packer->nvec,
                                    code_packer->block_size);
  il->init(nullptr);
  idx->base_index_->invlists = il;
  idx->base_index_->own_invlists = true;
  idx->base_index_->init_code_packer();
  delete code_packer;
  return true;
}

void save_hakes_params(hakes::IOWriter* f, const HakesIndex* idx) {
  if (idx->base_index_ == nullptr) {
    printf("save_hakes_params: filter index is not initialized\n");
    return;
  }
  write_hakes_pretransform(f, &idx->vts_);
  write_hakes_ivf(f, idx);
  write_hakes_pq(f, idx);
}

void save_init_params(hakes::IOWriter* f,
                      const std::vector<VectorTransform*>* vts,
                      ProductQuantizer* pq, IndexFlat* ivf) {
  write_hakes_pretransform(f, vts);
  // save ivf centroids
  int32_t d = ivf->d;
  uint64_t ntotal = 0;
  uint8_t metric_type = (ivf->metric_type == METRIC_L2) ? 0 : 1;
  int32_t nlist = ivf->ntotal;
  WRITE1(d);
  WRITE1(ntotal);
  WRITE1(metric_type);
  WRITE1(nlist);
  size_t code_size = nlist * d * sizeof(float);
  WRITEANDCHECK(ivf->codes.data(), code_size);
  printf("write_hakes_ivf: d: %d, ntotal: %ld, nlist: %d\n", d, ntotal, nlist);

  // save pq codebook
  d = pq->d;
  int32_t M = pq->M;
  int32_t nbits = pq->nbits;
  WRITE1(d);
  WRITE1(M);
  WRITE1(nbits);
  size_t centroids_size = pq->M * pq->ksub * pq->dsub;
  WRITEANDCHECK(pq->centroids.data(), centroids_size);
}

}  // namespace faiss
