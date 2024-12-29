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

namespace faiss {

/**
 * streamlined functions from index_read.cpp and index_write.cpp
 */

namespace {

void read_index_header(Index* idx, hakes::IOReader* f) {
  READ1(idx->d);
  READ1(idx->ntotal);
  idx_t dummy;
  READ1(dummy);
  READ1(dummy);
  READ1(idx->is_trained);
  READ1(idx->metric_type);
  if (idx->metric_type > 1) {
    READ1(idx->metric_arg);
  }
  idx->verbose = false;
}

static void read_ArrayInvertedLists_sizes(hakes::IOReader* f,
                                          std::vector<size_t>& sizes) {
  uint32_t list_type;
  READ1(list_type);
  if (list_type == fourcc("full")) {
    size_t os = sizes.size();
    READVECTOR(sizes);
    // FAISS_THROW_IF_NOT(os == sizes.size());
    assert(os == sizes.size());
  } else if (list_type == fourcc("sprs")) {
    std::vector<size_t> idsizes;
    READVECTOR(idsizes);
    for (size_t j = 0; j < idsizes.size(); j += 2) {
      // FAISS_THROW_IF_NOT(idsizes[j] < sizes.size());
      assert(idsizes[j] < sizes.size());
      sizes[idsizes[j]] = idsizes[j + 1];
    }
  } else {
    // FAISS_THROW_FMT(
    //         "list_type %ud (\"%s\") not recognized",
    //         list_type,
    //         fourcc_inv_printable(list_type).c_str());
    assert(!"list_type not recognized");
  }
}

InvertedLists* read_BlockInvertedLists(hakes::IOReader* f) {
  size_t nlist, code_size, n_per_block, block_size;
  std::vector<int> load_list;
  READ1(nlist);
  READ1(code_size);
  READ1(n_per_block);
  READ1(block_size);
  READVECTOR(load_list);

  BlockInvertedListsL* il =
      new BlockInvertedListsL(nlist, n_per_block, block_size);

  for (size_t i = 0; i < il->nlist; i++) {
    il->lists_[i].read(f);
  }

  // set load list and active, capacity is 0 to keep current loaded lists
  il->init(nullptr, load_list);

  return il;
}

InvertedLists* read_InvertedLists(hakes::IOReader* f, int io_flags) {
  uint32_t h;
  READ1(h);
  if (h == fourcc("il00")) {
    // fprintf(stderr,
    //         "read_InvertedLists:"
    //         " WARN! inverted lists not stored with IVF object\n");
    assert(!"inverted lists not stored with IVF object");
    return nullptr;
  } else if (h == fourcc("ilar") && !(io_flags & IO_FLAG_SKIP_IVF_DATA)) {
    auto ails = new ArrayInvertedLists(0, 0);
    READ1(ails->nlist);
    READ1(ails->code_size);
    ails->ids.resize(ails->nlist);
    ails->codes.resize(ails->nlist);
    std::vector<size_t> sizes(ails->nlist);
    read_ArrayInvertedLists_sizes(f, sizes);
    for (size_t i = 0; i < ails->nlist; i++) {
      ails->ids[i].resize(sizes[i]);
      ails->codes[i].resize(sizes[i] * ails->code_size);
    }
    for (size_t i = 0; i < ails->nlist; i++) {
      size_t n = ails->ids[i].size();
      if (n > 0) {
        READANDCHECK(ails->codes[i].data(), n * ails->code_size);
        READANDCHECK(ails->ids[i].data(), n);
      }
    }
    return ails;

  } else if (h == fourcc("ilar") && (io_flags & IO_FLAG_SKIP_IVF_DATA)) {
    // // code is always ilxx where xx is specific to the type of invlists we
    // // want so we get the 16 high bits from the io_flag and the 16 low bits
    // // as "il"
    // int h2 = (io_flags & 0xffff0000) | (fourcc("il__") & 0x0000ffff);
    // size_t nlist, code_size;
    // READ1(nlist);
    // READ1(code_size);
    // std::vector<size_t> sizes(nlist);
    // read_ArrayInvertedLists_sizes(f, sizes);
    // return InvertedListsIOHook::lookup(h2)->read_ArrayInvertedLists(
    //         f, io_flags, nlist, code_size, sizes);
    assert(!"ilar read skip ivf data not implemented");
    return nullptr;
  } else {
    assert(h == fourcc("ibll"));
    // return InvertedListsIOHook::lookup(h)->read(f, io_flags);
    return read_BlockInvertedLists(f);
  }
}

void read_ProductQuantizer(ProductQuantizer* pq, hakes::IOReader* f) {
  READ1(pq->d);
  READ1(pq->M);
  READ1(pq->nbits);
  pq->set_derived_values();
  READVECTOR(pq->centroids);
}

void write_index_header(const Index* idx, hakes::IOWriter* f) {
  WRITE1(idx->d);
  WRITE1(idx->ntotal);
  idx_t dummy = 1 << 20;
  WRITE1(dummy);
  WRITE1(dummy);
  WRITE1(idx->is_trained);
  WRITE1(idx->metric_type);
  if (idx->metric_type > 1) {
    WRITE1(idx->metric_arg);
  }
}

void write_BlockInvertedLists(const InvertedLists* ils_in, hakes::IOWriter* f) {
  uint32_t h = fourcc("ibll");
  WRITE1(h);
  const BlockInvertedListsL* il =
      dynamic_cast<const BlockInvertedListsL*>(ils_in);
  WRITE1(il->nlist);
  WRITE1(il->code_size);
  WRITE1(il->n_per_block_);
  WRITE1(il->block_size_);

  // write the load list
  WRITEVECTOR(il->load_list_);

  for (size_t i = 0; i < il->nlist; i++) {
    il->lists_[i].write(f);
  }
}

void write_InvertedLists(const InvertedLists* ils, hakes::IOWriter* f) {
  if (ils == nullptr) {
    uint32_t h = fourcc("il00");
    WRITE1(h);
  } else if (const auto& ails = dynamic_cast<const ArrayInvertedLists*>(ils)) {
    uint32_t h = fourcc("ilar");
    WRITE1(h);
    WRITE1(ails->nlist);
    WRITE1(ails->code_size);
    // here we store either as a full or a sparse data buffer
    size_t n_non0 = 0;
    for (size_t i = 0; i < ails->nlist; i++) {
      if (ails->ids[i].size() > 0) n_non0++;
    }
    if (n_non0 > ails->nlist / 2) {
      uint32_t list_type = fourcc("full");
      WRITE1(list_type);
      std::vector<size_t> sizes;
      for (size_t i = 0; i < ails->nlist; i++) {
        sizes.push_back(ails->ids[i].size());
      }
      WRITEVECTOR(sizes);
    } else {
      int list_type = fourcc("sprs");  // sparse
      WRITE1(list_type);
      std::vector<size_t> sizes;
      for (size_t i = 0; i < ails->nlist; i++) {
        size_t n = ails->ids[i].size();
        if (n > 0) {
          sizes.push_back(i);
          sizes.push_back(n);
        }
      }
      WRITEVECTOR(sizes);
    }
    // make a single contiguous data buffer (useful for mmapping)
    for (size_t i = 0; i < ails->nlist; i++) {
      size_t n = ails->ids[i].size();
      if (n > 0) {
        WRITEANDCHECK(ails->codes[i].data(), n * ails->code_size);
        WRITEANDCHECK(ails->ids[i].data(), n);
      }
    }

  } else {
    // // printf("type name: %s\n", typeid(*ils).name());
    // InvertedListsIOHook::lookup_classname(typeid(*ils).name())
    //         ->write(ils, f);
    write_BlockInvertedLists(ils, f);
  }
}

void write_ProductQuantizer(const ProductQuantizer* pq, hakes::IOWriter* f) {
  WRITE1(pq->d);
  WRITE1(pq->M);
  WRITE1(pq->nbits);
  WRITEVECTOR(pq->centroids);
}

void write_ivfl_header(const IndexIVFL* ivf, hakes::IOWriter* f) {
  write_index_header(ivf, f);
  WRITE1(ivf->nlist);
  WRITE1(ivf->nprobe);
  // subclasses write by_residual (some of them support only one setting of
  // by_residual).
  write_index_ext(ivf->quantizer, f);
}

void read_ivfl_header(IndexIVFL* ivf, hakes::IOReader* f,
                      std::vector<std::vector<idx_t>>* ids = nullptr) {
  read_index_header(ivf, f);
  READ1(ivf->nlist);
  READ1(ivf->nprobe);
  ivf->quantizer = read_index_ext(f);
  ivf->own_fields = true;
  if (ids) {  // used in legacy "Iv" formats
    ids->resize(ivf->nlist);
    for (size_t i = 0; i < ivf->nlist; i++) READVECTOR((*ids)[i]);
  }
}

void write_refine_map(const std::unordered_map<idx_t, idx_t>& m,
                      hakes::IOWriter* f) {
  std::vector<std::pair<idx_t, idx_t>> v;
  v.resize(m.size());
  std::copy(m.begin(), m.end(), v.begin());
  WRITEVECTOR(v);
}

void read_refine_map(std::unordered_map<idx_t, idx_t>* m, hakes::IOReader* f) {
  std::vector<std::pair<idx_t, idx_t>> v;
  READVECTOR(v);
  m->clear();
  m->reserve(v.size());
  for (auto& p : v) {
    (*m)[p.first] = p.second;
  }
}

void read_InvertedLists(IndexIVFL* ivf, hakes::IOReader* f, int io_flags) {
  InvertedLists* ils = read_InvertedLists(f, io_flags);
  if (ils) {
    // FAISS_THROW_IF_NOT(ils->nlist == ivf->nlist);
    assert(ils->nlist == ivf->nlist);
    // FAISS_THROW_IF_NOT(ils->code_size == InvertedLists::INVALID_CODE_SIZE ||
    //                    ils->code_size == ivf->code_size);
    assert(ils->code_size == InvertedLists::INVALID_CODE_SIZE ||
           ils->code_size == ivf->code_size);
  }
  ivf->invlists = ils;
  ivf->own_invlists = true;
}
}  // anonymous namespace

void write_index_ext(const Index* idx, hakes::IOWriter* f) {
  // check the new types before falling back to the original implementation
  if (const IndexFlatL* idxf = dynamic_cast<const IndexFlatL*>(idx)) {
    // same impl as IndexFlat, but with different fourcc for load
    uint32_t h = fourcc(idxf->metric_type == METRIC_INNER_PRODUCT ? "IlFI"
                        : idxf->metric_type == METRIC_L2          ? "IlF2"
                                                                  : "IlFl");
    WRITE1(h);
    write_index_header(idx, f);
    WRITEXBVECTOR(idxf->codes);
  } else if (const IndexRefineL* idxrf =
                 dynamic_cast<const IndexRefineL*>(idx)) {
    // Here we also need to store the mapping
    uint32_t h = fourcc("IlRF");
    WRITE1(h);
    // additionally store the two mapping
    write_refine_map(idxrf->off_to_idx, f);
    write_refine_map(idxrf->idx_to_off, f);

    write_index_header(idxrf, f);
    write_index_ext(idxrf->base_index, f);
    write_index_ext(idxrf->refine_index, f);
    WRITE1(idxrf->k_factor);
  } else if (const IndexIVFPQFastScanL* ivpq_2 =
                 dynamic_cast<const IndexIVFPQFastScanL*>(idx)) {
    // here we need to use the block inverted list locking IO
    uint32_t h = fourcc("IlPf");
    WRITE1(h);
    write_ivfl_header(ivpq_2, f);
    WRITE1(ivpq_2->by_residual);
    WRITE1(ivpq_2->code_size);
    WRITE1(ivpq_2->bbs);
    WRITE1(ivpq_2->M2);
    WRITE1(ivpq_2->implem);
    WRITE1(ivpq_2->qbs2);
    write_ProductQuantizer(&ivpq_2->pq, f);
    write_InvertedLists(ivpq_2->invlists, f);
  } else {
    // write_index(idx, f);
    assert(!"other index not supported to write in hakes");
  }
}

Index* read_index_ext(hakes::IOReader* f, int io_flags) {
  Index* idx = nullptr;
  uint32_t h;
  READ1(h);
  if (h == fourcc("IlFI") || h == fourcc("IlF2") || h == fourcc("IlFl")) {
    IndexFlatL* idxf;
    if (h == fourcc("IlFI")) {
      idxf = new IndexFlatLIP();
    } else if (h == fourcc("IlF2")) {
      idxf = new IndexFlatLL2();
    } else {
      idxf = new IndexFlatL();
    }
    read_index_header(idxf, f);
    idxf->code_size = idxf->d * sizeof(float);
    idxf->codes.reserve(idxf->ntotal * idxf->code_size * 2);
    READXBVECTOR(idxf->codes);
    // FAISS_THROW_IF_NOT(idxf->codes.size() == idxf->ntotal * idxf->code_size);
    assert(idxf->codes.size() == idxf->ntotal * idxf->code_size);
    idx = idxf;
  } else if (h == fourcc("IlRF")) {
    IndexRefineL* idxrf = new IndexRefineL();
    read_refine_map(&idxrf->off_to_idx, f);
    read_refine_map(&idxrf->idx_to_off, f);

    read_index_header(idxrf, f);
    idxrf->base_index = read_index_ext(f, io_flags);
    // print memory after loading base index
    // printf("Memory after loading base index: %ld\n",
    //        getCurrentRSS() / 1024 / 1024);
    idxrf->refine_index = read_index_ext(f, io_flags);
    READ1(idxrf->k_factor);
    if (dynamic_cast<IndexFlatL*>(idxrf->refine_index)) {
      // then make a RefineFlat with it
      IndexRefineL* idxrf_old = idxrf;
      idxrf = new IndexRefineFlatL();
      *idxrf = *idxrf_old;
      delete idxrf_old;
    }
    idxrf->own_fields = true;
    idxrf->own_refine_index = true;
    idx = idxrf;
    // printf("Memory after loading refine index: %ld\n",
    //        getCurrentRSS() / 1024 / 1024);
  } else if (h == fourcc("IlPf")) {
    IndexIVFPQFastScanL* ivpq = new IndexIVFPQFastScanL();
    read_ivfl_header(ivpq, f);
    READ1(ivpq->by_residual);
    READ1(ivpq->code_size);
    READ1(ivpq->bbs);
    READ1(ivpq->M2);
    READ1(ivpq->implem);
    READ1(ivpq->qbs2);
    read_ProductQuantizer(&ivpq->pq, f);
    read_InvertedLists(ivpq, f, io_flags);
    ivpq->precompute_table();

    const auto& pq = ivpq->pq;
    ivpq->M = pq.M;
    ivpq->nbits = pq.nbits;
    ivpq->ksub = (1 << pq.nbits);
    ivpq->code_size = pq.code_size;
    // printf("code_size: %ld\n", ivpq->code_size);
    ivpq->init_code_packer();

    idx = ivpq;
  } else {
    // idx = read_index(f, io_flags);
    assert(!"other index not supported to read in hakes");
  }
  return idx;
}

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
  return true;
}

IndexFlatL* read_hakes_ivf(hakes::IOReader* f, MetricType metric,
                           bool* use_residual) {
  // open ivf file
  int32_t by_residual;
  READ1(by_residual);
  *use_residual = (by_residual == 1);
  int32_t nlist, d;
  READ1(nlist);
  READ1(d);
  IndexFlatL* ivf = new IndexFlatL(d, metric);
  size_t code_size = nlist * d * sizeof(float);
  std::vector<uint8_t> codes(code_size);
  READANDCHECK(codes.data(), code_size);
  // printf("codes read size: %ld\n", code_size);
  ivf->codes = std::move(codes);
  ivf->is_trained = true;
  ivf->ntotal = nlist;
  return ivf;
}

bool read_hakes_pq(hakes::IOReader* f, ProductQuantizer* pq) {
  // open pq file
  int32_t d, M, nbits;
  READ1(d);
  READ1(M);
  READ1(nbits);
  pq->d = d;
  pq->M = M;
  pq->nbits = nbits;
  pq->set_derived_values();
  pq->train_type = ProductQuantizer::Train_hot_start;
  size_t centroids_size = pq->M * pq->ksub * pq->dsub;
  READANDCHECK(pq->centroids.data(), centroids_size);
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

bool write_hakes_ivf(hakes::IOWriter* f, const Index* idx, bool use_residual) {
  const IndexFlatL* quantizer = dynamic_cast<const IndexFlatL*>(idx);
  if (quantizer == nullptr) {
    // printf("write_hakes_ivf: Only IndexFlatL is supported\n");
    return false;
  }
  int32_t by_residual = use_residual ? 1 : 0;
  WRITE1(by_residual);
  int32_t nlist = quantizer->ntotal;
  int32_t d = quantizer->d;
  WRITE1(nlist);
  WRITE1(d);
  size_t code_size = nlist * d * sizeof(float);
  WRITEANDCHECK(quantizer->codes.data(), code_size);
  return true;
}

bool write_hakes_pq(hakes::IOWriter* f, const ProductQuantizer& pq) {
  int32_t d = pq.d;
  int32_t M = pq.M;
  int32_t nbits = pq.nbits;
  WRITE1(d);
  WRITE1(M);
  WRITE1(nbits);
  size_t centroids_size = M * pq.ksub * pq.dsub;
  WRITEANDCHECK(pq.centroids.data(), centroids_size);
  return true;
}

std::unordered_map<faiss::idx_t, faiss::idx_t> read_pa_mapping(
    hakes::IOReader* f) {
  // open pa mapping file
  std::vector<std::pair<idx_t, idx_t>> v;
  READVECTOR(v);
  std::unordered_map<faiss::idx_t, faiss::idx_t> pa_mapping;
  pa_mapping.reserve(v.size());
  for (auto& p : v) {
    pa_mapping[p.first] = p.second;
  }
  return pa_mapping;
}

bool write_hakes_ivf2(hakes::IOWriter* f, const IndexFlat* idx,
                      bool use_residual) {
  int32_t by_residual = use_residual ? 1 : 0;
  WRITE1(by_residual);
  int32_t nlist = idx->ntotal;
  int32_t d = idx->d;
  WRITE1(nlist);
  WRITE1(d);
  size_t code_size = nlist * d * sizeof(float);
  WRITEANDCHECK(idx->codes.data(), code_size);
  return true;
}

bool write_hakes_vt_quantizers(hakes::IOWriter* f,
                               const std::vector<VectorTransform*>& pq_vts,
                               const IndexFlat* ivf_centroids,
                               const ProductQuantizer* pq) {
  if ((ivf_centroids == nullptr) || (pq == nullptr)) {
    // printf("write_hakes_vt_quantizers: ivf_centroids or pq is nullptr\n");
    return false;
  }
  // write pq vts
  if (!write_hakes_pretransform(f, &pq_vts)) {
    // printf("write_hakes_vt_quantizers: write pq vts failed\n");
    return false;
  }

  // write ivf
  if (!write_hakes_ivf2(f, ivf_centroids, false)) {
    // printf("write_hakes_vt_quantizers: write ivf failed\n");
    return false;
  }

  // write pq
  if (!write_hakes_pq(f, *pq)) {
    // printf("write_hakes_vt_quantizers: write pq failed\n");
    return false;
  }

  return true;
}

Index* load_hakes_vt_quantizers(hakes::IOReader* f, MetricType metric,
                                std::vector<VectorTransform*>* pq_vts) {
  assert(pq_vts != nullptr);

  // load pq vts
  read_hakes_pretransform(f, pq_vts);

  // load ivf
  bool use_residual;
  IndexFlatL* quantizer = read_hakes_ivf(f, metric, &use_residual);

  // load pq
  IndexIVFPQFastScanL* base_index = new IndexIVFPQFastScanL();
  read_hakes_pq(f, &base_index->pq);
  // read_index_header
  // use the opq vt to get the d
  base_index->d = pq_vts->back()->d_out;
  base_index->metric_type = metric;
  // read_ivfl_header
  base_index->nlist = quantizer->ntotal;
  base_index->quantizer = quantizer;
  base_index->own_fields = true;
  // read_index_ext IVFPQFastScanL branch
  base_index->by_residual = use_residual;
  base_index->code_size = base_index->pq.code_size;
  // printf("code size: %ld\n", base_index->code_size);
  base_index->bbs = 32;
  base_index->M = base_index->pq.M;
  base_index->M2 = (base_index->M + 1) / 2 * 2;
  // printf("M2: %ld\n", base_index->M2);
  base_index->implem = 0;
  base_index->qbs2 = 0;

  // read_InvertedLists
  CodePacker* code_packer = base_index->get_CodePacker();
  BlockInvertedListsL* il = new BlockInvertedListsL(
      base_index->nlist, code_packer->nvec, code_packer->block_size);
  il->init(nullptr, std::vector<int>());
  base_index->invlists = il;
  base_index->own_invlists = true;
  base_index->nbits = base_index->pq.nbits;
  base_index->ksub = 1 << base_index->pq.nbits;
  base_index->code_size = base_index->pq.code_size;
  base_index->init_code_packer();

  base_index->is_trained = true;

  delete code_packer;
  return base_index;
}

// |---#vts----|---vts---|---ivf---|---pq---|
bool write_hakes_index_params(hakes::IOWriter* f,
                              const std::vector<VectorTransform*>& vts,
                              const std::vector<VectorTransform*>& ivf_vts,
                              const IndexFlatL* ivf_centroids,
                              const ProductQuantizer* pq) {
  // write vts
  if (!write_hakes_pretransform(f, &vts)) {
    // printf("write_hakes_index_params: write pq vts failed\n");
    return false;
  }

  // write ivf vts
  if (!write_hakes_pretransform(f, &ivf_vts)) {
    // printf("write_hakes_index_params: write ivf vts failed\n");
    return false;
  }

  // write ivf
  if (!write_hakes_ivf(f, ivf_centroids, false)) {
    // printf("write_hakes_index_params: write ivf failed\n");
    return false;
  }

  // write pq
  if (!write_hakes_pq(f, *pq)) {
    // printf("write_hakes_index_params: write pq failed\n");
    return false;
  }
  return true;
}

// many fields of the returned index is not initialized. just the parameters
HakesIndex* load_hakes_index_params(hakes::IOReader* f) {
  HakesIndex* index = new HakesIndex();

  // load pq vts
  read_hakes_pretransform(f, &index->vts_);

  // load ivf vts
  // read_hakes_pretransform(f, &index->ivf_vts_);

  // load ivf
  bool use_residual;
  IndexFlatL* quantizer =
      read_hakes_ivf(f, METRIC_INNER_PRODUCT, &use_residual);

  // load pq

  // assemble
  IndexIVFPQFastScanL* base_index = new IndexIVFPQFastScanL();
  read_hakes_pq(f, &base_index->pq);
  base_index->d = index->vts_.back()->d_out;
  base_index->metric_type = METRIC_INNER_PRODUCT;
  base_index->nlist = quantizer->ntotal;
  base_index->quantizer = quantizer;
  base_index->own_fields = true;
  base_index->by_residual = use_residual;
  base_index->code_size = base_index->pq.code_size;
  base_index->bbs = 32;
  base_index->M = base_index->pq.M;
  base_index->M2 = (base_index->M + 1) / 2 * 2;
  base_index->implem = 0;
  base_index->qbs2 = 0;

  index->base_index_.reset(base_index);
  index->cq_ = index->base_index_->quantizer;
  return index;
}

bool load_hakes_index_single_file(hakes::IOReader* f, HakesIndex* idx) {
  if (!read_hakes_pretransform(f, &idx->vts_)) {
    return false;
  }
  idx->base_index_.reset(
      dynamic_cast<faiss::IndexIVFPQFastScanL*>(read_index_ext(f)));
  idx->mapping_->load(f);
  idx->refine_index_.reset(dynamic_cast<faiss::IndexFlatL*>(read_index_ext(f)));
  return true;
}

bool write_hakes_index_single_file(hakes::IOWriter* f, const HakesIndex* idx) {
  if (!write_hakes_pretransform(f, &idx->vts_)) {
    return false;
  }
  write_index_ext(idx->base_index_.get(), f);
  if (!idx->mapping_->save(f)) {
    return false;
  }
  write_index_ext(idx->refine_index_.get(), f);
  return true;
}

bool load_hakes_findex(hakes::IOReader* ff, HakesIndex* idx, bool keep_pa) {
  if (!read_hakes_pretransform(ff, &idx->vts_)) {
    return false;
  }
  idx->base_index_.reset(
      dynamic_cast<faiss::IndexIVFPQFastScanL*>(read_index_ext(ff)));
  return (idx->base_index_ != nullptr);
}

bool load_hakes_rindex(hakes::IOReader* rf, int d, HakesIndex* idx,
                       bool keep_pa) {
  idx->mapping_->load(rf);
  idx->refine_index_.reset(
      dynamic_cast<faiss::IndexFlatL*>(read_index_ext(rf)));
  return true;
}

bool load_hakes_index(hakes::IOReader* ff, hakes::IOReader* rf, HakesIndex* idx,
                      bool keep_pa) {
  if (!load_hakes_findex(ff, idx, keep_pa)) {
    return false;
  }
  if (rf) {
    return load_hakes_rindex(rf, idx->vts_.back()->d_out, idx, keep_pa);
  } else {
    idx->mapping_.reset(new faiss::IDMapImpl());
    int refine_d = (idx->vts_.empty()) ? idx->base_index_->d
                                       : idx->vts_.front()->d_in;
    idx->refine_index_.reset(new faiss::IndexFlatL(refine_d, idx->base_index_->metric_type));
    return true;
  }
}

}  // namespace faiss
