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

#ifndef HAKES_SEARCHWORKER_INDEX_EXT_INDEXIOEXT_H_
#define HAKES_SEARCHWORKER_INDEX_EXT_INDEXIOEXT_H_

#include "search-worker/index/IndexFlat.h"
#include "search-worker/index/VectorTransform.h"
#include "search-worker/index/ext/HakesIndex.h"
#include "search-worker/index/ext/IdMap.h"
#include "search-worker/index/impl/io.h"

namespace faiss {

/**
 * from index_io.h
 */
// The read_index flags are implemented only for a subset of index types.
const int IO_FLAG_READ_ONLY = 2;
// strip directory component from ondisk filename, and assume it's in
// the same directory as the index file
const int IO_FLAG_ONDISK_SAME_DIR = 4;
// don't load IVF data to RAM, only list sizes
const int IO_FLAG_SKIP_IVF_DATA = 8;
// don't initialize precomputed table after loading
const int IO_FLAG_SKIP_PRECOMPUTE_TABLE = 16;
// try to memmap data (useful to load an ArrayInvertedLists as an
// OnDiskInvertedLists)
const int IO_FLAG_MMAP = IO_FLAG_SKIP_IVF_DATA | 0x646f0000;

/**
 * from index_io.h
 */

struct Index;

struct StringIOReader : IOReader {
  StringIOReader(const char* data, size_t data_len)
      : data(data), data_len(data_len) {}
  const char* data;
  size_t data_len;
  size_t rp = 0;
  size_t operator()(void* ptr, size_t size, size_t nitems) override;
};

struct StringIOWriter : IOWriter {
  std::string data;
  size_t operator()(const void* ptr, size_t size, size_t nitems) override;
};

// the io utilities are not thread safe, needs external synchronization

// void write_index_ext(const Index* idx, const char* fname);
void write_index_ext(const Index* idx, IOWriter* f);

Index* read_index_ext(IOReader* f, int io_flags = 0);

bool write_hakes_vt_quantizers(IOWriter* f,
                               const std::vector<VectorTransform*>& pq_vts,
                               const IndexFlat* ivf_centroids,
                               const ProductQuantizer* pq);

Index* load_hakes_vt_quantizers(IOReader* f, MetricType metric,
                                std::vector<VectorTransform*>* pq_vts);

bool write_hakes_index_params(IOWriter* f,
                              const std::vector<VectorTransform*>& pq_vts,
                              const std::vector<VectorTransform*>& ivf_vts,
                              const IndexFlatL* ivf_centroids,
                              const ProductQuantizer* pq);

HakesIndex* load_hakes_index_params(IOReader* f);

bool load_hakes_index_single_file(IOReader* f, HakesIndex* idx);
bool write_hakes_index_single_file(IOWriter* f, const HakesIndex* idx);

}  // namespace faiss

#endif  // HAKES_SEARCHWORKER_INDEX_EXT_INDEXIOEXT_H_
