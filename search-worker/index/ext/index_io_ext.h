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
#include "search-worker/index/ext/HakesFlatIndex.h"
#include "search-worker/index/ext/HakesIndex.h"
#include "search-worker/index/ext/IdMap.h"
#include "utils/io.h"

namespace faiss {
// the io utilities are not thread safe, needs external synchronization

bool load_hakes_params(hakes::IOReader* f, HakesIndex* idx);
void save_hakes_params(hakes::IOWriter* f, const HakesIndex* idx);

bool load_hakes_findex(hakes::IOReader* ff, HakesIndex* idx);
bool load_hakes_rindex(hakes::IOReader* rf, HakesIndex* idx);
bool load_hakes_index(hakes::IOReader* ff, hakes::IOReader* rf, HakesIndex* idx,
                      int mode);

void save_hakes_findex(hakes::IOWriter* ff, const HakesIndex* idx);

void save_hakes_rindex(hakes::IOWriter* rf, const HakesIndex* idx);

void save_hakes_uindex(hakes::IOWriter* uf, const HakesIndex* idx);

void save_hakes_index(hakes::IOWriter* ff, hakes::IOWriter* rf,
                      const HakesIndex* idx);

bool load_hakes_flatindex(hakes::IOReader* f, HakesFlatIndex* idx);
void save_hakes_flatindex(hakes::IOWriter* f, const HakesFlatIndex* idx);

void save_init_params(hakes::IOWriter* f,
                      const std::vector<VectorTransform*>* vts,
                      ProductQuantizer* pq, IndexFlat* ivf);

}  // namespace faiss

#endif  // HAKES_SEARCHWORKER_INDEX_EXT_INDEXIOEXT_H_
