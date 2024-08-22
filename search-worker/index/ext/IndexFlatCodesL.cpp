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

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "IndexFlatCodesL.h"

#include <cassert>
#include <chrono>
#include <cstring>

#include "search-worker/index/impl/CodePacker.h"

namespace faiss {

void IndexFlatCodesL::add(idx_t n, const float* x) {
  // std::chrono::high_resolution_clock::time_point add_start =
  //     std::chrono::high_resolution_clock::now();

  // FAISS_THROW_IF_NOT(is_trained);
  assert(is_trained);
  if (n == 0) {
    return;
  }
  // exclusive lock
  // std::unique_lock lock(this->codes_mutex);
  pthread_rwlock_wrlock(&this->codes_mutex);

  codes.resize((ntotal + n) * code_size);
  // std::cout << "resize from " << codes.size() << " to "
  //           << (ntotal + n) * code_size << "\n";

  // std::chrono::high_resolution_clock::time_point resize_end =
  //     std::chrono::high_resolution_clock::now();

  sa_encode(n, x, codes.data() + (ntotal * code_size));
  ntotal += n;

  pthread_rwlock_unlock(&this->codes_mutex);

  // std::chrono::high_resolution_clock::time_point add_end =
  //     std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> resize_diff = resize_end - add_start;
  // std::chrono::duration<double> add_diff = add_end - resize_end;
  // std::cout << "resize time (s): " << resize_diff.count() * 1000000
  //           << " encode time (s): " << add_diff.count() * 1000000 << "\n";
}

void IndexFlatCodesL::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
  // std::chrono::high_resolution_clock::time_point add_start =
  //     std::chrono::high_resolution_clock::now();
  // FAISS_THROW_IF_NOT(is_trained);
  assert(is_trained);
  if (n == 0) {
    return;
  }
  // exclusive lock
  {
    // std::unique_lock lock(this->codes_mutex);
    pthread_rwlock_wrlock(&this->codes_mutex);
    int max = xids[n - 1];
    codes.resize((max + 1) * code_size);
    // std::cout << "resize from " << codes.size() << " to "
    //           << (max + 1) * code_size << "\n";
    // std::chrono::high_resolution_clock::time_point resize_end =
    //     std::chrono::high_resolution_clock::now();
    sa_encode(n, x, codes.data() + xids[0] * code_size);
    if (xids[0] > ntotal) {
      std::memset(codes.data() + ntotal * code_size, 0,
                  (xids[0] - ntotal) * code_size);
    }
    if (max + 1 > ntotal) {
      ntotal = max + 1;
    }
    pthread_rwlock_unlock(&this->codes_mutex);
  }
}

void IndexFlatCodesL::reset() {
  // exclusive lock
  // std::unique_lock lock(this->codes_mutex);
  pthread_rwlock_wrlock(&this->codes_mutex);
  codes.clear();
  ntotal = 0;
  pthread_rwlock_unlock(&this->codes_mutex);
}

void IndexFlatCodesL::reserve(idx_t n) {
  // exclusive lock
  // std::unique_lock lock(this->codes_mutex);
  pthread_rwlock_wrlock(&this->codes_mutex);
  codes.reserve(n * code_size);
  pthread_rwlock_unlock(&this->codes_mutex);
}

size_t IndexFlatCodesL::sa_code_size() const { return code_size; }

size_t IndexFlatCodesL::remove_ids(const IDSelector& sel) {
  // FAISS_THROW_MSG("remove ids not implemented for IndexFlatCodesL");
  assert(!"remove ids not implemented for IndexFlatCodesL");
  return -1;
}

void IndexFlatCodesL::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
  // FAISS_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));
  assert(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));
  // sa_decode is not implemented at IndexFlatCodesL. subclasses shall be aware
  // that we are taking the read access
  // std::shared_lock lock(this->codes_mutex);
  pthread_rwlock_rdlock(&this->codes_mutex);
  sa_decode(ni, codes.data() + i0 * code_size, recons);
  pthread_rwlock_unlock(&this->codes_mutex);
}

void IndexFlatCodesL::reconstruct(idx_t key, float* recons) const {
  reconstruct_n(key, 1, recons);
}

FlatCodesDistanceComputer* IndexFlatCodesL::get_FlatCodesDistanceComputer()
    const {
  // FAISS_THROW_MSG("not implemented");
  assert(!"not implemented");
}

void IndexFlatCodesL::check_compatible_for_merge(
    const Index& otherIndex) const {
  // minimal sanity checks
  const IndexFlatCodesL* other =
      dynamic_cast<const IndexFlatCodesL*>(&otherIndex);
  // FAISS_THROW_IF_NOT(other);
  assert(other);
  // FAISS_THROW_IF_NOT(other->d == d);
  assert(other->d == d);
  // FAISS_THROW_IF_NOT(other->code_size == code_size);
  assert(other->code_size == code_size);
  // FAISS_THROW_IF_NOT_MSG(typeid(*this) == typeid(*other),
  //                        "can only merge indexes of the same type");
  assert(typeid(*this) == typeid(*other));
}

void IndexFlatCodesL::merge_from(Index& otherIndex, idx_t add_id) {
  // FAISS_THROW_IF_NOT_MSG(add_id == 0, "cannot set ids in FlatCodes index");
  assert(add_id == 0);
  check_compatible_for_merge(otherIndex);
  IndexFlatCodesL* other = static_cast<IndexFlatCodesL*>(&otherIndex);

  // exclusive lock
  // std::unique_lock lock(this->codes_mutex);
  pthread_rwlock_wrlock(&this->codes_mutex);

  codes.resize((ntotal + other->ntotal) * code_size);
  memcpy(codes.data() + (ntotal * code_size), other->codes.data(),
         other->ntotal * code_size);
  ntotal += other->ntotal;
  other->reset();
  pthread_rwlock_unlock(&this->codes_mutex);
}

CodePacker* IndexFlatCodesL::get_CodePacker() const {
  return new CodePackerFlat(code_size);
}

// it will also affect the idx_t, likely not used
void IndexFlatCodesL::permute_entries(const idx_t* perm) {
  // exclusive lock
  // std::unique_lock lock(this->codes_mutex);
  pthread_rwlock_wrlock(&this->codes_mutex);

  std::vector<uint8_t> new_codes(codes.size());

  for (idx_t i = 0; i < ntotal; i++) {
    memcpy(new_codes.data() + i * code_size, codes.data() + perm[i] * code_size,
           code_size);
  }
  std::swap(codes, new_codes);
  pthread_rwlock_unlock(&this->codes_mutex);
}

}  // namespace faiss
