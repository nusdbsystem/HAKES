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

#ifndef HAKES_SEARCHWORKER_INDEX_EXT_BLOCKINVERTEDLISTSL_H_
#define HAKES_SEARCHWORKER_INDEX_EXT_BLOCKINVERTEDLISTSL_H_

#include <pthread.h>

#include <atomic>

#include "utils/io.h"
#include "search-worker/index/invlists/InvertedLists.h"
#include "search-worker/index/utils/AlignedTable.h"

namespace faiss {

struct CodePacker;

struct InvListUnit {
  size_t n_per_block = 0;  // nb of vectors stored per block
  size_t block_size = 0;   // nb bytes per block

  bool active = false;
  std::atomic_int count_{0};
  AlignedTable<uint8_t> codes_;
  std::vector<idx_t> ids_;
  mutable pthread_rwlock_t mu_;
  InvListUnit() = default;

  // move constructor
  InvListUnit(InvListUnit&& other) noexcept {
    // lock the other
    pthread_rwlock_wrlock(&other.mu_);
    n_per_block = other.n_per_block;
    block_size = other.block_size;
    active = other.active;
    count_.store(other.count_.load(std::memory_order_relaxed),
                 std::memory_order_relaxed);
    codes_ = std::move(other.codes_);
    ids_ = std::move(other.ids_);
    pthread_rwlock_unlock(&other.mu_);
  }

  InvListUnit& operator=(InvListUnit&& other) noexcept {
    if (this == &other) {
      return *this;
    }

    // lock the other
    pthread_rwlock_wrlock(&other.mu_);

    n_per_block = other.n_per_block;
    block_size = other.block_size;
    active = other.active;
    count_.store(other.count_.load(std::memory_order_relaxed),
                 std::memory_order_relaxed);
    codes_ = std::move(other.codes_);
    ids_ = std::move(other.ids_);
    pthread_rwlock_unlock(&other.mu_);
    return *this;
  }

  size_t get_size() const { return count_.load(std::memory_order_relaxed); }
  size_t resize(size_t new_size);
  size_t resize_nolock(size_t new_size);
  size_t set_active(size_t n_per_block, size_t block_size,
                    size_t init_list_cap = 0);
  size_t add_entries(size_t n_entry, const idx_t* ids_in, const uint8_t* code,
                     const CodePacker* packer);

  const uint8_t* get_codes() const;
  const idx_t* get_ids() const;

  void release_codes(const uint8_t* codes) const;
  void release_ids(const idx_t* ids) const;

  inline void lock_exclusive() const { pthread_rwlock_wrlock(&mu_); }
  inline void unlock_exclusive() const { pthread_rwlock_unlock(&mu_); }

  void read(hakes::IOReader* f);
  void write(hakes::IOWriter* f) const;
};

/** Inverted Lists that are organized by blocks.
 *
 * Different from the regular inverted lists, the codes are organized by blocks
 * of size block_size bytes that reprsent a set of n_per_block. Therefore, code
 * allocations are always rounded up to block_size bytes. The codes are also
 * aligned on 32-byte boundaries for use with SIMD.
 *
 * To avoid misinterpretations, the code_size is set to (size_t)(-1), even if
 * arguably the amount of memory consumed by code is block_size / n_per_block.
 *
 * The writing functions add_entries and update_entries operate on block-aligned
 * data.
 */
struct BlockInvertedListsL : InvertedLists {
  size_t n_per_block_ = 0;  // nb of vectors stored per block
  size_t block_size_ = 0;   // nb bytes per block

  // required to interpret the content of the blocks (owned by this)
  const CodePacker* packer_ = nullptr;

  std::vector<InvListUnit> lists_;
  std::vector<int> load_list_;

  explicit BlockInvertedListsL(size_t nlist, size_t vec_per_block,
                               size_t block_size);

  void init(const CodePacker* packer, const std::vector<int>& load_list,
            size_t init_list_cap = 0);

  void set_code_packer(const CodePacker* packer);

  size_t list_size(size_t list_no) const override;
  const uint8_t* get_codes(size_t list_no) const override;
  const idx_t* get_ids(size_t list_no) const override;

  void release_codes(size_t list_no, const uint8_t* codes) const override;
  void release_ids(size_t list_no, const idx_t* ids) const override;

  inline void lock_exclusive(size_t list_no) const {
    lists_[list_no].lock_exclusive();
  }
  inline void unlock_exclusive(size_t list_no) const {
    lists_[list_no].unlock_exclusive();
  }

  const uint8_t* get_single_code(size_t list_no, size_t offset) const override;

  // works only on empty BlockInvertedListsL
  // the codes should be of size ceil(n_entry / n_per_block) * block_size
  // and padded with 0s
  size_t add_entries(size_t list_no, size_t n_entry, const idx_t* ids,
                     const uint8_t* code) override;

  /// not implemented
  void update_entries(size_t list_no, size_t offset, size_t n_entry,
                      const idx_t* ids, const uint8_t* code) override;

  // also pads new data with 0s
  void resize(size_t list_no, size_t new_size) override;
  void resize_nolock(size_t list_no, size_t new_size);

  // ~BlockInvertedListsL() override;
  virtual ~BlockInvertedListsL() override;
};

}  // namespace faiss

#endif  // HAKES_SEARCHWORKER_INDEX_EXT_BLOCKINVERTEDLISTSL_H_