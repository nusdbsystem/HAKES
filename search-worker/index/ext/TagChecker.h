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

#ifndef HAKES_SEARCHWORKER_INDEX_EXT_TAGCHECKER_H_
#define HAKES_SEARCHWORKER_INDEX_EXT_TAGCHECKER_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_set>

namespace faiss {

template <typename T>
class TagChecker {
 public:
  TagChecker() { pthread_rwlock_init(&mu_, nullptr); };
  ~TagChecker() { pthread_rwlock_destroy(&mu_); };

  inline void add_reader() const { pthread_rwlock_rdlock(&mu_); }
  inline void release_reader() const { pthread_rwlock_unlock(&mu_); }

  inline uint8_t check(const T& id) const {
    bool ret = tags_.find(id) != tags_.end();
    return (ret) ? 1 : 0;
  };

  inline void set(const T& id) {
    pthread_rwlock_wrlock(&mu_);
    tags_.insert(id);
    pthread_rwlock_unlock(&mu_);
  }
  inline void set(uint32_t n, const T* ids) {
    pthread_rwlock_wrlock(&mu_);
    for (size_t i = 0; i < n; i++) {
      tags_.insert(ids[i]);
    }
    pthread_rwlock_unlock(&mu_);
  }

 private:
  mutable pthread_rwlock_t mu_;
  std::unordered_set<T> tags_;
};

}  // namespace faiss

#endif  // HAKES_SEARCHWORKER_INDEX_EXT_TAGCHECKER_H_
