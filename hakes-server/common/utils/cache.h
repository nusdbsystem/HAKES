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

#ifndef HAKES_UTILS_CACHE_H_
#define HAKES_UTILS_CACHE_H_

// A simple cache implementation for storing keys and the serving model.
// Note that the concurrency of a serving serverless sandbox will be limited.
//  set to average or low load of the function. Scale out during burst load.
// Only one model will be served in a sandbox at a time.
// so this cache is quite primitive (maybe get better impl in the future)

#include <cassert>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace hakes {

template <typename T>
class SimpleCache {
 private:
  struct CacheEntry {
    std::unique_ptr<T> entry;
    int ref_count;
  };

 public:
  SimpleCache() : capacity_(1){};
  SimpleCache(uint64_t capacity) : capacity_(capacity) {
    cache_.reserve(capacity_);
  };
  ~SimpleCache() = default;

  SimpleCache(const SimpleCache&) = delete;
  SimpleCache(SimpleCache&&) = delete;
  SimpleCache& operator=(const SimpleCache&) = delete;
  SimpleCache& operator=(SimpleCache&&) = delete;

  /**
   * @brief check the existence of an entry.
   *  if not existent, the entry will loaded by the checker and added later
   *
   * @param id
   * @return int
   *  0: non-existent, checker will load and add it.
   *  1: exist (the entry is ready or being loaded by another checker)
   * -1: cache is full and all entries are being used. check again later.
   */
  inline int CheckAndTakeRef(const std::string& id) {
    std::lock_guard<std::mutex> lg(mu_);
    auto it = cache_.find(id);
    if (it != cache_.end()) {
      // found in cache (ready or another thread is fetching it)
      it->second.ref_count++;
      return 1;
    }
    if (cache_.size() >= capacity_) {
      auto c = cache_.begin();
      while (c != cache_.end()) {
        auto tmp = c++;
        if (tmp->second.ref_count == 0) cache_.erase(tmp);
      }
      if (cache_.size() >= capacity_) return -1;
    }
    cache_.emplace(std::move(id), SimpleCache<T>::CacheEntry{nullptr, 1});
    return 0;
  }

  // run check and TakeRef first
  inline T* AddToCache(const std::string& id, std::unique_ptr<T> entry) {
    std::lock_guard<std::mutex> lg(mu_);
    assert(cache_.find(id) != cache_.end());
    auto& holder = cache_.at(id).entry;
    holder.reset(entry.release());
    return holder.get();
  }

  // run check and TakeRef first
  inline T* RetrieveFromCache(const std::string& id) const {
    std::lock_guard<std::mutex> lg(mu_);
    assert(cache_.find(id) != cache_.end());
    return cache_.at(id).entry.get();
  }

  inline void Release(const std::string& id) {
    std::lock_guard<std::mutex> lg(mu_);
    auto it = cache_.find(id);
    if (it == cache_.end()) return;
    it->second.ref_count--;
    assert(it->second.ref_count >= 0);
  }

  inline bool Delete(const std::string& id) {
    std::lock_guard<std::mutex> lg(mu_);
    auto it = cache_.find(id);
    // already deleted
    if (it == cache_.end()) return true;
    // still hold by other threads
    if (it->second.ref_count > 0) return false;
    // safe to delete
    cache_.erase(std::move(id));
    return true;
  }

 private:
  uint64_t capacity_;
  mutable std::mutex mu_;
  std::unordered_map<std::string, SimpleCache<T>::CacheEntry> cache_;
};

}  // namespace hakes

#endif  // HAKES_UTILS_CACHE_H_
