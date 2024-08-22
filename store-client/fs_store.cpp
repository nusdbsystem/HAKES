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

#include "fs_store.h"

#include <cassert>
#include <fstream>
#include <thread>

#include "utils/fileutil.h"

namespace hakes {
namespace {
inline std::string CreatePath(const std::string& key,
  const std::string& path) {
  return path + '/' + key;
}
}  // anonymous namespace

int FsStore::Get(const std::string& key, std::string* value) {
  assert(value);
  value->clear();
  printf("remote_store latency %dms\n", sim_remote_lat_);
  printf("cache path %s\n", cache_path_.c_str());
  printf("remote store path %s\n", path_.c_str());

  // no cache, load directly from path 
  if (cache_path_.empty()) {
    if(sim_remote_lat_) {
      printf("Get from remote store\n");
      std::this_thread::sleep_for(std::chrono::milliseconds(sim_remote_lat_));
    }
    ReadFileToString(CreatePath(key, path_), value);
    return value->empty() ? -1 : 0; // success indicated by content existence
  }

  // check cache first
  ReadFileToString(CreatePath(key, cache_path_), value);
  if (!value->empty()) return 0;
  // load from remote
  if (sim_remote_lat_) {
    printf("Get remote store\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(sim_remote_lat_));
  }
  ReadFileToString(CreatePath(key, path_), value);
  if (value->empty()) return -1;
  if (cache_path_.empty()) return 0;
  // store in cache
  return (WriteStringToFile(CreatePath(key, cache_path_), *value) == 0)
    ? 0 : -1;
}

std::unique_ptr<char[]> FsStore::Get(const std::string& key, size_t* len) {
  printf("remote_store latency %dms\n", sim_remote_lat_);
  printf("cache path %s\n", cache_path_.c_str());
  printf("remote store path %s\n", path_.c_str());

  // no cache, load directly from path 
  if (cache_path_.empty()) {
    if(sim_remote_lat_) {
      printf("Get from remote store\n");
      std::this_thread::sleep_for(std::chrono::milliseconds(sim_remote_lat_));
    }
    return ReadFileToCharArray(CreatePath(key, path_).c_str(), len);
  }

  // check cache first
  auto ret = ReadFileToCharArray(CreatePath(key, path_).c_str(), len);
  if (ret) return ret;
  // load from remote
  if (sim_remote_lat_) {
    printf("Get remote store\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(sim_remote_lat_));
  }
  ret = ReadFileToCharArray(CreatePath(key, path_).c_str(), len);
  if (!ret) return nullptr;
  if (cache_path_.empty()) return 0;
  // store in cache
  if (WriteCharArrayToFile(CreatePath(key, cache_path_).c_str(), ret.get(), *len) != 0) {
    *len = 0;
    return nullptr;
  }
  return ret;
}

int FsStore::Put(const std::string& key, const std::string& value) {
  if ((!cache_path_.empty())
    && (WriteStringToFile(CreatePath(key, cache_path_), value) != 0)) {
    return -1;
  }
  return (WriteStringToFile(CreatePath(std::move(key), path_),
    std::move(value)) == 0) ? 0 : -1;
}

Store* OpenFsStore(const std::string& path) {
  return new FsStore(std::move(path)); 
}

Store* OpenFsStore(const std::string& path, const std::string& cache_path,
  int sim_remote_lat) {
  return new FsStore(std::move(path), std::move(cache_path), sim_remote_lat); 
}

}  // namespace hakes
