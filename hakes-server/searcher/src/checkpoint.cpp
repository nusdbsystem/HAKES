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

#include "search-worker/checkpoint.h"

#include <filesystem>

namespace hakes {

std::string get_latest_checkpoint_path(const std::string& path) {
  std::string last_checkpoint;
  std::string prefix = CheckpointDirPrefix;
  int latest_checkpoint_no = -1;
  // get the directory with matching prefix and largest suffix number
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    auto fname = entry.path().filename().string();
    if (!entry.is_directory() || fname.find(prefix) != 0) {
      continue;
    }
    std::string suffix = fname.substr(prefix.size());
    try {
      int checkpoint_no = std::stoi(suffix);
      if (checkpoint_no > latest_checkpoint_no) {
        latest_checkpoint_no = checkpoint_no;
        last_checkpoint = fname;
      }
    } catch (...) {
      continue;
    }
  }
  return last_checkpoint;
}

}  // namespace hakes
