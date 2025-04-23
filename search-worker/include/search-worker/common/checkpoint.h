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

#ifndef HAKES_SERVING_UTILS_CHECKPOINT_H_
#define HAKES_SERVING_UTILS_CHECKPOINT_H_

#include <stdexcept>
#include <string>

namespace hakes {

/**
 * @brief The format of checkpoint directory is in the form of
 * "checkpoint_%d"
 *
 */
const char CheckpointDirPrefix[] = "checkpoint_";

std::string get_latest_checkpoint_path(const std::string& path);

inline std::string format_checkpoint_path(int checkpoint_no) {
  // return checkpoint_%06d
  return CheckpointDirPrefix + std::to_string(checkpoint_no);
}

inline int get_checkpoint_no(const std::string& path) {
  if (path.find(CheckpointDirPrefix) != 0) {
    return -1;
  }
  std::string suffix = path.substr(std::string(CheckpointDirPrefix).size());
  try {
    auto checkpoint_no = std::stoi(suffix);
    return checkpoint_no;
  } catch (...) {
    return -1;
  }
}

}  // namespace hakes

#endif  // HAKES_SERVING_UTILS_CHECKPOINT_H_
