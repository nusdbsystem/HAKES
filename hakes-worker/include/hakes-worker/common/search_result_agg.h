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

#ifndef HAKES_HAKESWORKER_COMMON_SEARCHRESULTAGG_H_
#define HAKES_HAKESWORKER_COMMON_SEARCHRESULTAGG_H_

#include <cstdint>
#include <vector>

namespace hakes_worker {

struct result_set {
  std::vector<int64_t> ids;
  std::vector<float> scores;
};

result_set search_result_agg(const std::vector<result_set> partial_results, int k);

}  // namespace hakes_worker

#endif  // HAKES_HAKESWORKER_COMMON_SEARCHRESULTAGG_H_
