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

#ifndef HAKES_SEARCHWORKER_COMMON_GEN_H_
#define HAKES_SEARCHWORKER_COMMON_GEN_H_

#include <memory>
#include <random>

inline std::unique_ptr<float[]> gen_vectors(int n, int d, bool normalize, int seed = 123) {
  std::unique_ptr<float[]> vecs{new float[n * d]};
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      vecs[i * d + j] = dis(gen);  // Generate a random float
    }
    if (normalize) {
      float sum = 0;
      for (int j = 0; j < d; j++) {
        sum += vecs[i * d + j] * vecs[i * d + j];
      }
      float norm = std::sqrt(sum);
      for (int j = 0; j < d; j++) {
        vecs[i * d + j] /= norm;
      }
    }
  }
  return vecs;
}

#endif  // HAKES_SEARCHWORKER_COMMON_GEN_H_
