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

#ifndef HAKES_UTILS_DATALOADER_H_
#define HAKES_UTILS_DATALOADER_H_

#include <cstdio>
#include <cstdlib>

inline float* load_data(const char* filename, size_t d, size_t n) {
  FILE* f = fopen(filename, "rb");
  if (!f) {
    fprintf(stderr, "could not open %s\n", filename);
    exit(-1);
  }
  float* data = new float[d * n];
  size_t ret = fread(data, sizeof(float), n * d, f);
  if (ret != (size_t)(n * d)) {
    fprintf(stderr, "could not read %s\n", filename);
    exit(-1);
  }
  fclose(f);
  return data;
}

inline int* load_groundtruth(const char* filename, size_t gt_len, size_t nq) {
  FILE* f = fopen(filename, "rb");
  if (!f) {
    fprintf(stderr, "could not open %s\n", filename);
    exit(-1);
  }
  int* gt = new int[gt_len * nq];
  size_t ret = fread(gt, sizeof(int), nq * gt_len, f);
  if (ret != (size_t)(nq * gt_len)) {
    fprintf(stderr, "could not read %s\n", filename);
    exit(-1);
  }
  fclose(f);
  return gt;
}

#endif  // HAKES_UTILS_DATALOADER_H_