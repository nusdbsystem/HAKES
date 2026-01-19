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

#include "bert_input.h"

#include <cstring>
#include <string>

BertInput encode_bert_input(float* words, float* segments, float* valid_len,
                            int32_t max_seq_length) {
  BertInput bert_input;
  bert_input.words = words;
  bert_input.segments = segments;
  bert_input.valid_len = valid_len;
  bert_input.max_seq_length = max_seq_length;
  bert_input.data = std::unique_ptr<float[]>(new float[2 + 2 * max_seq_length]);
  auto start = bert_input.data.get();
  std::memcpy(start, &max_seq_length, sizeof(int32_t));
  start += 1;
  std::memcpy(start, valid_len, sizeof(float));
  start += 1;
  std::memcpy(start, words, max_seq_length * sizeof(float));
  start += max_seq_length;
  std::memcpy(start, segments, max_seq_length * sizeof(float));
  return bert_input;
}

void* get_encoded_bert_input(const BertInput& input, size_t* bytes) {
  if (bytes == nullptr) {
    return nullptr;
  }
  if (input.data == nullptr) {
    *bytes = 0;
    return nullptr;
  }

  *bytes = sizeof(int32_t) + sizeof(float) +
           2 * sizeof(float) * input.max_seq_length;
  return input.data.get();
}

void view_bert_input(BertInput* viewer, float* data, size_t size) {
  if ((data == nullptr) || (viewer == nullptr)) {
    return;
  }
  if ((size < 2) || (size % 2 != 0)) {
    return;
  }

  viewer->max_seq_length = *(reinterpret_cast<int32_t*>(data));
  if (viewer->max_seq_length != (size - 2) / 2) {
    return;
  }

  viewer->valid_len = data + 1;
  viewer->words = data + 2;
  viewer->segments = data + 2 + viewer->max_seq_length;
  return;
}

std::string bert_input_to_string(const BertInput& input) {
  std::string result =
      "max_seq_length: " + std::to_string(input.max_seq_length) + "\n";
  result += "valid_len: " + std::to_string(*input.valid_len) + "\n";
  result += "words: ";
  for (int i = 0; i < input.max_seq_length; i++) {
    result += std::to_string(input.words[i]) + " ";
  }
  result += "\nsegments: ";
  for (int i = 0; i < input.max_seq_length; i++) {
    result += std::to_string(input.segments[i]) + " ";
  }
  return result;
}
