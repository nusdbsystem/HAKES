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

#ifndef HAKES_EMBEDWORKER_INFERENCERUNTIME_TVMBERT_BERTINPUT_H_
#define HAKES_EMBEDWORKER_INFERENCERUNTIME_TVMBERT_BERTINPUT_H_

#include <memory>

/**
 * @brief Prepare bert input
 * |-max-seq-length (4bytes, size_t)-|-valid-len (4bytes, float)-|-words
 * (max-seq-length*4bytes, float[])-|-segments (max-seq-length*4bytes,
 * float[])-|
 *
 */

struct BertInput {
  float* words;
  float* segments;
  float* valid_len;
  int32_t max_seq_length;
  std::unique_ptr<float[]> data;
};

BertInput encode_bert_input(float* words, float* segments, float* valid_len,
                            int32_t max_seq_length);

void* get_encoded_bert_input(const BertInput& input, size_t* bytes);

void view_bert_input(BertInput* viewer, float* data, size_t size);

std::string bert_input_to_string(const BertInput& input);

#endif  // HAKES_EMBEDWORKER_INFERENCERUNTIME_TVMBERT_BERTINPUT_H_
