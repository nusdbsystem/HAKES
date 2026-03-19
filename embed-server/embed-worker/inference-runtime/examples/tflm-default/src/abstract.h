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

#ifndef HAKES_EMBEDWORKER_INFERENCERUNTIME_TFLMDEFAULT_ABSTRACT_H_
#define HAKES_EMBEDWORKER_INFERENCERUNTIME_TFLMDEFAULT_ABSTRACT_H_

#include <stdlib.h>

#include <memory>

#include "embed-worker/inference-runtime/model.h"

namespace embed_worker {

class ModelContextImpl : public ModelContext {
 public:
  ModelContextImpl(char* content, size_t len)
      : content_(content), content_len_(len) {}
  ~ModelContextImpl() {
    if (content_) free(content_);
  }

  // delete copy and move
  ModelContextImpl(const ModelContextImpl&) = delete;
  ModelContextImpl(ModelContextImpl&&) = delete;
  ModelContextImpl& operator=(const ModelContextImpl&) = delete;
  ModelContextImpl& operator=(ModelContextImpl&&) = delete;

  const char* content() const { return content_; }
  size_t len() const { return content_len_; }

 private:
  char* content_;
  size_t content_len_;
};

constexpr char model_file_suffix[] = ".model";

std::unique_ptr<ModelContext> load_model(const char* model_id,
                                         size_t model_id_len,
                                         const char* dec_key, void* store,
                                         int* status);

} // namespace embed_worker

#endif  // HAKES_EMBEDWORKER_INFERENCERUNTIME_TFLMDEFAULT_ABSTRACT_H_
