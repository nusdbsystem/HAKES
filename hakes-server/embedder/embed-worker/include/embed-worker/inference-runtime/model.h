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

#ifndef HAKES_EMBEDWORKER_INFERENCERT_MODEL_H_
#define HAKES_EMBEDWORKER_INFERENCERT_MODEL_H_

#include <cstddef>
#include <memory>

namespace embed_worker {

class ModelContext {
 public:
  virtual ~ModelContext() {}
};

std::unique_ptr<ModelContext> load_model(const char* model_id,
                                         size_t model_id_len,
                                         const char* dec_key, void* store,
                                         int* status);

/**
 * @brief abstraction of the runtime used by a TCS for inferencing
 *
 */
class ModelRT {
 public:
  virtual ~ModelRT() {}
};

/**
 * @brief initialization of model runtime from the context.
 *  caller owns the runtime resource.
 *
 * @param ctx model context to initialize a model runtime
 * @return ModelRT* initialized model runtime. return nullptr upon failure
 */
ModelRT* model_rt_init(const ModelContext& ctx, int* status);

/**
 * @brief free runtime resource
 *
 * @param rt model runtime
 */
void free_model_rt(ModelRT* rt);

/**
 * @brief execute inference with decrypted input and model
 *
 * @param input_src : decrypted input binary
 * @param input_src_size
 * @param model_context: model context
 * @param model_rt: model runtime initialized by model_rt_init
 * @param output : buffer caller allocated to store output
 * @param output_buf_capacity : size of the output buffer
 * @param output_size : actual output size set by the function on return
 * @return int : SGX_SUCCESS for success
 */
int execute_inference(const char* input_src, size_t input_src_size,
                      const ModelContext& model_context, ModelRT* model_rt,
                      char* output, size_t output_buf_capacity,
                      size_t* output_size);

}  // namespace embed_worker

#endif  // HAKES_EMBEDWORKER_INFERENCERT_MODEL_H_
