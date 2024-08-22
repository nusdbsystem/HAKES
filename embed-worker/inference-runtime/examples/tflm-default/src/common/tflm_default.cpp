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

/**
 * implemented with reference to tensorflow/tflite-micro person detection
 *  example and Jumpst3r/tensorflow-lite-sgx sample
 */

#include <cstdint>
#include <cstring>
#include <memory>

#include "embed-worker/common/config.h"
#include "embed-worker/inference-runtime/examples/tflm-default/src/common/abstract.h"
#include "embed-worker/inference-runtime/model/model.h"
#include "embed-worker/inference-runtime/tflm/src/common/abstract.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define TensorArenaSize MODELRT_BUF_SZ

namespace embed_worker {
namespace {

#ifdef USE_SGX

int printf(const char* fmt, ...) {
  char buf[8192] = {'\0'};
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, 8192, fmt, ap);
  va_end(ap);
  tflm_default_printf(buf);
  return (int)strnlen(buf, 8192 - 1) + 1;
}
#endif  // USE_SGX

class ModelRTImpl : public ModelRT {
 public:
  ModelRTImpl()
      : model_(nullptr),
        micro_error_reporter_(),
        interpreter_(nullptr),
        tensor_arena_(nullptr) {}
  ~ModelRTImpl() {
    if (tensor_arena_) free(tensor_arena_);
    tensor_arena_ = nullptr;
  }

  int maybe_init(const char* model_buf) {
    if (model_ != model_buf) {
      tflite::InitializeTarget();
      model_ = model_buf;
      const tflite::Model* model =
          tflite::GetModel(reinterpret_cast<const int8_t*>(model_buf));
      if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf(
            "Model provided is schema version %d not equal to \
          supported version %d\n",
            model->version(), TFLITE_SCHEMA_VERSION);
        return 1;  // SGX_ERROR_UNEXPECTED;
      }
      int kTensorArenaSize = TensorArenaSize;
      if (!tensor_arena_) {
        tensor_arena_ = (uint8_t*)malloc(kTensorArenaSize);
        if (!tensor_arena_) return 3;  // SGX_ERROR_OUT_OF_MEMORY;
      }
      memset(tensor_arena_, 0, kTensorArenaSize);
      interpreter_.reset(new tflite::MicroInterpreter(
          model, resolver_, tensor_arena_, kTensorArenaSize,
          &micro_error_reporter_));
      auto status = interpreter_->AllocateTensors();
      if (status != kTfLiteOk) {
        printf("AllocateTensors() failed\n");
        return 1;  // SGX_ERROR_UNEXPECTED;
      }
    }
    return 0;  // SGX_SUCCESS;
  }

  tflite::MicroInterpreter* interpreter() { return interpreter_.get(); }

 private:
  const char* model_;  // does not own.
  tflite::MicroErrorReporter micro_error_reporter_;
  tflite::AllOpsResolver
      resolver_;  // resolver lifetime needs to be longer than interpreter.
  std::unique_ptr<tflite::MicroInterpreter> interpreter_;  // ownership
  uint8_t* tensor_arena_;                                  // ownership
};

constexpr char model_file_suffix[] = ".model";
}  // anonymous namespace

ModelRT* model_rt_init(const ModelContext& ctx, int* status) {
  ModelRTImpl* ret = new ModelRTImpl();
  *status =
      ret->maybe_init(static_cast<const ModelContextImpl&>(ctx).content());
  if (*status != 0 /*SGX_SUCCESS*/) {
    delete ret;
    return nullptr;
  }
  tflm_default_print_time();
  return ret;
}

void free_model_rt(ModelRT* rt) {
  delete (rt);
  rt = nullptr;
}

int execute_inference(const char* input_src, size_t input_src_size,
                      const ModelContext& model_context, ModelRT* model_rt,
                      char* output, size_t output_buf_capacity,
                      size_t* output_size) {
  ModelRTImpl* rt = reinterpret_cast<ModelRTImpl*>(model_rt);
  auto ret = rt->maybe_init(
      static_cast<const ModelContextImpl&>(model_context).content());
  assert(ret == 0);

  const int8_t* tfinput = reinterpret_cast<const int8_t*>(input_src);
  auto input = rt->interpreter()->input(0);
  memcpy(input->data.raw, tfinput, input->bytes);

  auto status = rt->interpreter()->Invoke();
  if (status != kTfLiteOk) {
#ifndef NDEBUG
    printf("Invoke() failed\n");
#endif         // NDEBUG
    return 1;  // SGX_ERROR_UNEXPECTED;
  }

  // copy result into output

#ifndef NDEBUG
  // auto output_dim = interpreter.output(0)->dims[1].data[0];
  // printf("output: \n");
  // // for (int i = 0; i < output_dim; i++) {
  // for (int i = 0; i < 200; i++) {
  // // printf("byte index %d : %d\n", i, interpreter.output(0)->data.int8[i]);
  //   printf("byte index %d : %f\n", i, interpreter.output(0)->data.f[i]);
  // }
#endif  // NDEBUG

  auto output_sz = rt->interpreter()->output(0)->bytes;
  if (output_sz > output_buf_capacity) {
#ifndef NDEBUG
    printf("Not enough output buffer\n");
#endif         // NDEBUG
    return 1;  // SGX_ERROR_UNEXPECTED;
  }

  memcpy(output, rt->interpreter()->output(0)->data.raw, output_sz);
  *output_size = output_sz;

  return 0;  // SGX_SUCCESS;
}

}  // namespace embed_worker
