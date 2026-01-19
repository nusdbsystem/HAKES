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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include "embed-worker/config.h"
#include "embed-worker/inference-runtime/examples/tvm-default/src/abstract.h"
#include "embed-worker/inference-runtime/model.h"
#include "embed-worker/inference-runtime/tvm_crt/src/abstract.h"
#include "embed-worker/inference-runtime/tvm_crt/src/bundle.h"

#define OUTPUT_LEN (OUTPUT_BUF_SZ / sizeof(float))
#define TVM_CRT_MEMORY_BUFFER_SZ MODELRT_BUF_SZ

namespace embed_worker {
namespace {
class ModelContextImpl : public ModelContext {
 public:
  ModelContextImpl(char* graph, size_t graph_len, char* params_data,
                   size_t params_len)
      : graph_(graph),
        graph_len_(graph_len),
        params_data_(params_data),
        params_len_(params_len) {}
  ~ModelContextImpl() {
    // ocall_tvm_print_string("model context deleted");
    tvm_default_print_string("model context deleted");
    if (graph_) free(graph_);
    if (params_data_) free(params_data_);
  }

  // delete copy and move
  ModelContextImpl(const ModelContextImpl&) = delete;
  ModelContextImpl(ModelContextImpl&&) = delete;
  ModelContextImpl& operator=(const ModelContextImpl&) = delete;
  ModelContextImpl& operator=(ModelContextImpl&&) = delete;

  const char* graph() const { return graph_; }
  size_t graph_len() const { return graph_len_; }
  const char* params_data() const { return params_data_; }
  size_t params_len() const { return params_len_; }

 private:
  char* graph_;
  size_t graph_len_;
  char* params_data_;
  size_t params_len_;
};

class ModelRTImpl : public ModelRT {
 public:
  ModelRTImpl(uint8_t* buf, void* handle)
      : tvm_memory_buffer_(buf), handle_(handle) {}

  ~ModelRTImpl() {
    // ocall_tvm_print_string("model rt deleted");
    tvm_default_print_string("model rt deleted");
    if (handle_) tvm_runtime_destroy(handle_);
    handle_ = nullptr;
    if (tvm_memory_buffer_) free(tvm_memory_buffer_);
    tvm_memory_buffer_ = nullptr;
  }

  void* handle() { return handle_; }

 private:
  uint8_t* tvm_memory_buffer_;
  void* handle_;
};
}  // anonymous namespace

ModelRT* model_rt_init(const ModelContext& ctx, int* status) {
  uint8_t* buf = (uint8_t*)malloc(TVM_CRT_MEMORY_BUFFER_SZ);
  if (!buf) {
    *status = 3;
    return nullptr;
  }
  const ModelContextImpl& mc = static_cast<const ModelContextImpl&>(ctx);
  void* handle =
      tvm_runtime_create(mc.graph(), mc.params_data(), mc.params_len(), buf,
                         TVM_CRT_MEMORY_BUFFER_SZ);
  // ocall_print_time();
  tvm_default_print_time();
  return new ModelRTImpl{buf, handle};
}

void free_model_rt(ModelRT* rt) {
  delete (rt);
  rt = nullptr;
}

namespace {
const char graph_file_suffix[] = ".graph_json";
const char params_file_suffix[] = ".params_data";
constexpr int suffix_max_len =
    (sizeof(graph_file_suffix) > sizeof(params_file_suffix))
        ? sizeof(graph_file_suffix)
        : sizeof(params_file_suffix);

}  // anonymous namespace

std::unique_ptr<ModelContext> load_model(const char* model_id,
                                         size_t model_id_len,
                                         const char* dec_key, void* store,
                                         int* status) {
  assert(status);
  assert(model_id);
  assert(model_id_len > 0);

  char fetch_id[model_id_len + suffix_max_len];
  memcpy(fetch_id, model_id, model_id_len);
  memcpy(fetch_id + model_id_len, graph_file_suffix, sizeof(graph_file_suffix));
  char* graph = nullptr;
  size_t graph_len =
      load_and_decode(fetch_id, model_id_len + sizeof(graph_file_suffix),
                      dec_key, store, &graph, status);
  if (!graph) return nullptr;

  memcpy(fetch_id + model_id_len, params_file_suffix,
         sizeof(params_file_suffix));
  char* params = nullptr;
  size_t params_len =
      load_and_decode(fetch_id, model_id_len + sizeof(params_file_suffix),
                      dec_key, store, &params, status);
  if (!params) {
    free(graph);
    return nullptr;
  }

  return std::unique_ptr<ModelContext>(
      new ModelContextImpl(graph, graph_len, params, params_len));
}

int execute_inference(const char* input_src, size_t input_src_size,
                      const ModelContext& /*ctx*/, ModelRT* model_rt,
                      char* output_buf, size_t output_buf_capacity,
                      size_t* output_size) {
  void* handle = static_cast<ModelRTImpl*>(model_rt)->handle();

  DLTensor input;
  input.data = (char*)input_src;
  DLDevice dev = {kDLCPU, 0};
  input.device = dev;
  input.ndim = 4;
  DLDataType dtype = {kDLFloat, 32, 1};
  input.dtype = dtype;
  int64_t shape[4] = {1, 3, 224, 224};
  input.shape = shape;
  input.strides = NULL;
  input.byte_offset = 0;

  tvm_runtime_set_input(handle, "data", &input);

  // ocall_print_time();
  tvm_default_print_time();

  tvm_runtime_run(handle);

  // ocall_print_time();
  tvm_default_print_time();

  float output_storage[OUTPUT_LEN];
  DLTensor output;
  output.data = output_storage;
  DLDevice out_dev = {kDLCPU, 0};
  output.device = out_dev;
  output.ndim = 2;
  DLDataType out_dtype = {kDLFloat, 32, 1};
  output.dtype = out_dtype;
  int64_t out_shape[2] = {1, OUTPUT_LEN};
  output.shape = out_shape;
  output.strides = NULL;
  output.byte_offset = 0;

  tvm_runtime_get_output(handle, 0, &output);

  // ocall_print_time();
  tvm_default_print_time();

  int output_sz = OUTPUT_LEN * sizeof(float);
  if (output_sz > output_buf_capacity) {
    return 1;
  }

  memcpy(output_buf, output_storage, output_sz);
  *output_size = output_sz;

  // ocall_print_time();
  tvm_default_print_time();

  return 0;
}

}  // namespace embed_worker
