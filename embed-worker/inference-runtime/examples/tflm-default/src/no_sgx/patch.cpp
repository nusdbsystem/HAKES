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
#include <cstring>
#include <memory>

#include "embed-worker/inference-runtime/examples/tflm-default/src/common/abstract.h"
#include "store-client/store.h"

namespace embed_worker {

std::unique_ptr<ModelContext> load_model(const char* model_id,
                                         size_t model_id_len,
                                         const char* /* dec_key*/, void* store,
                                         int* status) {
  assert(status);
  assert(model_id);
  assert(model_id_len > 0);

  hakes::Store* store_ptr = static_cast<hakes::Store*>(store);

  size_t load_len = 0;

  char fetch_id[model_id_len + sizeof(model_file_suffix)];
  memcpy(fetch_id, model_id, model_id_len);
  memcpy(fetch_id + model_id_len, model_file_suffix, sizeof(model_file_suffix));
  size_t fetch_id_len = sizeof(fetch_id);

  printf("fetch_id: %s\n", fetch_id);
  std::unique_ptr<char[]> loaded =
      store_ptr->Get(std::string(fetch_id, fetch_id_len), &load_len);
  if (!loaded) {
    printf("Failed to load model %s\n", fetch_id);
    *status = -1;
    return nullptr;
  }
  *status = 0;
  // prepare return
  return std::unique_ptr<ModelContext>(
      new ModelContextImpl(loaded.release(), load_len));
}

}  // namespace embed_worker
