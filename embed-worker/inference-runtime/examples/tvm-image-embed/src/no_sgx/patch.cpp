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

#include <string.h>

#include <string>

#include "embed-worker/inference-runtime/examples/tvm-default/src/common/abstract.h"
#include "store-client/store.h"

namespace embed_worker {

size_t load_and_decode(const char* id, size_t len, const char* /*dec_key*/,
                       void* store, char** out, int* status) {
  hakes::Store* store_ptr = static_cast<hakes::Store*>(store);
  size_t loaded_len = 0;
  std::unique_ptr<char[]> loaded =
      store_ptr->Get(std::string(id, len), &loaded_len);
  *out = loaded.get();
  loaded.release();
  *status = 0;
  printf("loaded: %s\n", std::string(id, len).c_str());
  return loaded_len;
};

}  // namespace embed_worker
