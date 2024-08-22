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

#ifndef HAKES_EMBEDWORKER_INFERENCERUNTIME_TVMBERT_COMMON_ABSTRACT_H_
#define HAKES_EMBEDWORKER_INFERENCERUNTIME_TVMBERT_COMMON_ABSTRACT_H_

#include <stdlib.h>

namespace embed_worker {

size_t load_and_decode(const char* id, size_t len, const char* dec_key,
                       void* store, char** out, int* status);

}  // namespace embed_worker

#endif  // HAKES_EMBEDWORKER_INFERENCERUNTIME_TVMBERT_COMMON_ABSTRACT_H_
