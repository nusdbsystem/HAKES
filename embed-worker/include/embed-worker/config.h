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

#ifndef HAKES_EMBEDWORKER_CONFIG_H_
#define HAKES_EMBEDWORKER_CONFIG_H_

// configs to enforce the memory buffers sizes and times for processing inside enclave.

/**
 * @brief fixed buffer size for storing plaintext 
 *  image_224_f32: 602112
 *  image_299_f32: 1072812
 *  bert hello world! 1032
 */
#define INPUT_BUF_SZ 1032
/**
 * @brief fixed buffer size for storing the output and its encryption 
 * tflm sample models: 4004, 1000 imagenet + 1 background classes.
 * tvm sample models: 4000, it dictate the output size of tvm runtime output
 *  so incorrect value will cause tvm runtime to crash.
 */
// #define OUTPUT_BUF_SZ 4000
// #define OUTPUT_BUF_SZ 4004 // tflm-mb
// #define OUTPUT_BUF_SZ 1024*4 // tvm-mbnetembed
// #define OUTPUT_BUF_SZ 2048*4 // tvm-resnetembed
#define OUTPUT_BUF_SZ 768*4 // tvm-bert

/**
 * @brief config the model runtime memory consumption for
 *  storing models and intermediates
 * 
 * tflm:
 *  mobilenet_v1_1.0_224 5000000
 *  resnet_v2_101_299 23600000
 *  densenet 12000000
 * tvm crt:
 *  mobilenet1.0: 30 MB
 *  resnet101_v2: 205 MB
 *  densenet121: 55 MB
 *  bert: 460 MB
 */
// #define MODELRT_BUF_SZ (30<<20)
// #define MODELRT_BUF_SZ (205<<20) // tvm-resnetembed
#define MODELRT_BUF_SZ (460<<20) // tvm-bert
// #define MODELRT_BUF_SZ (5000000) // tflm-mb

/**
 * @brief equalize a sandbox processing time
 *  (init one fixed model and fetch key everytime) 
 */
#define EQUAL_EXEC_TIME false

/**
 * @brief force the use of model inside enclave,
 *  when EQUAL_EXEC_TIME is true
 */
#define PERMITTED_MODEL "mobilenet1.0"

#endif // HAKES_EMBEDWORKER_CONFIG_H_
