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

#include "search-worker/common/worker.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <filesystem>
#include <iostream>
#include <memory>

#include "message/searchservice.h"
#include "search-worker/common/workerImpl.h"
#include "utils/data_loader.h"
#include "utils/fileutil.h"

#define TEST_COLLECTION "test"


int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " INDEX_PATH" << std::endl;
    exit(1);
  }

  search_worker::WorkerImpl worker{};

  std::string index_path = argv[1];
  // check if the index path is a existing directory
  if (!std::filesystem::exists(index_path)) {
    std::cout << "INDEX_PATH does not exist" << std::endl;
    exit(1);
  }
  std::cout << "INDEX_PATH: " << index_path << std::endl;

  std::unique_ptr<char[]> fcontent;
  std::unique_ptr<char[]> rcontent;
  std::unique_ptr<char[]> ucontent;

  bool status = worker.Initialize(false, 1, 0, std::filesystem::absolute(index_path).string());
  if (!status) {
    printf("Failed to initialize\n");
    exit(1);
  }

  // generate vectors
  int n = 30;
  int d = 768;
  int nq = 10;
  int search_k = 10;
  int nprobe = 100;
  int k_factor = 2;
  float* xb = new float[d * n];
  float* xq = new float[d * nq];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) xb[d * i + j] = drand48();
    xb[d * i] += i / 1000.;
  }
  for (int i = 0; i < nq; i++) {
    for (int j = 0; j < d; j++) xq[d * i + j] = drand48();
    xq[d * i] += i / 1000.;
  }
  std::cout << "\nData generated ..." << std::endl;

  hakes::SearchWorkerLoadRequest load_req;
  auto resp_load = std::unique_ptr<char[]>(new char[4096 * 4096]);
  load_req.collection_name = TEST_COLLECTION;
  std::string encoded_load_req =
      hakes::encode_search_worker_load_request(load_req);
  assert(worker.LoadCollection(encoded_load_req.c_str(), encoded_load_req.size(),
                               resp_load.get(), 4096 * 4096));
  std::cout << "Index loaded" << std::endl;

  // add vectors.
  for (int i = 0; i < n; i++) {
    hakes::SearchWorkerAddRequest add_req;
    add_req.d = d;
    add_req.vecs =
        hakes::encode_hex_floats(xb + i * d, d);
    add_req.collection_name = TEST_COLLECTION;
    int64_t ids[1] = {i};
    add_req.ids = hakes::encode_hex_int64s(ids, 1);

    std::string encoded_req = hakes::encode_search_worker_add_request(add_req);

    auto resp = std::unique_ptr<char[]>(new char[4096]);
    auto success = worker.AddWithIds(encoded_req.c_str(), encoded_req.size(),
                                     resp.get(), 4096);
    if (success) {
      printf("Output: %s\n", resp.get());
    } else {
      printf("Failed to add\n");
      exit(1);
    }
  }

  // search the first vector
  hakes::SearchWorkerSearchRequest search_req;
  search_req.d = d;
  search_req.vecs = hakes::encode_hex_floats(xb, d);
  search_req.k = search_k;
  search_req.nprobe = nprobe;
  search_req.k_factor = k_factor;
  search_req.metric_type = 1;
  search_req.collection_name = TEST_COLLECTION;

  std::string encoded_search_req =
      hakes::encode_search_worker_search_request(search_req);

  auto resp = std::unique_ptr<char[]>(new char[4096 * 4096]);
  status = worker.Search(encoded_search_req.c_str(), encoded_search_req.size(),
                         resp.get(), 4096 * 4096);

  if (status) {
    printf("Output: %s\n", resp.get());
  } else {
    printf("Failed to search\n");
    exit(1);
  }

  // decode the search result
  hakes::SearchWorkerSearchResponse search_resp;
  status =
      hakes::decode_search_worker_search_response(resp.get(), &search_resp);
  if (!status) {
    printf("Failed to decode search result\n");
    exit(1);
  } else {
    size_t content_len = 0;
    auto ids = hakes::decode_hex_int64s(search_resp.ids, &content_len);
    printf("ids size: %ld\n", content_len);
    for (int i = 0; i < content_len; i++) {
      printf("id: %ld\n", ids[i]);
    }
  }
  // rerank
  hakes::SearchWorkerRerankRequest rerank_req;
  rerank_req.d = d;
  rerank_req.k = search_k;
  rerank_req.metric_type = 1;
  rerank_req.vecs = search_req.vecs;
  rerank_req.input_ids = search_resp.ids;
  rerank_req.collection_name = TEST_COLLECTION;

  std::string encoded_rerank_req =
      hakes::encode_search_worker_rerank_request(rerank_req);

  auto resp_rerank = std::unique_ptr<char[]>(new char[4096 * 4096]);
  status = worker.Rerank(encoded_rerank_req.c_str(), encoded_rerank_req.size(),
                         resp_rerank.get(), 4096 * 4096);

  if (status) {
    printf("Output: %s\n", resp_rerank.get());
  } else {
    printf("Failed to rerank\n");
    exit(1);
  }

  // decode the rerank result
  hakes::SearchWorkerRerankResponse rerank_resp;
  status = hakes::decode_search_worker_rerank_response(resp_rerank.get(),
                                                       &rerank_resp);

  if (!status) {
    printf("Failed to decode rerank result\n");
    exit(1);
  } else {
    size_t content_len = 0;
    auto ids = hakes::decode_hex_int64s(rerank_resp.ids, &content_len);
    printf("ids size: %ld\n", content_len);
    auto scores = hakes::decode_hex_floats(rerank_resp.scores, &content_len);
    printf("scores size: %ld\n", content_len);
    for (int i = 0; i < content_len; i++) {
      printf("id: %ld, scores: %f\n", ids[i], scores[i]);
    }
  }

  worker.Close();
  delete[] xb;
  delete[] xq;
  return 0;
}
