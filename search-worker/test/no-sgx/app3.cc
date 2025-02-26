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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <memory>

#include "message/searchservice.h"
#include "search-worker/common/worker.h"
#include "search-worker/common/workerImpl.h"
#include "utils/data_loader.h"
#include "utils/fileutil.h"

struct Config {
  size_t data_n = 1183514;
  size_t data_num_query = 1000;
  size_t data_dim = 200;
  size_t search_k = 10;
  size_t data_groundtruth_len = 100;
  std::string data_train_path;
  std::string data_query_path;
  std::string data_groundtruth_path;
  // index params
  int nprobe = 1024;
  int k_factor = 50;
  std::string index_path;
};

Config parse_config(int argc, char** argv) {
  Config cfg;
  if (argc < 12) {
    std::cout
        << "Usage: " << argv[0]
        << " DATA_N DATA_NUM_QUERY DATA_DIM SEARCH_K DATA_GROUNDTRUTH_LEN "
           "DATA_TRAIN_PATH DATA_QUERY_PATH DATA_GROUNDTRUTH_PATH NPROBE "
           "K_FACTOR INDEX_PATH"
        << std::endl;
    exit(1);
  }

  cfg.data_n = std::stoul(argv[1]);
  cfg.data_num_query = std::stoul(argv[2]);
  cfg.data_dim = std::stoul(argv[3]);
  cfg.search_k = std::stoul(argv[4]);
  cfg.data_groundtruth_len = std::stoul(argv[5]);
  cfg.data_train_path = argv[6];
  cfg.data_query_path = argv[7];
  cfg.data_groundtruth_path = argv[8];
  cfg.nprobe = std::stoi(argv[9]);
  cfg.k_factor = std::stoi(argv[10]);
  cfg.index_path = argv[11];

  std::cout << "DATA_N: " << cfg.data_n << std::endl;
  std::cout << "DATA_NUM_QUERY: " << cfg.data_num_query << std::endl;
  std::cout << "DATA_DIM: " << cfg.data_dim << std::endl;
  std::cout << "SEARCH_K: " << cfg.search_k << std::endl;
  std::cout << "DATA_GROUNDTRUTH_LEN: " << cfg.data_groundtruth_len
            << std::endl;
  std::cout << "DATA_TRAIN_PATH: " << cfg.data_train_path << std::endl;
  std::cout << "DATA_QUERY_PATH: " << cfg.data_query_path << std::endl;
  std::cout << "DATA_GROUNDTRUTH_PATH: " << cfg.data_groundtruth_path
            << std::endl;
  std::cout << "NPROBE: " << cfg.nprobe << std::endl;
  std::cout << "K_FACTOR: " << cfg.k_factor << std::endl;
  std::cout << "INDEX_PATH: " << cfg.index_path << std::endl;
  return cfg;
}

int main(int argc, char* argv[]) {
  // Config cfg;
  Config cfg = parse_config(argc, argv);

  search_worker::WorkerImpl worker0{};
  search_worker::WorkerImpl worker1{};
  search_worker::WorkerImpl worker2{};

  // load index file
  size_t content_len = 0;
  auto content =
      hakes::ReadFileToCharArray(cfg.index_path.c_str(), &content_len);
  printf("content_len: %ld\n", content_len);

  {
    auto r = hakes::StringIOReader(content.get(), content_len);
    bool status = worker0.Initialize("main", &r, nullptr, nullptr, false, 3, 0);
    if (!status) {
      printf("Failed to initialize\n");
      exit(1);
    }
  }
  {
    auto r = hakes::StringIOReader(content.get(), content_len);
    bool status = worker1.Initialize("main", &r, nullptr, nullptr, false, 3, 1);
    if (!status) {
      printf("Failed to initialize\n");
      exit(1);
    }
  }
  {
    auto r = hakes::StringIOReader(content.get(), content_len);
    bool status = worker2.Initialize("main", &r, nullptr, nullptr, false, 3, 2);
    if (!status) {
      printf("Failed to initialize\n");
      exit(1);
    }
  }
  // load data files
  float* data =
      load_data(cfg.data_train_path.c_str(), cfg.data_dim, cfg.data_n);
  float* query =
      load_data(cfg.data_query_path.c_str(), cfg.data_dim, cfg.data_num_query);
  int* groundtruth =
      load_groundtruth(cfg.data_groundtruth_path.c_str(),
                       cfg.data_groundtruth_len, cfg.data_num_query);

  // add

  for (int i = 0; i < 10; i++) {
    hakes::SearchWorkerAddRequest add_req;
    add_req.d = cfg.data_dim;
    add_req.vecs =
        hakes::encode_hex_floats(data + i * cfg.data_dim, cfg.data_dim);
    int64_t ids[1] = {i};
    add_req.ids = hakes::encode_hex_int64s(ids, 1);

    std::string encoded_req = hakes::encode_search_worker_add_request(add_req);

    auto resp = std::unique_ptr<char[]>(new char[4096]);
    {
      auto success = worker0.AddWithIds(encoded_req.c_str(), encoded_req.size(),
                                        resp.get(), 4096);
      if (success) {
        printf("Output: %s\n", resp.get());
      } else {
        printf("Failed to add\n");
        exit(1);
      }
    }
    {
      auto success = worker1.AddWithIds(encoded_req.c_str(), encoded_req.size(),
                                        resp.get(), 4096);
      if (success) {
        printf("Output: %s\n", resp.get());
      } else {
        printf("Failed to add\n");
        exit(1);
      }
    }
    {
      auto success = worker2.AddWithIds(encoded_req.c_str(), encoded_req.size(),
                                        resp.get(), 4096);
      if (success) {
        printf("Output: %s\n", resp.get());
      } else {
        printf("Failed to add\n");
        exit(1);
      }
    }
  }

  // search the first vector
  hakes::SearchWorkerSearchRequest search_req;
  search_req.d = cfg.data_dim;
  search_req.vecs = hakes::encode_hex_floats(query, cfg.data_dim);
  search_req.k = cfg.search_k;
  search_req.nprobe = cfg.nprobe;
  search_req.k_factor = cfg.k_factor;
  search_req.metric_type = 1;

  std::string encoded_search_req =
      hakes::encode_search_worker_search_request(search_req);

  auto resp = std::unique_ptr<char[]>(new char[4096 * 4096]);
  auto status =
      worker0.Search(encoded_search_req.c_str(), encoded_search_req.size(),
                     resp.get(), 4096 * 4096);

  if (status) {
    printf("Output: %s\n", resp.get());
  } else {
    printf("Failed to search\n");
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
  rerank_req.d = cfg.data_dim;
  rerank_req.k = cfg.search_k;
  rerank_req.nprobe = cfg.nprobe;
  rerank_req.metric_type = 1;
  rerank_req.vecs = search_req.vecs;
  rerank_req.input_ids = search_resp.ids;

  std::string encoded_rerank_req =
      hakes::encode_search_worker_rerank_request(rerank_req);

  {
    auto resp_rerank = std::unique_ptr<char[]>(new char[4096 * 4096]);
    status =
        worker0.Rerank(encoded_rerank_req.c_str(), encoded_rerank_req.size(),
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
  }
  {
    auto resp_rerank = std::unique_ptr<char[]>(new char[4096 * 4096]);
    status =
        worker1.Rerank(encoded_rerank_req.c_str(), encoded_rerank_req.size(),
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
  }
  {
    auto resp_rerank = std::unique_ptr<char[]>(new char[4096 * 4096]);
    status =
        worker2.Rerank(encoded_rerank_req.c_str(), encoded_rerank_req.size(),
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
  }
  worker0.Close();
  worker1.Close();
  worker2.Close();

  delete[] data;
  delete[] query;
  delete[] groundtruth;
  return 0;
}
