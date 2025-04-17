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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>

#include "message/searchservice.h"
#include "search-worker/common/worker.h"
#include "search-worker/common/workerImpl.h"
#include "search-worker/index/ext/HakesCollection.h"
#include "search-worker/index/ext/HakesFlatIndex.h"
#include "search-worker/index/ext/HakesIndex.h"
#include "utils/data_loader.h"
#include "utils/fileutil.h"

struct Config {
  std::string type = "hakes";
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
  std::string update_path = "";
  std::string save_path = ".";
};

std::string trim(std::string origin) {
  size_t start = origin.find_first_not_of(" \t\n\r\"");
  auto result = (start == std::string::npos) ? "" : origin.substr(start);
  size_t end = result.find_last_not_of(" \t\n\r\"");
  return (end == std::string::npos) ? "" : result.substr(0, end + 1);
}

Config parse_config(int argc, char** argv) {
  if (argc < 1) {
    std::cout << "Usage: " << argv[0] << " CONFIG_FILE" << std::endl;
    exit(1);
  }
  std::ifstream file(argv[1]);

  if (!file.is_open()) {
    std::cerr << "Failed to open file!" << std::endl;
    exit(1);
  }

  std::string line;
  Config cfg;
  while (std::getline(file, line)) {
    auto idx = line.find("=");
    auto key = trim(line.substr(0, idx));
    auto val = trim(line.substr(idx + 1));
    if (key.compare("data_n") == 0) {
      cfg.data_n = stoi(val);
    } else if (key.compare("data_num_query") == 0) {
      cfg.data_num_query = stoi(val);
    } else if (key.compare("data_dim") == 0) {
      cfg.data_dim = stoi(val);
    } else if (key.compare("search_k") == 0) {
      cfg.search_k = stoi(val);
    } else if (key.compare("data_groundtruth_len") == 0) {
      cfg.data_groundtruth_len = stoi(val);
    } else if (key.compare("data_train_path") == 0) {
      cfg.data_train_path = val;
    } else if (key.compare("data_query_path") == 0) {
      cfg.data_query_path = val;
    } else if (key.compare("data_groundtruth_path") == 0) {
      cfg.data_groundtruth_path = val;
    } else if (key.compare("nprobe") == 0) {
      cfg.nprobe = stoi(val);
    } else if (key.compare("k_factor") == 0) {
      cfg.k_factor = stoi(val);
    } else if (key.compare("index_path") == 0) {
      cfg.index_path = val;
    } else if (key.compare("update_path") == 0) {
      cfg.update_path = val;
    } else if (key.compare("save_path") == 0) {
      cfg.save_path = val;
    } else if (key.compare("type") == 0) {
      cfg.type = val;
    }
  }
  file.close();

  std::cout << "TYPE: " << cfg.type << std::endl;
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
  std::cout << "SAVE_PATH: " << cfg.save_path << std::endl;
  std::cout << "INDEX_PATH: " << cfg.index_path << std::endl;
  std::cout << "UPDATE_PATH: " << cfg.update_path << std::endl;
  return cfg;
}

int testHakesFlatIndex(Config cfg) {
  int n = cfg.data_n;
  int d = cfg.data_dim;
  int nq = cfg.data_num_query;
  int k = cfg.search_k;
  auto metric = faiss::MetricType::METRIC_L2;

  // generate vectors
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

  // build index
  std::unique_ptr<faiss::HakesFlatIndex> index(
      new faiss::HakesFlatIndex(d, metric));
  auto xids = std::make_unique<faiss::idx_t[]>(n);
  for (int i = 0; i < n; i++) {
    xids[i] = i;
  }
  std::unique_ptr<faiss::idx_t[]> assign = std::make_unique<faiss::idx_t[]>(n);
  int vecs_t_d;
  std::unique_ptr<float[]> vecs_t;
  index->AddWithIds(n, d, xb, xids.get(), assign.get(), &vecs_t_d, &vecs_t);
  std::cout << "Index: " << index->to_string() << std::endl;

  // search
  faiss::HakesSearchParams param;
  param.k = k;
  param.metric_type = metric;
  std::unique_ptr<float[]> distances;
  std::unique_ptr<faiss::idx_t[]> labels;
  index->Search(nq, d, xq, param, &distances, &labels);
  std::cout << "Search complete!" << std::endl;

  // save index
  if (!index->Checkpoint(cfg.save_path)) {
    std::cerr << "Failed to save index!" << std::endl;
    return 1;
  }
  std::cout << "Save complete!" << std::endl;

  // reload index
  std::unique_ptr<faiss::HakesFlatIndex> index2(
      new faiss::HakesFlatIndex(d, metric));
  if (!index2->Initialize(cfg.save_path, 0, false)) {
    std::cerr << "Failed to load index!" << std::endl;
    return 1;
  }
  std::cout << "Load complete!" << std::endl;

  return 0;
}

int testHakesIndex(Config cfg) {
  int n = cfg.data_n;
  int d = cfg.data_dim;
  int nq = cfg.data_num_query;
  int gt_len = cfg.data_groundtruth_len;
  int k = cfg.search_k;
  // search_worker::WorkerImpl worker{};
  std::unique_ptr<faiss::HakesIndex> index(new faiss::HakesIndex());
  index->use_refine_sq_ = false;

  if (!index->Initialize(cfg.index_path, 0, false)) {
    std::cerr << "Failed to initialize index!" << std::endl;
    return 1;
  }

  if (!cfg.update_path.empty()) {
    size_t content_len = 0;
    auto content =
        hakes::ReadFileToCharArray(cfg.update_path.c_str(), &content_len);
    index->UpdateParams(
        std::string(content.get(), content_len));  // update index
  }

  std::cout << "Index loaded" << std::endl;
  std::cout << index->to_string() << std::endl;

  if (false) {
    auto base_index =
        dynamic_cast<faiss::IndexIVFPQFastScanL*>(index->base_index_.get());
    if (base_index == nullptr) {
      std::cerr << "base index is not IndexIVFPQFastScanL" << std::endl;
      return 1;
    }
    int nlist = base_index->quantizer->ntotal;

    auto vt = dynamic_cast<faiss::LinearTransform*>(index->vts_[0]);
    printf("########## vt (%dx%d) ##########\n", vt->d_out, vt->d_in);
    printf("A\n");
    for (int i = 0; i < vt->d_out; i++) {
      for (int j = 0; j < vt->d_in; j++) {
        printf("%8.4f ", vt->A[i * vt->d_in + j]);
      }
      printf("\n");
    }
    if (vt->have_bias) {
      printf("b\n");
      for (int i = 0; i < vt->d_out; i++) {
        printf("%8.4f ", vt->b[i]);
      }
    } else {
      printf("no bias\n");
    }
    printf("########## vt (%dx%d) ##########\n", vt->d_out, vt->d_in);

    // dispplay the centroids of the coarse quantizer
    printf("########## centroids (count: %d) ##########\n", nlist);
    if (!index->use_ivf_sq_) {
      auto cq = dynamic_cast<faiss::IndexFlatL*>(base_index->quantizer);
      const float* centroids = cq->get_xb();
      for (int i = 0; i < nlist; i++) {
        printf("c%04d: ", i);
        for (int j = 0; j < cq->d; j++) {
          printf("%8.4f ", centroids[i * cq->d + j]);
        }
        printf("\n");
      }
    } else {
      printf("use ivf sq\n");
    }
    printf("########## centroids (count: %d) ##########\n", nlist);
    // int ksub = index->pq.ksub;
    // int dsub = index->pq.dsub;
    int ksub = base_index->pq.ksub;
    int dsub = base_index->pq.dsub;
    int pqm = base_index->pq.M;
    printf("########## pq codebook (%dx%dx%d) ##########\n", pqm, ksub, dsub);
    float* codebook = base_index->pq.centroids.data();
    for (int j = 0; j < ksub; j++) {
      printf("|");
      for (int m = 0; m < pqm; m++) {
        for (int k = 0; k < dsub; k++) {
          printf("%8.4f ", codebook[(m * ksub + j) * dsub + k]);
        }
        printf("|");
      }
      printf("\n");
    }
    printf("########## pq codebook (%dx%dx%d) ##########\n", pqm, ksub, dsub);
    if (index->has_q_index_) {
      auto vt = dynamic_cast<faiss::LinearTransform*>(index->q_vts_[0]);
      printf("########## vt (%dx%d) ##########\n", vt->d_out, vt->d_in);
      printf("A\n");
      for (int i = 0; i < vt->d_out; i++) {
        for (int j = 0; j < vt->d_in; j++) {
          printf("%8.4f ", vt->A[i * vt->d_in + j]);
        }
        printf("\n");
      }
      if (vt->have_bias) {
        printf("b\n");
        for (int i = 0; i < vt->d_out; i++) {
          printf("%8.4f ", vt->b[i]);
        }
      } else {
        printf("no bias\n");
      }
      printf("########## vt (%dx%d) ##########\n", vt->d_out, vt->d_in);
      printf("########## centroids (count: %d) ##########\n", nlist);
      if (!index->use_ivf_sq_) {
        auto cq = dynamic_cast<faiss::IndexFlatL*>(index->q_cq_);
        const float* centroids = cq->get_xb();
        for (int i = 0; i < nlist; i++) {
          printf("c%04d: ", i);
          for (int j = 0; j < cq->d; j++) {
            printf("%8.4f ", centroids[i * cq->d + j]);
          }
          printf("\n");
        }
      } else {
        printf("use ivf sq\n");
      }
      printf("########## centroids (count: %d) ##########\n", nlist);
    }
    if (base_index->has_q_pq) {
      printf("########## qpq codebook (%dx%dx%d) ##########\n", pqm, ksub,
             dsub);
      float* codebook = base_index->q_pq.centroids.data();
      for (int j = 0; j < ksub; j++) {
        printf("|");
        for (int m = 0; m < pqm; m++) {
          for (int k = 0; k < dsub; k++) {
            printf("%8.4f ", codebook[(m * ksub + j) * dsub + k]);
          }
          printf("|");
        }
        printf("\n");
      }
      printf("########## pq codebook (%dx%dx%d) ##########\n", pqm, ksub, dsub);
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
  auto xids = std::make_unique<faiss::idx_t[]>(n);
  for (int i = 0; i < n; i++) {
    xids[i] = i;
  }
  std::unique_ptr<faiss::idx_t[]> assign = std::make_unique<faiss::idx_t[]>(n);
  int vecs_t_d;
  std::unique_ptr<float[]> vecs_t;

  auto batch_size = 100000;
  // auto batch_size = 100;
  for (int i = 0; i < n; i += batch_size) {
    index->AddWithIds(std::min(batch_size, n - i), d, data + i * d,
                      xids.get() + i, assign.get() + i, &vecs_t_d, &vecs_t);

    std::cout << "Inserted " << i << " vector" << std::endl;
  }

  std::cout << "Index: " << index->to_string() << std::endl;

  // save index
  if (!index->Checkpoint(cfg.save_path)) {
    std::cerr << "Failed to save index!" << std::endl;
    return 1;
  }

  {
    index->base_index_->use_early_termination_ = true;
    index->base_index_->et_params.beta = 200;
    index->base_index_->et_params.ce = 30;
  }

  auto nprobe_list = std::vector<int>{1, 5, 10, 50, 100, 200, 300};
  auto k_factor_list = std::vector<int>{1, 10, 20, 50, 100, 200, 300};

  for (auto nprobe : nprobe_list) {
    for (auto k_factor : k_factor_list) {
      auto k_base = k * k_factor;
      faiss::HakesSearchParams params{nprobe, k, k_factor,
                                      faiss::METRIC_INNER_PRODUCT};

      auto result = std::make_unique<faiss::idx_t[]>(k * nq);
      auto start = std::chrono::high_resolution_clock::now();

      for (int i = 0; i < nq; ++i) {
        // printf("\nquery %d: ", i);
        std::unique_ptr<faiss::idx_t[]> candidates;
        std::unique_ptr<float[]> distances;
        index->Search(1, d, query + i * d, params, &distances, &candidates);
        std::unique_ptr<faiss::idx_t[]> k_base_count =
            std::make_unique<faiss::idx_t[]>(1);
        k_base_count[0] = k_base;
        index->Rerank(1, d, query + i * d, k, k_base_count.get(),
                      candidates.get(), distances.get(), &distances,
                      &candidates);
        std::memcpy(result.get() + i * k, candidates.get(),
                    k * sizeof(faiss::idx_t));
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = end - start;

      int correct = 0;
      for (int i = 0; i < nq; i++) {
        std::unordered_set<int> gts(k);
        for (int j = 0; j < k; j++) {
          gts.insert(groundtruth[i * gt_len + j]);
        }

        for (int j = 0; j < k; j++) {
          if (gts.find(result[i * k + j]) != gts.end()) {
            correct++;
          }
        }
      }

      std::cout << "nprobe: " << nprobe << " k_factor: " << k_factor
                << " search time (s): " << diff.count()
                << " recall: " << correct / (float)(nq * k) << std::endl;
    }
  }

  // test reload
  faiss::HakesIndex index2;
  if (!index2.Initialize(cfg.save_path, 0, false)) {
    std::cerr << "Failed to load index!" << std::endl;
    return 1;
  }
  std::cout << "Index2 loaded" << std::endl;
  std::cout << index2.to_string() << std::endl;

  {
    index2.base_index_->use_early_termination_ = true;
    index2.base_index_->et_params.beta = 200;
    index2.base_index_->et_params.ce = 30;
  }

  auto nprobe_list2 = std::vector<int>{1, 5, 10, 50, 100, 200, 300};
  auto k_factor_list2 = std::vector<int>{1, 10, 20, 50, 100, 200, 300};

  for (auto nprobe : nprobe_list2) {
    for (auto k_factor : k_factor_list2) {
      auto k_base = k * k_factor;
      faiss::HakesSearchParams params{nprobe, k, k_factor,
                                      faiss::METRIC_INNER_PRODUCT};

      auto result = std::make_unique<faiss::idx_t[]>(k * nq);
      auto start = std::chrono::high_resolution_clock::now();

      for (int i = 0; i < nq; ++i) {
        // printf("\nquery %d: ", i);
        std::unique_ptr<faiss::idx_t[]> candidates;
        std::unique_ptr<float[]> distances;
        index2.Search(1, d, query + i * d, params, &distances, &candidates);
        std::unique_ptr<faiss::idx_t[]> k_base_count =
            std::make_unique<faiss::idx_t[]>(1);
        k_base_count[0] = k_base;
        index2.Rerank(1, d, query + i * d, k, k_base_count.get(),
                      candidates.get(), distances.get(), &distances,
                      &candidates);
        std::memcpy(result.get() + i * k, candidates.get(),
                    k * sizeof(faiss::idx_t));
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = end - start;

      int correct = 0;
      for (int i = 0; i < nq; i++) {
        std::unordered_set<int> gts(k);
        for (int j = 0; j < k; j++) {
          gts.insert(groundtruth[i * gt_len + j]);
        }

        for (int j = 0; j < k; j++) {
          if (gts.find(result[i * k + j]) != gts.end()) {
            correct++;
          }
        }
      }

      std::cout << "nprobe: " << nprobe << " k_factor: " << k_factor
                << " search time (s): " << diff.count()
                << " recall: " << correct / (float)(nq * k) << std::endl;
    }
  }

  delete[] data;
  delete[] query;
  delete[] groundtruth;
  return 0;
}

int main(int argc, char* argv[]) {
  // Config cfg;
  Config cfg = parse_config(argc, argv);

  if (cfg.type.compare("hakes") == 0) {
    testHakesIndex(cfg);
  } else if (cfg.type.compare("hakesflat") == 0) {
    testHakesFlatIndex(cfg);
  } else {
    std::cerr << "Unknown index type. Please input hakes|hakesflat"
              << std::endl;
    exit(1);
  }
  return 0;
}