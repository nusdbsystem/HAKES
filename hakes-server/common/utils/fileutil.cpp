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

#include "fileutil.h"

#include <cassert>
#include <sstream>

namespace hakes {

int ReadFileToString(const std::string& file_path, std::string* content) {
  assert(content);
  std::ifstream infile(file_path);
  if (!infile.good()) {
    printf("file failed to load: %s\n", file_path.c_str());
    return 0;  // cannot differentiate file not exist or error in openning
  }
  std::stringstream ss;
  ss << infile.rdbuf();
  *content = ss.str();
  return 0;
}

/**
 * @brief read the content of a file into a char array.
 *
 * @param file_path
 * @param output_len output_len will be updated with the len of read content,
 *  if succeeded
 * @return char* : read content.
 */
std::unique_ptr<char[]> ReadFileToCharArray(const char* file_path,
                                            size_t* output_len) {
  assert(file_path);
  assert(output_len);
  FILE* fp = std::fopen(file_path, "rb");
  if (fp == nullptr) {
    printf("failed to open file: %s\n", file_path);
    return nullptr;
  }
  if (std::fseek(fp, 0, SEEK_END) == -1) {
    printf("fseek error\n");
    return nullptr;
  }
  auto file_len = std::ftell(fp);
  if (file_len == -1) {
    printf("ftell error\n");
    return nullptr;
  }
  if (file_len == 0) {
    *output_len = 0;
    return nullptr;
  }
  auto ret = std::unique_ptr<char[]>(new char[file_len + 1]);
  if (ret == nullptr) {
    printf("malloc failed\n");
    return nullptr;
  }
  std::rewind(fp);
  if (std::fread(ret.get(), 1, file_len, fp) != static_cast<size_t>(file_len)) {
    printf("fread error");
    return nullptr;
  }
  if (std::fclose(fp) == -1) {
    printf("close error");
    return nullptr;
  }
  *output_len = file_len;
  ret.get()[file_len] = '\0';
  return ret;
}

int WriteStringToFile(const std::string& file_path,
                      const std::string& content) {
  std::ofstream outfile(file_path, std::ios::out);
  outfile << content;
  outfile.flush();
  outfile.close();
  return 0;
}

/**
 * @brief write a char array to a file
 *
 * @param file_path
 * @param src
 * @param len : len of the char array
 * @return int : 0 for success; -1 for failure
 */
int WriteCharArrayToFile(const char* file_path, const char* src, size_t len) {
  assert(file_path);
  assert(src);
  FILE* fp = std::fopen(file_path, "wb");
  if (fp == nullptr) {
    printf("failed to open file\n");
    return -1;
  }
  if (std::fwrite(src, 1, len, fp) != len) {
    printf("fwrite error");
    return -1;
  }
  if (std::fclose(fp) == -1) {
    printf("close error");
    return -1;
  }
  return 0;
}

}  // namespace hakes
