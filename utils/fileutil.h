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

#ifndef HAKES_UTILS_FILEUTIL_H_
#define HAKES_UTILS_FILEUTIL_H_

#include <fstream>
#include <memory>

namespace hakes {

inline bool IsFileExist(const std::string& file_path) {
  std::ifstream infile(file_path);
  return infile.good();
}

int ReadFileToString(const std::string& file_path, std::string* content);

/**
 * @brief read the content of a file into a char array.
 *
 * @param file_path
 * @param output_len output_len will be updated with the len of read content,
 *  if succeeded
 * @return char* : read content.
 */
std::unique_ptr<char[]> ReadFileToCharArray(const char* file_path,
                                            size_t* output_len);

int WriteStringToFile(const std::string& file_path, const std::string& content);

/**
 * @brief write a char array to a file
 *
 * @param file_path
 * @param src
 * @param len : len of the char array
 * @return int : 0 for success; -1 for failure
 */
int WriteCharArrayToFile(const char* file_path, const char* src, size_t len);

}  // namespace hakes

#endif  // HAKES_UTILS_FILEUTIL_H_
