/*
 * Copyright 2024 The HAKES Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HAKES_SERVER_STORE_H_
#define HAKES_SERVER_STORE_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "store-client/models/user.h"

namespace hakes_store {

/**
 * @brief Interface for external storage services.
 *
 * This is an abstract base class that defines the interface for storage
 * operations. Subclasses should implement specific storage backends (e.g.,
 * MongoDB, etc.).
 */
class Store {
 public:
  /**
   * @brief Constructor for the Store.
   *
   * @param addr The address of the storage service.
   */
  explicit Store(const std::string& addr) : addr_(addr) {}

  /**
   * @brief Virtual destructor.
   */
  virtual ~Store() = default;

  /**
   * @brief Put key-value pairs into the store.
   *
   * @param keys A vector of keys to associate with the values.
   * @param values A vector of values to store (as bytes).
   * @param xids Optional vector of external IDs to associate with the keys.
   *             If nullptr, implementation may auto-generate IDs.
   *
   * @return A pair containing:
   *         - bool: true if successful, false otherwise
   *         - std::vector<std::string>: IDs that were successfully stored
   */
  virtual std::pair<bool, std::vector<std::string>> Put(
      const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values,
      const std::vector<std::vector<uint8_t>>* xids = nullptr) = 0;

  /**
   * @brief Get values and xids by a list of keys.
   *
   * @param keys A vector of keys to retrieve.
   *
   * @return A pair containing:
   *         - std::vector<std::vector<uint8_t>>: values in the same order as
   * input keys
   *         - std::vector<std::vector<uint8_t>>: xids in the same order as
   * input keys If a key is missing, returns empty vector for value and xid.
   */
  virtual std::pair<std::vector<std::vector<uint8_t>>,
                    std::vector<std::vector<uint8_t>>>
  GetByKeys(const std::vector<std::string>& keys) = 0;

  /**
   * @brief Get values by a list of external IDs.
   *
   * @param xids A vector of external IDs to retrieve.
   *
   * @return A vector of values in the same order as input xids.
   *         If an ID is missing, returns empty vector for that value.
   */
  virtual std::vector<std::vector<uint8_t>> GetByIds(
      const std::vector<std::vector<uint8_t>>& xids) = 0;

  /**
   * @brief Delete a key-value pair from the store.
   *
   * @param key The key to delete.
   *
   * @return true if successful, false otherwise.
   */
  virtual bool Delete(const std::string& key) = 0;

  /**
   * @brief Check if connected to the storage service.
   *
   * @return true if connected, false otherwise.
   */
  virtual bool Connected() = 0;

  /**
   * @brief Connect to the storage service.
   */
  virtual void Connect() {}

  /**
   * @brief Disconnect from the storage service.
   */
  virtual void Disconnect() {}

  virtual bool GetByUsername(const std::string& username, User* user) = 0;

  virtual bool CreateUser(const User& user) = 0;

  virtual bool UpdateUser(const User& user) = 0;

  virtual bool DeleteUser(const std::string& username) = 0;

 protected:
  std::string addr_;
};

}  // namespace hakes_store

#endif  // HAKES_SERVER_STORE_H_
