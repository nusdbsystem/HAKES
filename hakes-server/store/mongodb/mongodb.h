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

#ifndef HAKES_STORE_MONGODB_H_
#define HAKES_STORE_MONGODB_H_

#include "../store.h"
#include <mongocxx/client.hpp>
#include <mongocxx/database.hpp>
#include <mongocxx/collection.hpp>
#include <bsoncxx/document/view.hpp>
#include <memory>
#include <cstdint>
#include <vector>
#include <string>

namespace hakes {

/**
 * @brief MongoDB implementation of the Store interface.
 * 
 * This class provides MongoDB-specific functionality for storing and retrieving
 * key-value pairs with optional external IDs.
 */
class MongoDB : public Store {
 public:
  /**
   * @brief Constructor for MongoDB store.
   * 
   * @param addr The address of the MongoDB server (e.g., "mongodb://localhost:27017").
   * @param db_name The database name (default: "hakes").
   * @param collection_name The collection name (default: "default").
   * 
   * @throws std::runtime_error If MongoDB connection fails.
   */
  MongoDB(const std::string& addr,
          const std::string& db_name = "hakes",
          const std::string& collection_name = "default");

  /**
   * @brief Destructor. Closes the MongoDB connection.
   */
  ~MongoDB() override;

  /**
   * @brief Check if connected to MongoDB.
   * 
   * @return true if connected, false otherwise.
   */
  bool Connected() override;

  /**
   * @brief Connect to MongoDB (no-op, handled by constructor).
   */
  void Connect() override;

  /**
   * @brief Disconnect from MongoDB.
   */
  void Disconnect() override;

  /**
   * @brief Put key-value pairs into MongoDB.
   * 
   * Auto-increments xid if not provided. Always stores xids as 8-byte (64-bit signed) values.
   * 
   * @param keys A vector of keys to associate with the values.
   * @param values A vector of values to store.
   * @param xids Optional vector of external IDs. If nullptr, auto-increments.
   * 
   * @return A pair containing:
   *         - bool: true if successful, false otherwise
   *         - std::vector<std::string>: IDs that were successfully stored
   */
  std::pair<bool, std::vector<std::string>> Put(
      const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values,
      const std::vector<std::vector<uint8_t>>* xids = nullptr) override;

  /**
   * @brief Get values and xids by a list of keys.
   * 
   * @param keys A vector of keys to retrieve.
   * 
   * @return A pair containing:
   *         - std::vector<std::vector<uint8_t>>: values in the same order as input keys
   *         - std::vector<std::vector<uint8_t>>: xids in the same order as input keys
   *         If a key is missing, returns empty vector for value and xid.
   */
  std::pair<std::vector<std::vector<uint8_t>>, std::vector<std::vector<uint8_t>>> 
  GetByKeys(const std::vector<std::string>& keys) override;

  /**
   * @brief Get values by a list of external IDs.
   * 
   * Preserves the order of input xids in the output.
   * 
   * @param xids A vector of external IDs to retrieve.
   * 
   * @return A vector of values in the same order as input xids.
   *         If an ID is missing, returns empty vector for that value.
   */
  std::vector<std::vector<uint8_t>> GetByIds(
      const std::vector<std::vector<uint8_t>>& xids) override;

  /**
   * @brief Delete a key-value pair from MongoDB.
   * 
   * @param key The key to delete.
   * 
   * @return true if successful, false otherwise.
   */
  bool Delete(const std::string& key) override;

 private:
  /**
   * @brief Validate and format the MongoDB address.
   * 
   * @param addr The address to validate.
   * @return The validated and formatted MongoDB address.
   * 
   * @throws std::invalid_argument If the address is empty or invalid.
   */
  static std::string ValidateMongoDBAddress(const std::string& addr);

  /**
   * @brief Get the next batch of auto-incremented xids.
   * 
   * XIDs are always 8-byte (64-bit signed) values.
   * 
   * @param batch_size The number of IDs to generate.
   * @return A vector of generated xids in big-endian format.
   */
  std::vector<std::vector<uint8_t>> GetNextXids(int batch_size);

  /**
   * @brief Extract value from MongoDB document.
   * 
   * @param doc The MongoDB document.
   * @return The value as a vector of uint8_t, or empty vector if not found.
   */
  static std::vector<uint8_t> ExtractValueFromDoc(
      const bsoncxx::document::view& doc);

  /**
   * @brief Extract xid from MongoDB document.
   * 
   * @param doc The MongoDB document.
   * @return The xid as a vector of uint8_t, or empty vector if not found.
   */
  static std::vector<uint8_t> ExtractXidFromDoc(
      const bsoncxx::document::view& doc);

  std::shared_ptr<mongocxx::client> client_;
  mongocxx::database db_;
  mongocxx::collection collection_;
  mongocxx::collection counters_;
  std::string counter_key_;
};

}  // namespace hakes

#endif  // HAKES_STORE_MONGODB_H_
