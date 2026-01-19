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

#include "mongodb.h"

#include <mongocxx/client.hpp>
#include <mongocxx/instance.hpp>
#include <mongocxx/uri.hpp>
#include <bsoncxx/builder/stream/document.hpp>
#include <bsoncxx/builder/stream/array.hpp>
#include <bsoncxx/document/view.hpp>
#include <bsoncxx/types.hpp>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <cstdint>

namespace hakes {

using bsoncxx::builder::stream::document;
using bsoncxx::builder::stream::array;
using bsoncxx::builder::stream::finalize;
using bsoncxx::builder::stream::open_document;
using bsoncxx::builder::stream::close_document;

// Global MongoDB instance (must be created once per process)
static mongocxx::instance inst{};

std::string MongoDB::ValidateMongoDBAddress(const std::string& addr) {
  if (addr.empty()) {
    throw std::invalid_argument("MongoDB address is empty");
  }

  if (addr.find("mongodb://") == 0 || addr.find("mongodb+srv://") == 0) {
    return addr;
  }

  return "mongodb://" + addr;
}

MongoDB::MongoDB(const std::string& addr,
                 const std::string& db_name,
                 const std::string& collection_name)
    : Store(ValidateMongoDBAddress(addr)),
      counter_key_(collection_name) {
  try {
    // Create MongoDB client
    client_ = std::make_shared<mongocxx::client>(mongocxx::uri{addr_});

    // Get database and collection
    db_ = client_->database(db_name);
    collection_ = db_.collection(collection_name);
    counters_ = db_.collection("counters");

    // Create unique index on key for fast lookup
    auto index_opts = mongocxx::options::index{};
    collection_.create_index(document{} << "key" << 1 << finalize, index_opts);

    // Initialize counter if not present
    auto counter_doc = counters_.find_one(
        document{} << "_id" << counter_key_ << finalize);

    if (!counter_doc) {
      counters_.insert_one(
          document{} << "_id" << counter_key_ << "seq" << int64_t(0) << finalize);
    }
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Failed to connect to MongoDB: ") + e.what());
  }
}

MongoDB::~MongoDB() {
  Disconnect();
}

bool MongoDB::Connected() {
  try {
    if (!client_) return false;
    auto admin = client_->database("admin");
    auto result = admin.run_command(document{} << "ismaster" << 1 << finalize);
    // If we get here, we're connected
    return true;
  } catch (...) {
    return false;
  }
}

void MongoDB::Connect() {
  // No-op, handled by constructor
}

void MongoDB::Disconnect() {
  client_.reset();
}

std::vector<std::vector<uint8_t>> MongoDB::GetNextXids(int batch_size) {
  try {
    // Atomically increment the counter and get the starting value
    auto opts = mongocxx::options::find_one_and_update{};
    opts.return_document(mongocxx::options::return_document::k_after);
    opts.upsert(true);

    auto result = counters_.find_one_and_update(
        document{} << "_id" << counter_key_ << finalize,
        document{} << "$inc" << open_document << "seq" << batch_size
                   << close_document << finalize,
        opts);

    if (!result) {
      throw std::runtime_error("Failed to increment counter");
    }

    auto doc_view = result->view();
    int64_t last_id = doc_view["seq"].get_int64();
    int64_t first_id = last_id - batch_size + 1;

    // Generate xids as 8-byte big-endian values
    std::vector<std::vector<uint8_t>> xids;
    xids.reserve(batch_size);

    for (int64_t i = first_id; i <= last_id; ++i) {
      std::vector<uint8_t> xid(8);
      // Big-endian encoding
      xid[0] = static_cast<uint8_t>((i >> 56) & 0xFF);
      xid[1] = static_cast<uint8_t>((i >> 48) & 0xFF);
      xid[2] = static_cast<uint8_t>((i >> 40) & 0xFF);
      xid[3] = static_cast<uint8_t>((i >> 32) & 0xFF);
      xid[4] = static_cast<uint8_t>((i >> 24) & 0xFF);
      xid[5] = static_cast<uint8_t>((i >> 16) & 0xFF);
      xid[6] = static_cast<uint8_t>((i >> 8) & 0xFF);
      xid[7] = static_cast<uint8_t>(i & 0xFF);
      xids.push_back(std::move(xid));
    }

    return xids;
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Failed to get next xids: ") + e.what());
  }
}

std::pair<bool, std::vector<std::string>> MongoDB::Put(
    const std::vector<std::string>& keys,
    const std::vector<std::vector<uint8_t>>& values,
    const std::vector<std::vector<uint8_t>>* xids) {
  try {
    if (keys.size() != values.size()) {
      throw std::invalid_argument("keys and values must have the same length");
    }

    if (xids && xids->size() != keys.size()) {
      throw std::invalid_argument("xids and keys must have the same length");
    }

    size_t n = keys.size();
    std::vector<std::vector<uint8_t>> auto_xids;

    // Determine which xids to use
    const std::vector<std::vector<uint8_t>>* xids_to_use = xids;
    if (!xids) {
      auto_xids = GetNextXids(n);
      xids_to_use = &auto_xids;
    }

    std::vector<std::string> inserted_ids;

    // Perform individual upserts for each key-value pair
    for (size_t i = 0; i < n; ++i) {
      // Build update document
      auto update_content = document{};
      update_content << "key" << keys[i];

      // Store value as binary data
      update_content << "value" << bsoncxx::types::b_binary{
          bsoncxx::binary_sub_type::k_uuid,
          static_cast<uint32_t>(values[i].size()),
          reinterpret_cast<const uint8_t*>(values[i].data())
      };

      // Store xid as binary data
      update_content << "xid" << bsoncxx::types::b_binary{
          bsoncxx::binary_sub_type::k_uuid,
          static_cast<uint32_t>((*xids_to_use)[i].size()),
          reinterpret_cast<const uint8_t*>((*xids_to_use)[i].data())
      };

      // Perform upsert
      auto filter = document{} << "key" << keys[i] << finalize;
      auto update = document{} << "$set" << update_content << finalize;

      mongocxx::options::update opts{};
      opts.upsert(true);

      collection_.update_one(filter.view(), update.view(), opts);
      inserted_ids.push_back(keys[i]);
    }

    return {true, inserted_ids};
  } catch (const std::exception& e) {
    return {false, {}};
  }
}

std::vector<uint8_t> MongoDB::ExtractValueFromDoc(
    const bsoncxx::document::view& doc) {
  try {
    auto value_element = doc["value"];
    if (value_element && value_element.type() == bsoncxx::type::k_binary) {
      auto binary = value_element.get_binary();
      const uint8_t* data = binary.bytes;
      uint32_t size = binary.size;
      return std::vector<uint8_t>(data, data + size);
    }
  } catch (...) {
  }
  return std::vector<uint8_t>();
}

std::vector<uint8_t> MongoDB::ExtractXidFromDoc(
    const bsoncxx::document::view& doc) {
  try {
    auto xid_element = doc["xid"];
    if (xid_element && xid_element.type() == bsoncxx::type::k_binary) {
      auto binary = xid_element.get_binary();
      const uint8_t* data = binary.bytes;
      uint32_t size = binary.size;
      return std::vector<uint8_t>(data, data + size);
    }
  } catch (...) {
  }
  return std::vector<uint8_t>();
}

std::pair<std::vector<std::vector<uint8_t>>, std::vector<std::vector<uint8_t>>>
MongoDB::GetByKeys(const std::vector<std::string>& keys) {
  std::vector<std::vector<uint8_t>> values;
  std::vector<std::vector<uint8_t>> xids_result;

  if (keys.empty()) {
    return {values, xids_result};
  }

  try {
    // Build $in query for multiple keys
    auto in_array = array{};
    for (const auto& key : keys) {
      in_array << key;
    }

    auto query = document{} << "key" << open_document << "$in" << in_array
                            << close_document << finalize;

    // Execute query and build result maps
    auto cursor = collection_.find(query.view());

    std::map<std::string, std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> 
        key_to_data;

    for (auto&& doc : cursor) {
      auto key_element = doc["key"];
      if (key_element && key_element.type() == bsoncxx::type::k_string) {
        std::string key{key_element.get_string().value};
        auto value = ExtractValueFromDoc(doc);
        auto xid = ExtractXidFromDoc(doc);
        key_to_data[key] = std::make_pair(value, xid);
      }
    }

    // Build result vectors preserving input key order
    for (const auto& key : keys) {
      auto it = key_to_data.find(key);
      if (it != key_to_data.end()) {
        values.push_back(it->second.first);
        xids_result.push_back(it->second.second);
      } else {
        values.push_back(std::vector<uint8_t>());
        xids_result.push_back(std::vector<uint8_t>());
      }
    }

    return {values, xids_result};
  } catch (const std::exception& e) {
    // On error, return empty results for all keys
    values.assign(keys.size(), std::vector<uint8_t>());
    xids_result.assign(keys.size(), std::vector<uint8_t>());
    return {values, xids_result};
  }
}

std::vector<std::vector<uint8_t>> MongoDB::GetByIds(
    const std::vector<std::vector<uint8_t>>& xids) {
  std::vector<std::vector<uint8_t>> results;

  if (xids.empty()) {
    return results;
  }

  try {
    // Build $in query for multiple xids
    auto in_array = array{};
    for (const auto& xid : xids) {
      in_array << bsoncxx::types::b_binary{
          bsoncxx::binary_sub_type::k_uuid,
          static_cast<uint32_t>(xid.size()),
          reinterpret_cast<const uint8_t*>(xid.data())
      };
    }

    auto query = document{} << "xid" << open_document << "$in" << in_array
                            << close_document << finalize;

    // Execute query and build result map
    auto cursor = collection_.find(query.view());

    std::map<std::vector<uint8_t>, std::vector<uint8_t>> xid_to_value;

    for (auto&& doc : cursor) {
      auto xid = ExtractXidFromDoc(doc);
      if (!xid.empty()) {
        auto value = ExtractValueFromDoc(doc);
        xid_to_value[xid] = value;
      }
    }

    // Build result vector preserving input xid order
    for (const auto& xid : xids) {
      auto it = xid_to_value.find(xid);
      if (it != xid_to_value.end()) {
        results.push_back(it->second);
      } else {
        results.push_back(std::vector<uint8_t>());
      }
    }

    return results;
  } catch (const std::exception& e) {
    // On error, return empty results for all xids
    results.assign(xids.size(), std::vector<uint8_t>());
    return results;
  }
}

bool MongoDB::Delete(const std::string& key) {
  try {
    auto result = collection_.delete_one(
        document{} << "key" << key << finalize);
    return result->deleted_count() > 0;
  } catch (...) {
    return false;
  }
}

}  // namespace hakes
