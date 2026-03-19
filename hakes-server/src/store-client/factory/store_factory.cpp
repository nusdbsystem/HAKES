#include "store-client/factory/store_factory.h"

#include <fstream>
#include <stdexcept>

#include "store-client/impl/mongodb/mongodb.h"

namespace hakes_store {

StoreFactory::StoreFactory(StoreConfig config) : config_(std::move(config)) {}

std::unique_ptr<Store> StoreFactory::CreateStore() {
  if (config_.store_type == "mongodb") {
    std::unique_ptr<Store> store;
    store.reset(new MongoDB(config_.store_addr));
    return std::move(store);
  }

  throw std::runtime_error("Unsupported store_type in CreateStore: " +
                           config_.store_type);
}

}  // namespace hakes_store