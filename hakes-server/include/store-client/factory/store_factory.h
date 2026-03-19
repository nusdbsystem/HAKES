#pragma once

#include <memory>
#include <string>

#include "store-client/interface/store.h"

namespace hakes_store {

struct StoreConfig {
  std::string store_type;
  std::string store_addr;
};

class StoreFactory {
 public:
  explicit StoreFactory(StoreConfig config);

  std::unique_ptr<Store> CreateStore();

 private:
  StoreConfig config_;
};

}  // namespace hakes_store