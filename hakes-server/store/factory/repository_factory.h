#pragma once

#include <memory>
#include <string>

#include "repo/user_repository.h"

namespace hakes_store {

struct StoreConfig {
  std::string store_type;
  std::string store_addr;
};

class RepositoryFactory {
 public:
  explicit RepositoryFactory(StoreConfig config);

  std::unique_ptr<IUserRepository> CreateUserRepository();

 private:
  StoreConfig config_;
};

}  // namespace hakes_store