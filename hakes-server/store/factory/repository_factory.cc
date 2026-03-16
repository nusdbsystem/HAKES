#include "store/factory/repository_factory.h"

#include <fstream>
#include <stdexcept>

#include "infra/mongodb/mongodb.h"

namespace hakes_store {

RepositoryFactory::RepositoryFactory(StoreConfig config)
    : config_(std::move(config)) {}

std::unique_ptr<IUserRepository> RepositoryFactory::CreateUserRepository() {
  if (config_.store_type == "mongodb") {
    auto mongodb = std::make_shared<MongoDB>(config_.store_addr);
    return std::make_unique<MongoDBUserRepository>(mongodb);
  }

  throw std::runtime_error(
      "Unsupported backend_type in CreateUserRepository: " +
      config_.backend_type);
}

}  // namespace hakes_store