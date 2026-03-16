#pragma once

#include <memory>
#include <mongocxx/client.hpp>
#include <mongocxx/collection.hpp>

#include "repo/user_repository.h"

namespace hakes_store {

class MongoUserRepository : public IUserRepository {
 public:
  explicit MongoUserRepository(std::shared_ptr<mongocxx::client> client);

  User GetByUsername(const std::string& username) override;

  bool CreateUser(const User& user) override;

  bool UpdateUser(const User& user) override;

  bool DeleteUser(const std::string& username) override;

 private:
  mongocxx::collection collection_;

  User BsonToUser(const bsoncxx::document::view& doc);
  bsoncxx::document::value UserToBson(const User& user);
};

}  // namespace hakes_store