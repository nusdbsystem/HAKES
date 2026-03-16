#pragma once

#include <string>

#include "models/user.h"

namespace hakes_store {

class IUserRepository {
 public:
  virtual ~IUserRepository() = default;

  virtual User GetByUsername(const std::string& username) = 0;

  virtual bool CreateUser(const User& user) = 0;

  virtual bool UpdateUser(const User& user) = 0;

  virtual bool DeleteUser(const std::string& username) = 0;
};

}  // namespace hakes_store