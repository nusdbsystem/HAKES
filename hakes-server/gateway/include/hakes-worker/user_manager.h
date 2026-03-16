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

#ifndef HAKES_HAKESWORKER_USER_MANAGER_H_
#define HAKES_HAKESWORKER_USER_MANAGER_H_

#include "hakes-worker/user_schema.h"
#include "store.h"
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace hakes_worker {

/**
 * @brief Status codes for user management operations.
 */
enum class UserStatus {
  kSuccess = 0,
  kError = 1,
  kUserNotFound = 2,
  kUserAlreadyExists = 3,
  kInvalidCredentials = 4,
  kInvalidInput = 5,
  kAuthenticationFailed = 6,
  kDatabaseError = 7
};

/**
 * @brief User management class handling CRUD operations and authentication.
 * 
 * This class provides operations for creating, reading, updating, and deleting users,
 * as well as authentication functionality. User data is persisted using the Store interface.
 */
class UserManager {
 public:
  UserManager(Store* store);
  ~UserManager() = default;

  bool Initialize();

  UserStatus CreateUser(const std::string& username,
                        const std::string& password,
                        const std::vector<UserRole>& roles = {});

  UserStatus GetUser(const std::string& username, User* user);

  UserStatus DeleteUser(const std::string& username);

  UserStatus ListUsers(std::vector<User>& users);

  UserStatus Authenticate(const std::string& username,
                          const std::string& password);

  UserStatus ChangePassword(const std::string& username,
                            const std::string& old_password,
                            const std::string& new_password);

  UserStatus ResetPassword(const std::string& username,
                           const std::string& new_password);

  bool UserExists(const std::string& username);

  UserStatus AddUserRole(const std::string& username, UserRole role);
  
  UserStatus RemoveUserRole(const std::string& username, UserRole role);

  UserStatus SetUserRoles(const std::string& username,
                          const std::vector<UserRole>& roles);

 private:
  std::string HashPassword(const std::string& password);

  Store* store_;
};

}  // namespace hakes_worker

#endif  // HAKES_HAKESWORKER_USER_MANAGER_H_
