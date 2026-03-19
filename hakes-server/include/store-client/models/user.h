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

#ifndef HAKES_STORE_USER_H_
#define HAKES_STORE_USER_H_

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace hakes_store {

enum class UserRole {
  kAdmin = 0,       ///< Administrator role with full access
  kAppRw = 1,       ///< Application read-write role
  kAppRo = 2,       ///< Application read-only role
  kMonitoring = 3,  ///< Monitoring role for observability
  kUnknown = 9      ///< Unknown role, used for error handling
};

inline std::string ToString(UserRole role) {
  switch (role) {
    case UserRole::kAdmin:
      return "admin";
    case UserRole::kAppRw:
      return "app_rw";
    case UserRole::kAppRo:
      return "app_ro";
    case UserRole::kMonitoring:
      return "monitoring";
  }
  return "unknown";
}

inline UserRole UserRoleFromString(const std::string& role) {
  if (role == "admin")
    return UserRole::kAdmin;
  if (role == "app_rw")
    return UserRole::kAppRw;
  if (role == "app_ro")
    return UserRole::kAppRo;
  if (role == "monitoring")
    return UserRole::kMonitoring;
  else
    return UserRole::kUnknown;
}

class User {
 public:
  User() = default;

  User(std::string username, std::string password_hash,
       std::vector<UserRole> roles)
      : username_(std::move(username)),
        password_hash_(std::move(password_hash)),
        roles_(std::move(roles)) {}

  const std::string& Username() const { return username_; }

  const std::string& PasswordHash() const { return password_hash_; }

  const std::vector<UserRole>& Roles() const { return roles_; }

  bool HasRole(UserRole role) const {
    return std::find(roles_.begin(), roles_.end(), role) != roles_.end();
  }

  void AddRole(UserRole role) {
    if (!HasRole(role)) {
      roles_.push_back(role);
    }
  }

  void RemoveRole(UserRole role) {
    roles_.erase(std::remove(roles_.begin(), roles_.end(), role), roles_.end());
  }

 private:
  std::string username_;
  std::string password_hash_;
  std::vector<UserRole> roles_;
};

}  // namespace hakes_store

#endif  // HAKES_STORE_USER_H_
