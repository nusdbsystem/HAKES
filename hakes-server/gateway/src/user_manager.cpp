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

#include "hakes-worker/user_manager.h"
#include "store.h"
#include <openssl/sha.h>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstring>

namespace hakes_worker {

UserManager::UserManager(Store* store) : store_(store) {}

std::string UserManager::HashPassword(const std::string& password) {
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(password.c_str()),
         password.length(), hash);
  
  std::ostringstream oss;
  for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
    oss << std::hex << std::setw(2) << std::setfill('0') 
        << static_cast<int>(hash[i]);
  }
  return oss.str();
}

UserStatus UserManager::CreateUser(const std::string& username,
                                    const std::string& password,
                                    const std::vector<UserRole>& roles) {
  if (username.empty() || password.empty()) {
    return UserStatus::kInvalidInput;
  }

  // Check if username already exists
  User user;
  bool stat = store_->GetUser(username, &user);
  if (stat) {
    return UserStatus::kUserAlreadyExists;
  }

  // Hash the password
  std::string password_hash = HashPassword(password);

  // Create the user
  user = User(username, password_hash, roles);

  // Store in memory
  if (!store_->PutUser(username, user)) {
    return UserStatus::kDatabaseError;
  }

  return UserStatus::kSuccess;
}

UserStatus UserManager::GetUser(const std::string& username, User* user) {
  if (user == nullptr) {
    return UserStatus::kInvalidInput;
  }

  if (!store_->GetUser(username, user)) {
    return UserStatus::kUserNotFound;
  }

  return UserStatus::kSuccess;
}

UserStatus UserManager::DeleteUser(const std::string& username) {
  // Check if username already exists
  User user;
  bool stat = store_->GetUser(username, &user);
  if (!stat) {
    return UserStatus::kUserNotFound;
  }
  if (!store_->DeleteUser(username)) {
    return UserStatus::kDatabaseError;
  }
  return UserStatus::kSuccess;
}

UserStatus UserManager::ListUsers(std::vector<User>& users) {
  if (store_->GetAllUsers(users)) {
    return UserStatus::kSuccess;
  } else {
    return UserStatus::kDatabaseError;
  }

  return UserStatus::kSuccess;
}

UserStatus UserManager::Authenticate(const std::string& username,
                                     const std::string& password) {
  if (username.empty() || password.empty()) {
    return UserStatus::kInvalidInput;
  }

  User user;
  UserStatus status = GetUser(username, &user);

  if (status == UserStatus::kSuccess) {
    // Verify password hash
    if (user.GetPasswordHash().compare(HashPassword(password)) != 0) {
      return UserStatus::kInvalidCredentials;
    }
    return UserStatus::kSuccess;
  } else {
    return status;
  }
}

UserStatus UserManager::ChangePassword(const std::string& username,
                                        const std::string& old_password,
                                        const std::string& new_password) {
  if (old_password.empty() || new_password.empty()) {
    return UserStatus::kInvalidInput;
  }

  User user;
  UserStatus status = GetUser(username, &user);

  if (status == UserStatus::kSuccess) {
    // Verify password hash
    if (user.GetPasswordHash().compare(HashPassword(old_password)) != 0) {
      return UserStatus::kInvalidCredentials;
    }
    // Set new password
    std::string new_hash = HashPassword(new_password);
    user.SetPasswordHash(new_hash);
    user.UpdateTimestamp();
    if (!store_->PutUser(username, user)) {
      return UserStatus::kDatabaseError;
    }
    return UserStatus::kSuccess;
  } else {
    return status;
  }
}

UserStatus UserManager::ResetPassword(const std::string& username,
                                       const std::string& new_password) {
  if (new_password.empty()) {
    return UserStatus::kInvalidInput;
  }

  auto it = users_by_username_.find(username);
  if (it == users_by_username_.end()) {
    return UserStatus::kUserNotFound;
  }

  User& user = it->second;

  // Set new password without verifying old one
  std::string new_hash = HashPassword(new_password);
  user.SetPasswordHash(new_hash);
  user.UpdateTimestamp();

  // Persist to store
  if (store_ != nullptr) {
    if (!store_->PutUser(username, user)) {
      return UserStatus::kDatabaseError;
    }
  }

  return UserStatus::kSuccess;
}

UserStatus UserManager::DeactivateUser(const std::string& username) {
  auto it = users_by_username_.find(username);
  if (it == users_by_username_.end()) {
    return UserStatus::kUserNotFound;
  }

  User& user = it->second;
  user.SetActive(false);
  user.UpdateTimestamp();

  // Persist to store
  if (store_ != nullptr) {
    if (!store_->PutUser(username, user)) {
      return UserStatus::kDatabaseError;
    }
  }

  return UserStatus::kSuccess;
}

UserStatus UserManager::ActivateUser(const std::string& username) {
  auto it = users_by_username_.find(username);
  if (it == users_by_username_.end()) {
    return UserStatus::kUserNotFound;
  }

  User& user = it->second;
  user.SetActive(true);
  user.UpdateTimestamp();

  // Persist to store
  if (store_ != nullptr) {
    if (!store_->PutUser(username, user)) {
      return UserStatus::kDatabaseError;
    }
  }

  return UserStatus::kSuccess;
}

bool UserManager::UserExists(const std::string& username) {
  auto it = users_by_username_.find(username);
  if (it == users_by_username_.end()) {
    return false;
  }
  return it->second.IsActive();
}

bool UserManager::UserHasRole(const std::string& username, UserRole role) {
  auto it = users_by_username_.find(username);
  if (it == users_by_username_.end()) {
    return false;
  }
  return it->second.HasRole(role);
}

UserStatus UserManager::AddUserRole(const std::string& username, UserRole role) {
  auto it = users_by_username_.find(username);
  if (it == users_by_username_.end()) {
    return UserStatus::kUserNotFound;
  }

  User& user = it->second;
  user.AddRole(role);
  user.UpdateTimestamp();

  // Persist to store
  if (store_ != nullptr) {
    if (!store_->PutUser(username, user)) {
      return UserStatus::kDatabaseError;
    }
  }

  return UserStatus::kSuccess;
}

UserStatus UserManager::RemoveUserRole(const std::string& username, UserRole role) {
  auto it = users_by_username_.find(username);
  if (it == users_by_username_.end()) {
    return UserStatus::kUserNotFound;
  }

  User& user = it->second;
  user.RemoveRole(role);
  user.UpdateTimestamp();

  // Persist to store
  if (store_ != nullptr) {
    if (!store_->PutUser(username, user)) {
      return UserStatus::kDatabaseError;
    }
  }

  return UserStatus::kSuccess;
}

UserStatus UserManager::SetUserRoles(const std::string& username,
                                     const std::vector<UserRole>& roles) {
  auto it = users_by_username_.find(username);
  if (it == users_by_username_.end()) {
    return UserStatus::kUserNotFound;
  }

  User& user = it->second;
  user.SetRoles(roles);
  user.UpdateTimestamp();

  // Persist to store
  if (store_ != nullptr) {
    if (!store_->PutUser(username, user)) {
      return UserStatus::kDatabaseError;
    }
  }

  return UserStatus::kSuccess;
}

}  // namespace hakes_worker
