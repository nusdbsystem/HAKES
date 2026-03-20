#include <iostream>
#include <vector>

#include "gateway/config.h"
#include "store-client/impl/mongodb/mongodb.h"
#include "store-client/models/user.h"
#include "utils/fileutil.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: config_file\n");
    exit(1);
  }

  std::string path = argv[1];

  size_t content_len = 0;
  auto content = hakes::ReadFileToCharArray(path.c_str(), &content_len);
  if (content == nullptr) {
    fprintf(stderr, "Failed to load the data\n");
    exit(1);
  }
  hakes_worker::HakesWorkerConfig cfg = hakes_worker::ParseHakesWorkerConfig(
      std::string(content.get(), content_len));

  try {
    // Create MongoDB client and connect to the database
    hakes_store::MongoDB store(cfg.GetStoreAddr(), "hakes", "users");

    // Create root user with admin role
    std::string username = "root";
    std::string password_hash = "root";  // TODO: Use proper hashing
    std::vector<hakes_store::UserRole> roles = {hakes_store::UserRole::kAdmin};
    hakes_store::User root_user(username, password_hash, roles);

    // Insert root user into the database
    if (store.CreateUser(root_user)) {
      std::cout << "Root user created successfully." << std::endl;
    } else {
      std::cout << "Failed to create root user." << std::endl;
      return EXIT_FAILURE;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}