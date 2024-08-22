/*
 * Copyright 2024 The HAKES Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <sgx_tcrypto.h>
#include <sgx_trts.h>
#include <sgx_tseal.h>

#include <cassert>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

#include "Enclave_t.h"
#include "message/keyservice_user.h"
#include "message/keyservice_worker.h"
#include "utils/base64.h"
#include "utils/json.h"
#include "utils/seal_t.h"
#include "utils/tcrypto_ext.h"
#include "wolfssl/ssl.h"

#define MAXMSGSIZE BUFSIZ
#define MAXSTOREDSIZE BUFSIZ  // max size of sealed data outside

namespace {

// for hex conversion
const char _hextable[] = "0123456789abcdef";

char user_request_buf[MAXMSGSIZE];
char worker_request_buf[MAXMSGSIZE];

// password table cache
std::map<std::string, std::string> pass_table_;
// we can have a better cache later
std::map<std::string, std::string> cache_;

// helper class to avoid copy
struct Slice {
  const char* data_;
  size_t len_;
};

struct AccessListIterator {
  void reset_to_start() {
    if (len_ == 0) valid_ = false;
    curr_ = content_;
    auto next = memchr(curr_, '|', content_ + len_ - curr_);
    curr_len_ = (next == NULL) ? len_ : ((char*)next - curr_);
    valid_ = true;
  };

  Slice entry() { return Slice{curr_, curr_len_}; }

  void next() {
    if (curr_ + curr_len_ + 1 > content_ + len_) {
      valid_ = false;
      return;
    }
    curr_ += curr_len_;
    auto next = memchr(curr_, '|', content_ + len_ - curr_);
    curr_len_ =
        (next == NULL) ? (content_ + len_ - curr_) : ((char*)next - curr_);
  }

  const char* content_;
  size_t len_;
  bool valid_;
  const char* curr_;
  size_t curr_len_;
};

// helper class to manipulate access info.
struct AccessList {
  AccessList(std::string* info) : info_(info) { assert(info); }

  AccessListIterator NewIterator() const {
    return AccessListIterator{info_->data(), info_->size(), false, NULL, 0};
  }

  void Append(const std::string& entry) {
    info_->append((info_->size() == 0) ? entry : ('|' + entry));
  }

  std::string* info_;
};

struct RequestExecCtx {
  RequestExecCtx() : sgxStatus(SGX_SUCCESS), request_failure(false), msg("") {}
  RequestExecCtx(sgx_status_t _sgxStatus, bool _request_failure)
      : sgxStatus(_sgxStatus), request_failure(_request_failure), msg("") {}
  RequestExecCtx(sgx_status_t _sgxStatus, bool _request_failure,
                 const std::string& _msg)
      : sgxStatus(_sgxStatus), request_failure(_request_failure), msg(_msg) {}
  RequestExecCtx(RequestExecCtx&& other) {
    sgxStatus = other.sgxStatus;
    request_failure = other.request_failure;
    msg = std::move(other.msg);
  };
  RequestExecCtx& operator=(RequestExecCtx&& other) {
    if (this != &other) {
      sgxStatus = other.sgxStatus;
      request_failure = other.request_failure;
      msg = std::move(other.msg);
    }
    return *this;
  }

  // copy disabled
  RequestExecCtx(const RequestExecCtx&) = delete;
  RequestExecCtx& operator=(const RequestExecCtx&) = delete;

  sgx_status_t sgxStatus;
  bool request_failure;
  std::string msg;
};

inline std::string create_worker_request_key(const std::string& model_id,
                                             const std::string& mrenclave,
                                             const std::string& user_id) {
  return model_id + "-" + mrenclave + "-" + user_id;
}

inline bool check_model_id(const std::string& model_id,
                           const std::string& owner_id) {
  ocall_debug_print_string(
      std::string("checking model id: " + model_id + " with " + owner_id)
          .c_str());
  return memcmp(model_id.c_str(), owner_id.c_str(), owner_id.size()) == 0;
}

inline std::string create_model_access_info_key(const std::string& model_name) {
  return model_name + "-AccessInfo";
}

inline std::string create_model_access_info_record(const std::string& mrenclave,
                                                   const std::string& user_id) {
  return mrenclave + "-" + user_id;
}

// true for success load, false if not found or error
bool load_and_unseal_data(const std::string& key, std::string* output,
                          void* store, sgx_status_t* status) {
  void* value_buf = malloc(MAXSTOREDSIZE);
  if (!value_buf) {
    *status = SGX_ERROR_OUT_OF_MEMORY;
    return false;
  }
  size_t retrieved_size = 0;
  ocall_get_kv(key.data(), key.size(), value_buf, MAXSTOREDSIZE,
               &retrieved_size, store);
  if (retrieved_size <= 0) {
    // no key found (-1 for store error. we keep enclave alive)
    free(value_buf);
    return false;
  }
  uint32_t unseal_buf_size =
      hakes::get_unsealed_data_size((const uint8_t*)value_buf);
  char* unseal_buf = (char*)malloc(unseal_buf_size);
  if (!unseal_buf) {
    free(value_buf);
    *status = SGX_ERROR_OUT_OF_MEMORY;
    return false;
  }
  assert(retrieved_size < UINT32_MAX);
  *status = hakes::unseal_data(
      (const uint8_t*)value_buf, static_cast<uint32_t>(retrieved_size),
      key.c_str(), key.size(), unseal_buf, unseal_buf_size);
  if (*status == SGX_SUCCESS) {
    *output = std::string(unseal_buf, unseal_buf_size);
  }
  free(value_buf);
  free(unseal_buf);
  return *status == SGX_SUCCESS;
}

sgx_status_t seal_and_save_data(const std::string& key,
                                const std::string& value, void* store) {
  // use key that is file name as aad.
  int buf_size = hakes::get_sealed_data_size(
      static_cast<uint32_t>(key.size()), static_cast<uint32_t>(value.size()));
  uint8_t* seal_buf = (uint8_t*)malloc(buf_size);
  if (!seal_buf) return SGX_ERROR_OUT_OF_MEMORY;
  auto sgxStatus = hakes::seal_data(
      key.c_str(), static_cast<uint32_t>(key.size()), value.c_str(),
      static_cast<uint32_t>(value.size()), seal_buf, buf_size);
  if (sgxStatus == SGX_SUCCESS) {
    sgxStatus =
        ocall_save_kv(key.data(), key.size(), seal_buf, buf_size, store);
  }
  free(seal_buf);
  return sgxStatus;
}

inline sgx_status_t gen_user_id(const std::string& id_key_bytes,
                                std::string* uid) {
  sgx_sha256_hash_t output;
  auto status =
      sgx_sha256_msg(reinterpret_cast<const uint8_t*>(id_key_bytes.c_str()),
                     id_key_bytes.size(), &output);
  if (status == SGX_SUCCESS) {
    // create hex of the sha256 id
    char hexid[2 * SGX_SHA256_HASH_SIZE];
    char* bp = hexid;
    for (int i = 0; i < SGX_SHA256_HASH_SIZE; i++) {
      *bp = _hextable[output[i] >> 4];
      ++bp;
      *bp = _hextable[output[i] & 0xf];
      ++bp;
    }
    uid->assign(hexid, SGX_SHA256_HASH_SIZE * 2);
#ifndef NDEBUG
    ocall_debug_print(uid->c_str(), uid->size());
#endif  // NDEBUG
  }
  return status;
}

RequestExecCtx handle_user_register(const std::string& id_key, void* store) {
  // calculate the hash
  std::string user_id;
  auto id_key_bytes = hakes::base64_decode(
      reinterpret_cast<const uint8_t*>(id_key.c_str()), id_key.size());
  if (gen_user_id(id_key_bytes, &user_id) != SGX_SUCCESS) {
    // fail to gen uid.
    return {SGX_SUCCESS, true, "sgx error: fail to gen user id"};
  }

  if (pass_table_.find(user_id) != pass_table_.end()) {
    // user id taken.
    return {SGX_SUCCESS, true, "user id already taken"};
  }
  // load password from disk if any
  sgx_status_t status = SGX_SUCCESS;
  std::string load_key;
  if (load_and_unseal_data(user_id, &load_key, store, &status)) {
    // add to the password table cache
    pass_table_[user_id] = load_key;
#ifndef NDEBUG
    ocall_debug_print_string(user_id.c_str());
#endif  // NDEBUG
    return {status, true, "user id already taken"};
  }
  if (status != SGX_SUCCESS) return {status, true};
  // cache it
  pass_table_[user_id] = id_key_bytes;
  // seal the password and use ocall to persist
  status = seal_and_save_data(user_id, id_key_bytes, store);
  return {status, status != SGX_SUCCESS};
}

RequestExecCtx decrypt_payload(const std::string& user_id,
                               const std::string& payload, void* store,
                               std::string* output) {
  // get the id key
  std::string id_key;
  sgx_status_t status = SGX_SUCCESS;
  if (pass_table_.find(user_id) != pass_table_.end()) {
    id_key = pass_table_[user_id];
  } else if (!load_and_unseal_data(user_id, &id_key, store, &status)) {
    return {status, true, "fail to load id_key"};
  }

  // decrypt the buffer
  uint8_t* plain_txt;
  auto decode_bytes = hakes::base64_decode(
      reinterpret_cast<const uint8_t*>(payload.c_str()), payload.size());
  status = hakes::decrypt_content_with_key_aes(
      reinterpret_cast<const uint8_t*>(decode_bytes.c_str()),
      decode_bytes.size(), reinterpret_cast<const uint8_t*>(id_key.c_str()),
      &plain_txt);
  if (status != SGX_SUCCESS) {
    return {status, true, "failed to decrypt payload"};
  }
  output->assign(reinterpret_cast<const char*>(plain_txt),
                 hakes::get_aes_decrypted_size(decode_bytes.size()));
  free(plain_txt);
  plain_txt = NULL;
  return {SGX_SUCCESS, false};
}

sgx_status_t handle_upsert_decrypt_key(const std::string& record_key,
                                       const std::string& decrypt_key,
                                       void* store) {
  cache_[record_key] = decrypt_key;
  // seal the decrypt key and use ocall to persist
  return seal_and_save_data(record_key, decrypt_key, store);
}

static sgx_status_t handle_append_access_list(const std::string& record_key,
                                              const std::string& new_entry,
                                              void* store) {
  if (new_entry.empty()) return SGX_SUCCESS;  // noop

#ifndef NDEBUG
  ocall_debug_print_string("updating access list");
#endif  // NDEBUG

  std::string new_list;
  sgx_status_t sgxStatus = SGX_SUCCESS;
  if (cache_.find(record_key) == cache_.end()) {
    // try to load from storage
    if (!load_and_unseal_data(record_key, &new_list, store, &sgxStatus) &&
        (sgxStatus != SGX_SUCCESS)) {
      return sgxStatus;
    }
  } else {
    new_list = cache_[record_key];
  }

#ifndef NDEBUG
  ocall_debug_print_string(std::string{"old list: " + new_list}.c_str());
#endif  // NDEBUG

  AccessList(&new_list).Append(new_entry);

#ifndef NDEBUG
  ocall_debug_print_string(std::string{"new list: " + new_list}.c_str());
#endif  // NDEBUG

  // upsert into cache
  cache_[record_key] = new_list;
  // ocall and persist
  return seal_and_save_data(record_key, new_list, store);
}

/**
 * @brief send the reply to client in enc_client_service
 *
 * @param ssl :
 * @param ctx : the service execution context from which to prepare message
 * @return int : 0 for success; -1 for failure
 */
int send_client_reply(WOLFSSL* ssl, const RequestExecCtx& ctx) {
  auto reply =
      (ctx.sgxStatus != SGX_SUCCESS)
          ? hakes::KeyServiceReply(false,
                                   "Request failed: SGX error: " + ctx.msg)
                .EncodeTo()
          : ((ctx.request_failure)
                 ? hakes::KeyServiceReply(false, "Request failed: " + ctx.msg)
                       .EncodeTo()
                 : hakes::KeyServiceReply(true, "Request success").EncodeTo());
  assert(reply.size() < INT_MAX);
  wolfSSL_write(ssl, reply.data(), static_cast<int>(reply.size()));
  return 0;
}
}  // anonymous namespace

sgx_status_t enc_client_service(WOLFSSL* ssl, void* store) {
  // sgx_status_t enc_client_service(void* ssl, void* store) {
  // if (sgx_is_within_enclave(ssl, wolfSSL_GetObjectSize()) != 1)
  //   abort();
  RequestExecCtx ctx;
  memset(user_request_buf, 0, MAXMSGSIZE);
  if (wolfSSL_read(ssl, user_request_buf, MAXMSGSIZE - 1) == 0) {
    // read failed.
    return SGX_ERROR_UNEXPECTED;
  }
  hakes::KeyServiceRequest req = hakes::DecodeKeyServiceRequest(
      std::string(user_request_buf, strlen(user_request_buf)));
  if (req.type_ == hakes::USER_REGISTER) {
    ctx = handle_user_register(req.user_id_, store);
    send_client_reply(ssl, ctx);
    return ctx.sgxStatus;
  }
  std::string plain_payload;
  ctx = decrypt_payload(req.user_id_, req.payload_, store, &plain_payload);
  if (ctx.request_failure || (ctx.sgxStatus != SGX_SUCCESS)) {
    // reject the request
    send_client_reply(ssl, ctx);
    return ctx.sgxStatus;
  }
#ifndef NDEBUG
  ocall_debug_print_string(plain_payload.c_str());
#endif  // NDEBUG
  switch (req.type_) {
    // case UPSERT_WORKER_KEY: {
    case hakes::ADD_REQUEST_KEY: {
      hakes::AddRequestKeyRequest payload =
          hakes::DecodeAddRequestKeyRequest(std::move(plain_payload));
      ctx.sgxStatus = handle_upsert_decrypt_key(
          create_worker_request_key(payload.model_id_, payload.mrenclave_,
                                    req.user_id_),
          payload.decrypt_key_, store);
      break;
    }
    case hakes::UPSERT_MODEL_KEY: {
      hakes::UpsertModelKeyRequest payload =
          hakes::DecodeUpsertModelKeyRequest(std::move(plain_payload));
      if (!check_model_id(payload.model_id_, req.user_id_)) {
        ctx.request_failure = true;
        ctx.msg = "model id not binded to the owner id";
        break;
      }
      ctx.sgxStatus = handle_upsert_decrypt_key(payload.model_id_,
                                                payload.decrypt_key_, store);
      break;
    }
    case hakes::GRANT_MODEL_ACCESS: {
      hakes::GrantModelAccessRequest payload =
          hakes::DecodeGrantModelAccessRequest(std::move(plain_payload));
      if (!check_model_id(payload.model_id_, req.user_id_)) {
        ctx.request_failure = true;
        ctx.msg = "model id not binded to the owner id";
        break;
      }
      ctx.sgxStatus = handle_append_access_list(
          create_model_access_info_key(std::move(payload.model_id_)),
          create_model_access_info_record(std::move(payload.mrenclave_),
                                          std::move(payload.user_id_)),
          store);
      break;
    }
    default:
      // unkown request type
      ctx.request_failure = true;
      ctx.msg = "unknown request type";
      break;
  }
  send_client_reply(ssl, ctx);
  return ctx.sgxStatus;
}

namespace {
// TODO extract mrenclave from the certificate
std::string extract_mr_enclave(WOLFSSL* ssl) {
  // TODO needs to augment the ratls library with function that extracts
  // mrenclave from report / simply reply report. here we use the testing
  // mrenclave value
  return "89be6ad515c32baea88e319ffe23ead9a83760960dae33119162dedbd1ac232c";
}

bool check_in_access_list(const AccessList& access_list,
                          const std::string& entry) {
#ifndef NDEBUG
  ocall_debug_print_string(
      std::string("checking access list for: " + entry).c_str());
#endif  // NDEBUG
  auto itr = access_list.NewIterator();
  itr.reset_to_start();
  while (itr.valid_) {
#ifndef NDEBUG
    ocall_debug_print(itr.entry().data_, itr.entry().len_);
    ocall_debug_print_string(std::to_string(entry.size()).c_str());
    ocall_debug_print_string(std::to_string(itr.entry().len_).c_str());
#endif  // NDEBUG
    if (entry.size() == itr.entry().len_ &&
        (memcmp(entry.data(), itr.entry().data_, entry.size()) == 0)) {
      // match found
#ifndef NDEBUG
      ocall_debug_print_string("match found!");
#endif  // NDEBUG
      return true;
    }
    itr.next();
  }
  return false;
}

inline int send_worker_failure_reply(WOLFSSL* ssl,
                                     const std::string& error_msg) {
  auto reply = hakes::KeyServiceWorkerReply(std::move(error_msg)).EncodeTo();
  assert(reply.size() < INT_MAX);
  wolfSSL_write(ssl, reply.data(), static_cast<int>(reply.size()));
  return 0;
}

bool check_permission_in_access_list(const std::string& acl_id,
                                     const std::string& candidate, void* store,
                                     sgx_status_t* sgxStatus) {
  std::string acl;
  if (cache_.find(acl_id) != cache_.end())
    acl = cache_[acl_id];
  else if (load_and_unseal_data(acl_id, &acl, store, sgxStatus)) {
    cache_[acl_id] = acl;  // cache the loaded acl.
#ifndef NDEBUG
    ocall_debug_print_string(acl.c_str());
#endif  // NDEBUG
  }
  if (acl.empty()) return false;
  return check_in_access_list(AccessList(&acl), candidate);
}

std::string retrieve_decryption_key_for_worker(const std::string& entry_id,
                                               void* store,
                                               sgx_status_t* sgxStatus) {
  std::string decrypt_key{""};
  if (cache_.find(entry_id) != cache_.end()) {
    decrypt_key = cache_[entry_id];
  } else if (load_and_unseal_data(entry_id, &decrypt_key, store, sgxStatus)) {
    cache_[entry_id] = decrypt_key;
  } else {
    decrypt_key.clear();
  }
  return decrypt_key;
}
}  // anonymous namespace

sgx_status_t enc_worker_service(WOLFSSL* ssl, void* store) {
  if (sgx_is_within_enclave(ssl, wolfSSL_GetObjectSize()) != 1) abort();
  sgx_status_t sgxStatus = SGX_SUCCESS;
  memset(worker_request_buf, 0, MAXMSGSIZE);
  if (wolfSSL_read(ssl, worker_request_buf, MAXMSGSIZE - 1) <= 0) {
    // read failed.
    return SGX_ERROR_UNEXPECTED;
  }
  hakes::GetKeyRequest req = hakes::GetKeyRequest::DecodeFrom(
      std::string(worker_request_buf, strlen(worker_request_buf)));

  std::string mrenclave = extract_mr_enclave(ssl);

  if (!req.model_id().empty()) {
    // check if user id has access right to the model
    if (!check_permission_in_access_list(
            create_model_access_info_key(req.model_id()),
            create_model_access_info_record(mrenclave, req.user_id()), store,
            &sgxStatus)) {
#ifndef NDEBUG
      ocall_debug_print_string((sgxStatus == SGX_SUCCESS)
                                   ? "worker-user not in access list"
                                   : "SGX error during check acl");
#endif  // NDEBUG
      send_worker_failure_reply(ssl, (sgxStatus == SGX_SUCCESS)
                                         ? "worker-user not in access list"
                                         : "SGX error during check acl");
      return sgxStatus;
    }
  }

  auto user_decrypt_key = retrieve_decryption_key_for_worker(
      create_worker_request_key(req.model_id(), mrenclave, req.user_id()),
      store, &sgxStatus);
  if (user_decrypt_key.empty()) {
    send_worker_failure_reply(ssl, "user not found");
    return sgxStatus;
  }

  std::string model_decrypt_key;

  if (!req.model_id().empty()) {
    model_decrypt_key =
        retrieve_decryption_key_for_worker(req.model_id(), store, &sgxStatus);
    if (model_decrypt_key.empty()) {
      send_worker_failure_reply(ssl, "worker not found");
      return sgxStatus;
    }
  }
  // send reply
  auto reply = hakes::KeyServiceWorkerReply(req.user_id(), user_decrypt_key,
                                            req.model_id(), model_decrypt_key)
                   .EncodeTo();
#ifndef NDEBUG
  ocall_debug_print_string(reply.c_str());
#endif  // NDEBUG
  assert(reply.size() < INT_MAX);
  wolfSSL_write(ssl, reply.data(), static_cast<int>(reply.size()));
  return sgxStatus;
}
