#include <sgx_trts.h>
#include <sgx_tseal.h>

#include <cstring>
#include <memory>
#include <mutex>

#include "Enclave_t.h"
#include "embed-worker/common/config.h"
#include "embed-worker/inference-runtime/model/model.h"
#include "message/keyservice_worker.h"
#include "ratls-channel/common/channel_client.h"
#include "utils/base64.h"
#include "utils/cache.h"
#include "utils/tcrypto_ext.h"
#include "wolfssl/ssl.h"

#ifndef NDEBUG
#include <cassert>
#endif  // NDEBUG

namespace {

#if EQUAL_EXEC_TIME
bool initialized_once = false;
std::unique_ptr<ModelContext> model_context = nullptr;
#endif

// --- execution context --- //
// all global variables in enclave are thread local
//  combining with the TCSPolicy it ensures each worker handle has its
//  own execution context.
thread_local char input_key_buf[BUFSIZ];
thread_local size_t input_key_size;
thread_local char model_key_buf[BUFSIZ];
thread_local size_t model_key_size;
thread_local char prediction_buf[OUTPUT_BUF_SZ];
thread_local size_t prediction_size;

// ensure once-per-inference for ecalls
thread_local bool keys_checked = false;
thread_local bool model_checked = false;
// controls read from cache during enc_model_inference
thread_local char user_id_to_use[100];
thread_local size_t user_id_to_use_len;
thread_local char model_id_to_use[100];
thread_local size_t model_id_to_use_len;
thread_local bool keyset_ready = false;
thread_local bool key_cache_release_needed = false;
thread_local bool model_cache_release_needed = false;

thread_local char model_rt_id[100];
thread_local size_t model_rt_id_len;
thread_local embed_worker::ModelRT* model_rt;
// --- execution context --- //

struct KeySet {
  std::string user_key;
  std::string model_key;
};

inline std::string GetKeyCacheKey() {
  if ((user_id_to_use_len == 0) || (model_id_to_use_len == 0)) {
    return "";
  }
  return std::string(user_id_to_use, user_id_to_use_len) + "-" +
         std::string(model_id_to_use, model_id_to_use_len);
  ;
}

inline std::string GetModelCacheKey() {
  return std::string(model_id_to_use, model_id_to_use_len);
}

// decrypted model cache
hakes::SimpleCache<embed_worker::ModelContext> model_cache{1};
// caching the last pair of model-request key pair in use.
hakes::SimpleCache<KeySet> key_cache{1};
// shared ra channel
ratls::RatlsChannelClient* channel = nullptr;

// invoked after completing an inferencece no matter success/failure
void ClearRequestContext() {
  memset(model_key_buf, 0, BUFSIZ);
  memset(input_key_buf, 0, BUFSIZ);
  memset(prediction_buf, 0, OUTPUT_BUF_SZ);
  // key checked ensure that id to use are populated
  keys_checked = false;
  if (key_cache_release_needed) {
    assert(user_id_to_use_len > 0);
    key_cache.Release(GetKeyCacheKey());
    key_cache_release_needed = false;
  }
  // model checked
  model_checked = false;
  if (model_cache_release_needed) {
    assert(model_id_to_use_len > 0);
    model_cache.Release(GetModelCacheKey());
    model_cache_release_needed = false;
  }
  keyset_ready = false;
  user_id_to_use_len = 0;
  model_id_to_use_len = 0;
}
// invoke upon failure and final tear down.
void ClearExecutionContext() {
  ClearRequestContext();
  if (model_rt) free_model_rt(model_rt);
  model_rt = nullptr;
  model_rt_id_len = 0;
  if (channel) {
    channel->Close();
    delete channel;
    channel = nullptr;
  }
}

inline void printf(const char* msg) { ocall_print_string(msg); }

// int enc_decode_decryption_keys() {
int decode_decryption_keys(const std::string& ks_response) {
  // decode keys from reply message of key service
  std::string user_id;
  std::string input_key_b64;
  std::string model_id;
  std::string model_key_b64;
  int ret = hakes::DecodeKeyServiceWorkerReply(
      ks_response, &user_id, &input_key_b64, &model_id, &model_key_b64);

  // base64 decode the keys
  if (ret != 0) return ret;
  auto input_key_bytes = hakes::base64_decode(
      (const uint8_t*)input_key_b64.data(), input_key_b64.size());
  auto model_key_bytes = hakes::base64_decode(
      (const uint8_t*)model_key_b64.data(), model_key_b64.size());

  if ((user_id.size() != user_id_to_use_len) ||
      (model_id.size() != model_id_to_use_len) ||
      (memcmp(user_id.c_str(), user_id_to_use, user_id_to_use_len) != 0) ||
      (memcmp(model_id.c_str(), model_id_to_use, model_id_to_use_len) != 0)) {
    ocall_debug_print_string("unmatched expected id");
    return -1;
  }

  // cache into local execution context
  assert(input_key_bytes.size() < BUFSIZ);
  assert(model_key_bytes.size() < BUFSIZ);
  memset(input_key_buf, 0, BUFSIZ);
  memcpy(input_key_buf, input_key_bytes.c_str(), input_key_bytes.size());
  input_key_size = input_key_bytes.size();
  memset(model_key_buf, 0, BUFSIZ);
  memcpy(model_key_buf, model_key_bytes.c_str(), model_key_bytes.size());
  model_key_size = model_key_bytes.size();

#ifndef NDEBUG
  // ocall_debug_print_string("fetched keys from ks");
  // ocall_debug_print_hexstring(input_key_buf);
  // ocall_debug_print_hexstring(model_key_buf);
#endif  // NDEBUG

#if !EQUAL_EXEC_TIME
  // store the keys in cache
  key_cache.AddToCache(GetKeyCacheKey(), std::unique_ptr<KeySet>(new KeySet{
                                             std::move(input_key_bytes),
                                             std::move(model_key_bytes)}));
#endif  // EQUAL_EXEC_TIME
  keyset_ready = true;
  return ret;
}

sgx_status_t fetch_from_ks(const std::string& request,
                           const std::string& ks_addr, uint16_t ks_port,
                           std::string* ks_response) {
  assert(ks_response);
  if (!channel) {
    channel = new ratls::RatlsChannelClient{ks_port, std::move(ks_addr),
                                            ratls::CheckContext{}};
    if (channel->Initialize() == -1) {
      printf("failed to setup ratls ctx\n");
      return SGX_ERROR_SERVICE_UNAVAILABLE;
    }
  }

  if (channel->Connect() == -1) {
    printf("failed to setup ratls connection\n");
    return SGX_ERROR_SERVICE_UNAVAILABLE;
  }
  if (channel->Send(std::move(request)) == -1) {
    printf("failed to send request to key service\n");
    channel->Close();
    return SGX_ERROR_UNEXPECTED;
  };

  if (channel->Read(ks_response) == -1) {
    printf("failed to get keys from key service\n");
    ks_response->empty();
    channel->Close();
    return SGX_ERROR_UNEXPECTED;
  }
  // channel.Close();
  channel->CloseConnection();
  return SGX_SUCCESS;
}
}  // anonymous namespace

sgx_status_t enc_model_inference(const char* user_id, size_t user_id_size,
                                 const char* input_data, size_t input_data_size,
                                 const char* model_id, size_t model_id_size,
                                 void* store, const char* ks_addr,
                                 size_t addr_len, uint16_t ks_port,
                                 size_t* output_size) {
// only allow single permitted model under equal_exec_time setting
#if EQUAL_EXEC_TIME
  if ((sizeof(PERMITTED_MODEL) - 1 != model_id_size) ||
      (memcmp(PERMITTED_MODEL, model_id, model_id_size) != 0)) {
    ocall_debug_print(model_id, model_id_size);
    ocall_debug_print(PERMITTED_MODEL, sizeof(PERMITTED_MODEL));
    return SGX_ERROR_INVALID_PARAMETER;
  }
#endif  // EQUAL_EXEC_TIME

  // // disable checking of input size
  // if (input_data_size != hakes::get_aes_encrypted_size(INPUT_BUF_SZ)) {
  //   size_t sz = hakes::get_aes_encrypted_size(INPUT_BUF_SZ);
  //   std::string sz_str = "\n" + std::to_string(input_data_size) + " vs " +
  //                        std::to_string(sz) + "\n";
  //   ocall_print_string(sz_str.c_str());
  //   return SGX_ERROR_INVALID_PARAMETER;
  // }

  // ready the input key.
  memcpy(user_id_to_use, user_id, user_id_size);
  user_id_to_use_len = user_id_size;
  memcpy(model_id_to_use, model_id, model_id_size);
  model_id_to_use_len = model_id_size;
  int cache_ret = -1;

#if !EQUAL_EXEC_TIME
  do {
    cache_ret = key_cache.CheckAndTakeRef(GetKeyCacheKey());
  } while (cache_ret == -1);
  key_cache_release_needed = true;
  if (cache_ret == 0) {
#endif  // EQUAL_EXEC_TIME
    // fetch key from ks
    std::string ks_response;
    auto create_key_request_msg = [&]() -> std::string {
      return hakes::GetKeyRequest(std::string(user_id, user_id_size),
                                  std::string(model_id, model_id_size))
          .EncodeTo();
    };
    auto status =
        fetch_from_ks(create_key_request_msg(), std::string(ks_addr, addr_len),
                      ks_port, &ks_response);
    if (status != SGX_SUCCESS) return status;
    auto dec_ret = decode_decryption_keys(std::move(ks_response));
    if (dec_ret != 0) {
      ClearRequestContext();
      return SGX_ERROR_ECALL_NOT_ALLOWED;
    }
#if !EQUAL_EXEC_TIME
  } else {
    // get from cache
    KeySet* ret = nullptr;
    do {
      ret = key_cache.RetrieveFromCache(GetKeyCacheKey());
    } while (!ret);
    memcpy(input_key_buf, ret->user_key.data(), ret->user_key.size());
    memcpy(model_key_buf, ret->model_key.data(), ret->model_key.size());
  }
  key_cache.Release(GetKeyCacheKey());
  key_cache_release_needed = false;
#endif  // EQUAL_EXEC_TIME

  // allocate buffer to host decrypted contents
  unsigned char* decrypt_input = NULL;
  {
    // user input decryption (AES-GCM)
    auto ret = hakes::decrypt_content_with_key_aes(
        (const uint8_t*)input_data, input_data_size,
        (const uint8_t*)input_key_buf, &decrypt_input);
    if (ret != SGX_SUCCESS) {
      ClearRequestContext();
      return ret;  // note that the caller only return a buffer if success.
    }
  }

  // ready the model
  // sgx_status_t ret = SGX_SUCCESS;
  int ret = 0;
  embed_worker::ModelContext* decrypt_model = nullptr;
#if EQUAL_EXEC_TIME
  if (!initialized_once) {
    initialized_once = true;
    model_context = load_model(model_id_to_use, model_id_to_use_len,
                               model_key_buf, store, &ret);
    if (!model_context) {
      ClearRequestContext();
      return ret;
    }
  }
  decrypt_model = model_context.get();
  free_model_rt(model_rt);
  model_rt = model_rt_init(*decrypt_model, &ret);
  if (ret != SGX_SUCCESS) {
    ClearExecutionContext();
    return ret;
  }
#else
  cache_ret = -1;
  do {
    cache_ret = model_cache.CheckAndTakeRef(GetModelCacheKey());
  } while (cache_ret == -1);
  model_cache_release_needed = true;
  if (cache_ret == 0) {
    // ocall to load the model
    auto model_context = embed_worker::load_model(
        model_id_to_use, model_id_to_use_len, model_key_buf, store, &ret);
    if (!model_context) {
      ClearRequestContext();
      return (sgx_status_t)ret;
    }
    decrypt_model =
        model_cache.AddToCache(GetModelCacheKey(), std::move(model_context));
  } else {
    // untrusted part does not supply the model
    //  -- the handle was not responsible to fetch a new model
    //  -- it should be found in cache
    do {
      decrypt_model = model_cache.RetrieveFromCache(GetModelCacheKey());
    } while (!decrypt_model);
  }

  assert(decrypt_model);

  // initialize runtime if not the same of last model used
  // check if using last model, if so, can avoid model runtime init
  bool use_last_model = ((model_id_size == model_rt_id_len) &&
                         (memcmp(model_rt_id, model_id, model_id_size) == 0));

  if (!use_last_model) {
    free_model_rt(model_rt);
    model_rt = model_rt_init(*decrypt_model, &ret);
    if (ret != SGX_SUCCESS) {
      ClearExecutionContext();
      return (sgx_status_t)ret;
    }
  }
  memcpy(model_rt_id, model_id, model_id_size);
  model_rt_id_len = model_id_size;
#endif  // EQUAL_EXEC_TIME

  // All ready, run the model on input
  memset(prediction_buf, 0, sizeof(prediction_buf));
  ret = (sgx_status_t)execute_inference(
      (char*)decrypt_input, hakes::get_aes_decrypted_size(input_data_size),
      *decrypt_model, model_rt, prediction_buf, OUTPUT_BUF_SZ,
      &prediction_size);

#if !EQUAL_EXEC_TIME
  model_cache.Release(GetModelCacheKey());
  model_cache_release_needed = false;
#endif  // EQUAL_EXEC_TIME

  // set prediction size for untrusted memory allocation
  *output_size =
      (ret == SGX_SUCCESS) ? hakes::get_aes_encrypted_size(OUTPUT_BUF_SZ) : 0;

#ifndef NDEBUG
  // ocall_debug_print_hex(prediction_buf, prediction_size);
#endif  // NDEBUG

  // free resources
  free(decrypt_input);
  decrypt_input = NULL;
  return (sgx_status_t)ret;
}

sgx_status_t enc_get_encrypted_prediction(uint8_t* prediction, size_t size) {
  size_t whole_cipher_text_size = hakes::get_aes_encrypted_size(OUTPUT_BUF_SZ);

  // check if enough space in untrusted memory
  if ((sgx_is_outside_enclave(prediction, size) != 1) &&
      (size < whole_cipher_text_size))
    return SGX_ERROR_UNEXPECTED;

  // encryption
  unsigned char* output = NULL;
  sgx_status_t ret = hakes::encrypt_content_with_key_aes(
      (const uint8_t*)prediction_buf, OUTPUT_BUF_SZ,
      (const uint8_t*)input_key_buf, &output);
  if (ret == SGX_SUCCESS) memcpy(prediction, output, whole_cipher_text_size);
  free(output);

  // clear execution context
  ClearRequestContext();
  return ret;
}

void enc_clear_exec_context() { ClearExecutionContext(); }
