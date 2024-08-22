#include "embed_worker.h"

#include <cassert>
#include <cstring>
#include <memory>

#include "utils/json.h"
#include "utils/ow_message.h"

namespace embed_worker {

EmbedWorker::~EmbedWorker() { Close(); }

bool EmbedWorker::Initialize() {
  initialized_ = worker_->Initialize();
  return true;
}
bool EmbedWorker::Handle(const std::string& url, const std::string& input,
                         std::string* output) {
  bool error = false;
  std::string res;
  bool is_res_json = false;

  if (url == "/init") {
    if (!initialized_) {
      error = true;
      res = "Worker Init ERROR";
    } else {
      res = "Init OK";
    }
  } else if (url == "/run") {
    std::string req = is_ow_action_
                          ? ow_message::extract_ow_input(std::move(input))
                          : std::move(input);
    if (!worker_->Handle(0, req, &res)) {
      error = true;
      res = "Worker ERROR";
    } else {
      is_res_json = true;
    }
  } else {
    error = true;
    res = "Invalid URL";
  }
  output->assign((is_ow_action_
                      ? ow_message::package_ow_response(error, std::move(res), is_res_json)
                      : std::move(res)));
  return !error;
}
void EmbedWorker::Close() { worker_->Close(); }

}  // namespace embed_worker
