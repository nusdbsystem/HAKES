#ifndef HAKES_EMBEDSERVICE_H_
#define HAKES_EMBEDSERVICE_H_

#include <mutex>

#include "embed-worker/worker.h"
#include "server/worker.h"

namespace embed_worker {

class EmbedWorker : public hakes::ServiceWorker {
 public:
  EmbedWorker(std::unique_ptr<Worker>&& worker, bool is_ow_action = false)
      : initialized_(false),
        is_ow_action_(is_ow_action),
        worker_(std::move(worker)){};
  ~EmbedWorker();

  // delete copy and move
  EmbedWorker(const EmbedWorker&) = delete;
  EmbedWorker(EmbedWorker&&) = delete;
  EmbedWorker& operator=(const EmbedWorker&) = delete;
  EmbedWorker& operator=(EmbedWorker&&) = delete;

  bool Initialize() override;
  bool Handle(const std::string& url, const std::string& input,
              std::string* output) override;
  void Close() override;

 private:
  bool initialized_;
  bool is_ow_action_;
  std::unique_ptr<Worker> worker_;
};

}  // namespace embed_worker

#endif  // HAKES_EMBEDSERVICE_H_
