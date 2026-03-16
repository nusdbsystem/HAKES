# Extending HAKES `search-worker` with a New Index

This note explains how to add a new index implementation to HAKES `search-worker`.

## 1. Mental model of `search-worker`

At a high level, the extension points are:

- `server/main.cpp`: starts the HTTP server and instantiates `search_worker::WorkerImpl`.
- `server/search_worker.cpp`: maps HTTP routes such as `/load`, `/add`, `/search`, `/rerank`, `/delete`, and `/checkpoint` to worker methods.
- `include/search-worker/worker.h`: defines the top-level `search_worker::Worker` interface.
- `include/search-worker/workerImpl.h` + `src/worker.cpp`: implement collection loading, in-memory collection registry, and dispatch to the actual index implementation.
- `index/ext/HakesCollection.h`: defines the collection/index abstraction used by `WorkerImpl`.

So if your goal is to add **another searchable index type**, the main extension point is **`faiss::HakesCollection`** and its concrete implementations under `search-worker/index/ext/`.

---

## 2. Where to put the code

### Recommended placement

Put the new code under:

- Header: `search-worker/index/ext/MyNewIndex.h`
- Implementation: `search-worker/index/ext/MyNewIndex.cpp`

If your index needs helpers, keep them near the implementation, for example:

- `search-worker/index/ext/MyNewIndexIO.h`
- `search-worker/index/ext/MyNewIndexIO.cpp`
- `search-worker/index/ext/MyNewIndexUtils.h`

### Why `index/ext/`

`index/ext/` already contains the HAKES-specific collection/index implementations and I/O helpers, for example:

- `HakesCollection.h`
- `HakesIndex.h/.cpp`
- `HakesFlatIndex.h/.cpp`
- `index_io_ext.h/.cpp`

That directory is the right place for new HAKES-specific index types rather than `src/` or `server/`.

---

## 3. Which class to extend

### Extend `faiss::HakesCollection`

Your new index class should implement the abstract interface in `search-worker/index/ext/HakesCollection.h`.

A minimal skeleton looks like this:

```cpp
#pragma once

#include "search-worker/index/ext/HakesCollection.h"

namespace faiss {

class MyNewIndex : public HakesCollection {
 public:
  MyNewIndex() = default;
  ~MyNewIndex() override = default;

  bool Initialize(const std::string& path, int mode = 0,
                  bool keep_pa = false) override;

  void UpdateIndex(const HakesCollection* other) override;

  bool AddWithIds(int n, int d, const float* vecs,
                  const faiss::idx_t* ids,
                  faiss::idx_t* assign,
                  int* vecs_t_d,
                  std::unique_ptr<float[]>* vecs_t) override;

  bool AddBase(int n, int d, const float* vecs,
               const faiss::idx_t* ids) override;

  bool AddRefine(int n, int d, const float* vecs,
                 const faiss::idx_t* ids) override;

  bool Search(int n, int d, const float* query,
              const HakesSearchParams& params,
              std::unique_ptr<float[]>* distances,
              std::unique_ptr<faiss::idx_t[]>* labels) override;

  bool Rerank(int n, int d, const float* query, int k,
              faiss::idx_t* k_base_count,
              faiss::idx_t* base_labels,
              float* base_distances,
              std::unique_ptr<float[]>* distances,
              std::unique_ptr<faiss::idx_t[]>* labels) override;

  bool Checkpoint(const std::string& checkpoint_path) const override;

  std::string GetParams() const override;
  bool UpdateParams(const std::string& params) override;
  bool DeleteWithIds(int n, const faiss::idx_t* ids) override;
  std::string to_string() const override;
};

}  // namespace faiss
```

### What each method means in HAKES

- `Initialize(...)`: load the index from a checkpoint directory.
- `AddWithIds(...)`: add vectors and ids into the collection.
- `AddBase(...)`: add into the filter/base part only.
- `AddRefine(...)`: add into the refine/full-vector part only.
- `Search(...)`: first-stage search interface used by `/search`.
- `Rerank(...)`: second-stage rerank interface used by `/rerank`.
- `Checkpoint(...)`: persist the collection state into a new `checkpoint_*` directory.
- `GetParams()` / `UpdateParams(...)`: support online parameter export/import.
- `DeleteWithIds(...)`: delete tombstone or logical delete.
- `to_string()`: used for diagnostics in load responses.

If your new index does not support some optional behavior, keep the method but either implement a no-op or explicitly return `false` with a clear message.
