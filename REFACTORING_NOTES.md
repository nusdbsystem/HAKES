# HAKES Project Refactoring

## Overview
This document describes the structural refactoring of the HAKES project from a flat organization to a hierarchical structure.

## New Directory Structure

```
Root
├── docker/              # Container definitions (unchanged)
├── deps/                # Dependencies (unchanged)
├── conf/                # Configuration files (unchanged)
├── fnpacker/            # Function packing tool (unchanged)
├── licenses/            # Third-party licenses (unchanged)
├── client/
│   └── python/          # Python client (from client/py)
└── server/              # Main server component (from hakes-worker)
    ├── auth/            # Authentication module (empty - ready for development)
    ├── searcher/        # Search worker (from search-worker)
    │   ├── include/
    │   ├── index/
    │   ├── server/
    │   ├── src/
    │   ├── test/
    │   ├── build.mk
    │   └── Makefile
    ├── store/           # Storage components
    │   ├── hakes-store/         # Store service (from hakes-store)
    │   ├── mongodb/             # MongoDB adapter (empty - ready for development)
    │   ├── fs_store.cpp         # File system store (from store-client)
    │   ├── fs_store.h
    │   └── store.h
    ├── embedder/        # Embedding components
    │   ├── worker/              # Embed worker (from embed-worker)
    │   │   ├── include/
    │   │   ├── inference-runtime/
    │   │   ├── server/
    │   │   ├── src/
    │   │   ├── build.mk
    │   │   └── Makefile
    │   └── endpoints/           # Embedding endpoints (from embed-endpoint)
    │       ├── endpoint.cpp/h
    │       ├── openai_endpoint.cpp/h
    │       ├── ollama_endpoint.cpp/h
    │       └── huggingface_endpoint.cpp/h
    ├── common/          # Shared components
    │   ├── http/                # HTTP server framework (from server/)
    │   │   ├── message/         # Message definitions (from message/)
    │   │   │   ├── client_req.cpp/h
    │   │   │   ├── embed.cpp/h
    │   │   │   ├── message.cpp/h
    │   │   │   ├── kvservice.cpp/h
    │   │   │   └── searchservice.cpp/h
    │   │   ├── server.cpp/h
    │   │   ├── service.cpp/h
    │   │   └── worker.h
    │   ├── utils/               # Utilities (from utils/)
    │   │   ├── base64.cpp/h
    │   │   ├── fileutil.cpp/h
    │   │   ├── hexutil.cpp/h
    │   │   ├── http.cpp/h
    │   │   ├── json.cpp/h
    │   │   ├── io.cpp/h
    │   │   ├── crypto_ext.cpp/h
    │   │   ├── cache.h
    │   │   ├── data_loader.h
    │   │   └── ow_message.h
    │   └── tools/               # Build tools (from tools/)
    │       ├── aes_encrypt.py
    │       ├── gen_bert_input.py
    │       └── gen_index/
    ├── include/         # Main server headers (hakes-worker/include)
    ├── src/             # Main server source (hakes-worker/src)
    ├── server/          # Server entry point (hakes-worker/server)
    ├── build.mk         # Main build rules
    └── Makefile         # Main make targets
```

## Include Path Mapping

All `#include` directives have been updated to reflect the new structure:

| Old Path | New Path |
|----------|----------|
| `message/` | `common/http/message/` |
| `utils/` | `common/utils/` |
| `embed-endpoint/` | `embedder/endpoints/` |
| `server/` | `common/http/` |
| `store-client/` | `store/` |
| `hakes-worker/` | `hakes-worker/` (unchanged) |

## Build System Updates

### Updated Makefiles
- **server/build.mk**: Updated to reference new paths for:
  - `utils/*.cpp` → `server/common/utils/*.cpp`
  - `message/*.cpp` → `server/common/http/message/*.cpp`
  - `embed-endpoint/*.cpp` → `server/embedder/endpoints/*.cpp`
  - `server/*.cpp` → `server/common/http/*.cpp`
  - Added include paths for new component locations

- **server/searcher/build.mk**: 
  - Updated HAKES_ROOT_DIR path from `..` to `../..`
  - Updated file source paths to `server/common/*` locations
  - Added include paths for message and utility components

- **server/embedder/worker/build.mk**:
  - Updated HAKES_ROOT path from `..` to `../../..`
  - Updated file source paths to `server/common/*` and `server/store/` locations
  - Added include paths for all referenced components

### Include Directories Added
Most build files now include additional search paths:
- `-I$(HAKES_ROOT)/server/common/http`
- `-I$(HAKES_ROOT)/server/common/utils`
- `-I$(HAKES_ROOT)/server/embedder/endpoints`
- `-I$(HAKES_ROOT)/server/common/http/message`

## Source File Updates

### Include Statement Changes
All C++ source and header files have been updated with sed replacements:

```bash
# Pattern replacements applied:
#include "message/..." → #include "common/http/message/..."
#include "utils/..." → #include "common/utils/..."
#include "embed-endpoint/..." → #include "embedder/endpoints/..."
#include "server/..." → #include "common/http/..."
#include "store-client/..." → #include "store/..."
```

Total files processed: ~190 C++ source and header files

## Files Retained

The following project files have been preserved:
- `.gitignore` - Git ignore patterns
- `.gitmodules` - Git submodule configuration
- `Makefile` - Root level make targets
- `README.md` - Project documentation
- `LICENSE` - License file
- `NOTICE` - Notice file
- `CITATION.cff` - Citation file

## Backward Compatibility

### Breaking Changes
- Include paths have changed for all components
- Build directory references have been updated
- Any external references to old paths will need updating

### Building the Project

To build the refactored project:

```bash
cd /users/yc/HAKES/new_root
make preparation  # Initialize git submodules
make deps         # Build dependencies
cd server
make -f build.mk all  # Build main server
cd ../server/searcher
make -f build.mk search_server  # Build search worker
cd ../embedder/worker
make -f build.mk embed_server   # Build embed worker
```

## Next Steps

1. **Compilation Testing**: Run full compilation to identify any missing includes
2. **External Dependencies**: Update any external build systems or documentation
3. **Runtime Configuration**: Verify that configuration files reference correct paths
4. **Docker Builds**: Update Docker files if they reference old paths
5. **Documentation**: Update build and development documentation

## Notes

- The `new_root/` directory is a complete, standalone copy
- Original project structure in parent directory remains unchanged
- All relative path adjustments account for the new subdirectory depths
- Component dependencies have been preserved (e.g., searcher still uses message types)
