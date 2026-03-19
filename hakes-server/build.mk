### Project Settings ###
PROJECT_ROOT_DIR ?= $(shell readlink -f .)
HAKES_ROOT ?= $(shell readlink -f ..)
DEPS_INSTALL_DIR = $(HAKES_ROOT)/deps/install
LIBUV_DIR = $(DEPS_INSTALL_DIR)/libuv
LLHTTP_DIR = $(DEPS_INSTALL_DIR)/llhttp
BUILD_DIR = $(abspath ./build)

# Detect MongoDB installation from environment variables
# Priority: MONGOCXX_DIR env var -> extract from CPATH -> extract from LD_LIBRARY_PATH -> /usr/local
ifndef MONGOCXX_DIR
  ifdef CPATH
    # Try to find MongoDB headers in CPATH
    MONGOCXX_DIR := $(shell for path in $(subst :, ,$(CPATH)); do if [ -d $$path/mongocxx ]; then echo $$path/..; break; fi; done)
  endif
  ifndef MONGOCXX_DIR
    ifdef LD_LIBRARY_PATH
      # Try to find MongoDB libraries in LD_LIBRARY_PATH
      MONGOCXX_DIR := $(shell for path in $(subst :, ,$(LD_LIBRARY_PATH)); do if [ -f $$path/libmongocxx.so ] || [ -f $$path/libmongocxx.a ]; then echo $$path/..; break; fi; done)
    endif
  endif
  ifndef MONGOCXX_DIR
    # Default to /usr/local
    MONGOCXX_DIR := /usr/local
  endif
endif

BSONCXX_DIR ?= $(MONGOCXX_DIR)
MONGOC_DIR ?= $(MONGOCXX_DIR)

WARNING_IGNORE = -Wno-sign-compare -Wno-unused-variable -Wno-comment -Wno-unused-function -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-type-limits
COMMON_FLAGS = -O2 -march=native -ftree-vectorize -Wall -Wextra $(WARNING_IGNORE)
App_Cpp_Flags = $(COMMON_FLAGS)
COMMON_INCLUDE_FLAGS = -I. -I$(PROJECT_ROOT_DIR)/include -I${HAKES_ROOT}/common -I${HAKES_ROOT}/hakes-server/store
App_Cpp_Flags += $(COMMON_INCLUDE_FLAGS)

App_Link_Flags = -lrt -pthread -lm -lcrypto -lssl

# MongoDB C++ driver paths
MONGOCXX_CFLAGS := -I$(MONGOCXX_DIR)/include/mongocxx/v_noabi
MONGOCXX_LIBS := -L$(MONGOCXX_DIR)/lib -lmongocxx
BSONCXX_CFLAGS := -I$(BSONCXX_DIR)/include/bsoncxx/v_noabi
BSONCXX_LIBS := -L$(BSONCXX_DIR)/lib -lbsoncxx
MONGOC_CFLAGS := -I$(MONGOC_DIR)/include/mongoc-2.1.2 -I$(MONGOC_DIR)/include/mongoc-2.1.2/mongoc
MONGOC_LIBS := -L$(MONGOC_DIR)/lib -lmongoc2
BSON_CFLAGS := -I$(BSONCXX_DIR)/include/bson-2.1.2 -I$(BSONCXX_DIR)/include/bson-2.1.2/bson
BSON_LIBS := -L$(BSONCXX_DIR)/lib -lbson2

Store_Cpp_Flags += $(COMMON_INCLUDE_FLAGS) $(MONGOCXX_CFLAGS) $(BSONCXX_CFLAGS) $(MONGOC_CFLAGS) $(BSON_CFLAGS)
Store_Link_Flags = $(MONGOCXX_LIBS) $(BSONCXX_LIBS) $(MONGOC_LIBS) $(BSON_LIBS) -lpthread


### Project Settings ###

.PHONY: all clean

BUILD_SENTINEL = $(BUILD_DIR)/.build_dirs_created

all: $(BUILD_SENTINEL) libhakes_server.a hakes_server

$(BUILD_SENTINEL):
	@mkdir -p ${BUILD_DIR}
	@mkdir -p ${BUILD_DIR}/common
	@mkdir -p ${BUILD_DIR}/common/utils
	@mkdir -p ${BUILD_DIR}/common/server
	@mkdir -p ${BUILD_DIR}/common/server/message
	@mkdir -p ${BUILD_DIR}/server
	@mkdir -p ${BUILD_DIR}/gateway
	@mkdir -p ${BUILD_DIR}/embed-endpoint
	@mkdir -p ${BUILD_DIR}/store-client
	@touch $@

build_dir: $(BUILD_SENTINEL)

UTILS_SRC := $(shell find ${HAKES_ROOT}/common/utils -name '*.cpp')
MESSAGE_SRC := $(shell find ${HAKES_ROOT}/common/server/message -name '*.cpp')
COMMON_SRC := ${UTILS_SRC} ${MESSAGE_SRC}
COMMON_OBJS := $(patsubst ${HAKES_ROOT}/%.cpp,$(BUILD_DIR)/%.o,$(COMMON_SRC))

GATEWAY_SRC := $(shell find src/gateway -name '*.cpp')
GATEWAY_OBJS := $(patsubst src/gateway/%.cpp,$(BUILD_DIR)/gateway/%.o,$(GATEWAY_SRC))

EMBED_SRC := $(shell find src/embed-endpoint -name '*.cpp')
EMBED_OBJS := $(patsubst src/embed-endpoint/%.cpp,$(BUILD_DIR)/embed-endpoint/%.o,$(EMBED_SRC))

STORE_CLIENT_SRC := $(shell find src/store-client -name '*.cpp')
STORE_CLIENT_OBJS := $(patsubst src/store-client/%.cpp,$(BUILD_DIR)/store-client/%.o,$(STORE_CLIENT_SRC))

OBJS := ${COMMON_OBJS} ${GATEWAY_OBJS} ${EMBED_OBJS} ${STORE_CLIENT_OBJS}

# Compile common source files from HAKES_ROOT
$(BUILD_DIR)/common/utils/%.o: ${HAKES_ROOT}/common/utils/%.cpp
	@mkdir -p $(dir $@)
	@echo "Compile $< -> $@"
	$(CXX) $(App_Cpp_Flags) -c $< -o $@

$(BUILD_DIR)/common/server/message/%.o: ${HAKES_ROOT}/common/server/message/%.cpp
	@mkdir -p $(dir $@)
	@echo "Compile $< -> $@"
	$(CXX) $(App_Cpp_Flags) -c $< -o $@

# Compile gateway source files
$(BUILD_DIR)/gateway/%.o: src/gateway/%.cpp
	@mkdir -p $(dir $@)
	@echo "Compile $< -> $@"
	$(CXX) $(App_Cpp_Flags) -c $< -o $@

# Compile embed-endpoint source files
$(BUILD_DIR)/embed-endpoint/%.o: src/embed-endpoint/%.cpp
	@mkdir -p $(dir $@)
	@echo "Compile $< -> $@"
	$(CXX) $(App_Cpp_Flags) -c $< -o $@

# Compile store-client source files with special flags
$(BUILD_DIR)/store-client/%.o: src/store-client/%.cpp
	@mkdir -p $(dir $@)
	@echo "Compile $< -> $@"
	$(CXX) $(Store_Cpp_Flags) -c $< -o $@

libhakes_server.a: $(OBJS)
	ar -rcs $@ $^
	@echo "LINK => $@"

# ## Build server ##
Server_Additional_Include_Flags := -Iserver -I$(LIBUV_DIR)/include -I$(LLHTTP_DIR)/include -I$(HAKES_ROOT)/common/server
Server_Additional_Link_Flags := -L$(LIBUV_DIR)/lib -l:libuv_a.a -L$(LLHTTP_DIR)/lib -l:libllhttp.a -lrt -ldl -lcurl

${BUILD_DIR}/common/server/service.o: $(HAKES_ROOT)/common/server/service.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

${BUILD_DIR}/common/server/server.o: $(HAKES_ROOT)/common/server/server.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

${BUILD_DIR}/server/hakes_worker.o: src/server/hakes_worker.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

${BUILD_DIR}/server/main.o : src/server/main.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

Server_Objects := ${BUILD_DIR}/common/server/service.o ${BUILD_DIR}/common/server/server.o ${BUILD_DIR}/server/hakes_worker.o ${BUILD_DIR}/server/main.o

hakes_server: $(Server_Objects) libhakes_server.a
	$(CXX) $(Server_Objects) -o $@ -L. -l:libhakes_server.a $(App_Link_Flags) $(Store_Link_Flags) $(Server_Additional_Link_Flags)

clean:
	@rm -rf build libhakes_server.a hakes_server
