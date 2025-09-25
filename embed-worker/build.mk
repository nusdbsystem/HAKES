### Project Settings ###
PROJECT_ROOT_DIR ?= $(shell readlink -f .)
HAKES_ROOT ?= $(shell readlink -f ..)
DEPS_INSTALL_DIR = $(HAKES_ROOT)/deps/install
LIBUV_DIR = $(DEPS_INSTALL_DIR)/libuv
LLHTTP_DIR = $(DEPS_INSTALL_DIR)/llhttp

WARNING_IGNORE = -Wno-sign-compare -Wno-unused-variable -Wno-comment -Wno-unused-function -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-type-limits
COMMON_FLAGS = -O2 -march=native -fpic -ftree-vectorize -Wall -Wextra $(WARNING_IGNORE)
App_Cpp_Flags = $(COMMON_FLAGS)
COMMON_INCLUDE_FLAGS = -I. -I$(PROJECT_ROOT_DIR)/include -I${HAKES_ROOT} -I./src
App_Cpp_Flags += $(COMMON_INCLUDE_FLAGS)

App_Link_Flags = -lrt -pthread -lm -lcrypto -lssl
### Project Settings ###

.PHONY: all clean build_dir

all: app embed_server

## build files needed from other directory
src/base64.o: $(HAKES_ROOT)/utils/base64.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/fileutil.o: $(HAKES_ROOT)/utils/fileutil.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/hexutil.o: $(HAKES_ROOT)/utils/hexutil.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/json.o: $(HAKES_ROOT)/utils/json.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/embed.o: $(HAKES_ROOT)/message/embed.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/fs_store.o: $(HAKES_ROOT)/store-client/fs_store.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

## build files needed from other directory

src/%.o: src/%.cpp
	@echo "GEN2 $< from $@"
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

Objects := src/workerimpl.o \
	src/base64.o \
	src/fileutil.o \
	src/hexutil.o \
	src/json.o \
	src/embed.o \
	src/fs_store.o

libworker.a: $(Objects)
	ar -rcs $@ $^
	@echo "LINK => $@"

app: src/app.cpp libworker.a
	$(CXX) $(App_Cpp_Flags) $< -o $@ -L. -l:libworker.a -L$(INFERENCE_RT_LINK_DIR) -l:libinference_rt.a $(App_Link_Flags)

## Build embed server ##

Server_Additional_Include_Flags := -Iserver -I$(LIBUV_DIR)/include -I$(LLHTTP_DIR)/include -I$(HAKES_ROOT)/server
Server_Additional_Link_Flags := -L$(LIBUV_DIR)/lib -l:libuv_a.a -L$(LLHTTP_DIR)/lib -l:libllhttp.a -lrt -ldl

server/service.o: $(HAKES_ROOT)/server/service.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

server/server.o: $(HAKES_ROOT)/server/server.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

server/embed_worker.o: server/embed_worker.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

server/main.o : server/main.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

Server_Objects := server/service.o server/server.o server/embed_worker.o server/main.o

embed_server: $(Server_Objects) libworker.a
	$(CXX) $(Server_Objects) -o $@ -L. -l:libworker.a $(App_Link_Flags) -L$(INFERENCE_RT_LINK_DIR) -l:libinference_rt.a $(Server_Additional_Link_Flags)

clean:
	@rm -f app ${Objects} libworker.a server/*.o embed_server
