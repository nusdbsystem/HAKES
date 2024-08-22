### Project Settings ###
PROJECT_ROOT_DIR ?= $(shell readlink -f ../..)
HAKES_EMBED_ROOT ?= $(shell readlink -f ../../..)
CRT_SRCS_LOCAL = $(abspath deps/standalone_crt)

App_C_Flags := -Wall -Wextra -O2 -march=native -fPIC
App_Cpp_Flags := -Wall -Wextra -O2 -march=native -fPIC
COMMON_INCLUDE_FLAGS :=	-I./src -I$(CRT_SRCS_LOCAL)/include
App_C_Flags += $(COMMON_INCLUDE_FLAGS)
App_Cpp_Flags += $(COMMON_INCLUDE_FLAGS)
### Project Settings ###

.PHONY: all install clean mrproper

all: lib/libtvm.a

## build standalone crt libs 
no_sgx/build:
	mkdir -p no_sgx/build

CRT_DEPS_BUILD_LOCAL = $(abspath src/no_sgx/build/deps)

$(CRT_DEPS_BUILD_LOCAL):
	mkdir -p $(CRT_DEPS_BUILD_LOCAL)/lib
	mkdir -p $(CRT_DEPS_BUILD_LOCAL)/objs

$(CRT_DEPS_BUILD_LOCAL)/lib/libcommon.a: $(CRT_DEPS_BUILD_LOCAL)
	cd $(CRT_SRCS_LOCAL) && make QUIET= BUILD_DIR=$(CRT_DEPS_BUILD_LOCAL)/lib CRT_CONFIG=$(abspath src/common/crt_config.h) "EXTRA_CFLAGS=$(App_C_Flags)" common

$(CRT_DEPS_BUILD_LOCAL)/lib/libgraph_executor.a: $(CRT_DEPS_BUILD_LOCAL)
	cd $(CRT_SRCS_LOCAL) && make QUIET= BUILD_DIR=$(CRT_DEPS_BUILD_LOCAL)/lib CRT_CONFIG=$(abspath src/common/crt_config.h) "EXTRA_CFLAGS=$(App_C_Flags)" graph_executor
	
$(CRT_DEPS_BUILD_LOCAL)/lib/libmemory.a: $(CRT_DEPS_BUILD_LOCAL)
	cd $(CRT_SRCS_LOCAL) && make QUIET= BUILD_DIR=$(CRT_DEPS_BUILD_LOCAL)/lib CRT_CONFIG=$(abspath src/common/crt_config.h) "EXTRA_CFLAGS=$(App_C_Flags)" memory

$(CRT_DEPS_BUILD_LOCAL)/objs: $(CRT_DEPS_BUILD_LOCAL)/lib/libmemory.a $(CRT_DEPS_BUILD_LOCAL)/lib/libgraph_executor.a $(CRT_DEPS_BUILD_LOCAL)/lib/libcommon.a
	find $(CRT_DEPS_BUILD_LOCAL)/lib -type f -name "*.o" -exec cp -prv '{}' $(CRT_DEPS_BUILD_LOCAL)/objs ';'
## build standalone crt libs 

src/no_sgx/bundle_static.o: src/common/bundle_static.c
	$(CC) $(App_C_Flags) -Isrc/common -c $< -o $@
	@echo "CC   <=  $<"

src/no_sgx/tvm_patch.o: src/no_sgx/patch.cpp
	$(CXX) $(App_Cpp_Flags) -Isrc/common -c $< -o $@
	@echo "CXX   <=  $<"

Objects := src/no_sgx/bundle_static.o  src/no_sgx/tvm_patch.o

lib/libtvm.a: $(CRT_DEPS_BUILD_LOCAL)/objs src/no_sgx/bundle_static.o src/no_sgx/tvm_patch.o
	ar rcs $@ $(Objects) $(wildcard src/no_sgx/build/deps/objs/*.o)
	@echo "LINK => $@"

clean:
	@rm -f src/no_sgx/*.o
	@rm -rf src/no_sgx/build
