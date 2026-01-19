### Project Settings ###
PROJECT_ROOT_DIR ?= $(shell readlink -f ../../..)
HAKES_EMBED_ROOT ?= $(shell readlink -f ../../../..)
DEP_TVMCRT_DIR := $(shell readlink -f ../../tvm_crt)
MODULE_OBJS_DIR ?= $(DEP_TVMCRT_DIR)/module
CRT_SRCS_LOCAL = $(abspath ../../tvm_crt/deps/standalone_crt)

App_C_Flags := -g -Wall -Wextra -O2 -fPIC
App_Cpp_Flags := -g -Wall -Wextra -O2 -fPIC

MODEL_OBJ = $(MODULE_OBJS_DIR)/model_c/devc.o $(MODULE_OBJS_DIR)/model_c/lib0.o $(MODULE_OBJS_DIR)/model_c/lib1.o
COMMON_INCLUDE_FLAGS :=	-I./src -I$(HAKES_EMBED_ROOT) -I$(PROJECT_ROOT_DIR)/include -I$(CRT_SRCS_LOCAL)/include
App_C_Flags += $(COMMON_INCLUDE_FLAGS)
App_Cpp_Flags += $(COMMON_INCLUDE_FLAGS)
### Project Settings ###

.PHONY: all install clean

all: libinference_rt.a

src/tvm_default.o: src/tvm_default.cpp
	$(CXX) $(App_Cpp_Flags) -Isrc -c $< -o $@
	@echo "CXX   <=  $<"

src/patch.o: src/patch.cpp
	$(CXX) $(App_Cpp_Flags) -Isrc -c $< -o $@
	@echo "CXX   <=  $<"

Objects := src/tvm_default.o  src/patch.o $(MODEL_OBJ)

libinference_rt.a: $(Objects)
	@echo "Creating inference runtime"
	cp $(DEP_TVMCRT_DIR)/lib/libtvm.a $@
	ar r $@ $^
	@echo "Created inference runtime"

clean:
	@rm -f src/*.o libinference_rt.a
