# For tvm crt, we use the cmake files provided by tvm to install the sources files of standalone crt to the local deps directory. And build the dependencies libraries with enclave flags.
PROJECT_ROOT_DIR := $(shell readlink -f ../..)
HAKES_EMBED_ROOT := $(shell readlink -f ../../..)
DEP_LOCAL_DIR := deps

### Intel(R) SGX SDK Settings ###
SGX_SDK ?= /opt/intel/sgxsdk
SGX_MODE ?= HW
SGX_ARCH ?= x64
SGX_DEBUG ?= 0
ifeq ($(shell getconf LONG_BIT), 32)
	SGX_ARCH := x86
else ifeq ($(findstring -m32, $(CXXFLAGS)), -m32)
	SGX_ARCH := x86
endif

ifeq ($(SGX_ARCH), x86)
	SGX_COMMON_CFLAGS := -m32
	SGX_LIBRARY_PATH := $(SGX_SDK)/lib
	SGX_ENCLAVE_SIGNER := $(SGX_SDK)/bin/x86/sgx_sign
	SGX_EDGER8R := $(SGX_SDK)/bin/x86/sgx_edger8r
else
	SGX_COMMON_CFLAGS := -m64
	SGX_LIBRARY_PATH := $(SGX_SDK)/lib64
	SGX_ENCLAVE_SIGNER := $(SGX_SDK)/bin/x64/sgx_sign
	SGX_EDGER8R := $(SGX_SDK)/bin/x64/sgx_edger8r
endif

ifeq ($(SGX_DEBUG), 1)
ifeq ($(SGX_PRERELEASE), 1)
$(error Cannot set SGX_DEBUG and SGX_PRERELEASE at the same time!!)
endif
endif

ifeq ($(SGX_DEBUG), 1)
        SGX_COMMON_CFLAGS += -O0 -g -DSGX_DEBUG
else
        SGX_COMMON_CFLAGS += -O2 -march=native
endif

SGX_COMMON_CFLAGS += -Wall -Wextra -Wchar-subscripts -Wno-coverage-mismatch \
										-Winit-self -Wpointer-arith -Wreturn-type \
                    -Waddress -Wsequence-point -Wformat-security \
                    -Wmissing-include-dirs -Wfloat-equal -Wundef -Wshadow \
                    -Wcast-align -Wcast-qual -Wconversion -Wredundant-decls

# Three configuration modes - Debug, prerelease, release
#   Debug - Macro DEBUG enabled.
#   Prerelease - Macro NDEBUG and EDEBUG enabled.
#   Release - Macro NDEBUG enabled.
ifeq ($(SGX_DEBUG), 1)
        SGX_COMMON_CFLAGS += -DDEBUG -UNDEBUG -UEDEBUG
else ifeq ($(SGX_PRERELEASE), 1)
        SGX_COMMON_CFLAGS += -DNDEBUG -DEDEBUG -UDEBUG
else
        SGX_COMMON_CFLAGS += -DNDEBUG -UEDEBUG -UDEBUG
endif

SGX_Include_Paths := -I$(SGX_SDK)/include -I$(SGX_SDK)/include/tlibc \
						 -I$(SGX_SDK)/include/libcxx
### Intel(R) SGX SDK Settings ###

Flags_Just_For_C := -Wno-implicit-function-declaration -std=c11 \
	-Wjump-misses-init -Wstrict-prototypes -Wunsuffixed-float-constants
Flags_Just_For_Cpp := -Wnon-virtual-dtor -std=c++11 -nostdinc++
Common_C_Cpp_Flags := $(SGX_COMMON_CFLAGS) $(SGX_COMMON_FLAGS) -nostdinc -fvisibility=hidden -fpie -fstack-protector -fno-builtin -fno-builtin-printf -DUSE_SGX

Enclave_C_Flags := $(Flags_Just_For_C) $(Common_C_Cpp_Flags) $(SGX_Include_Paths)

Enclave_Cpp_Flags := $(Flags_Just_For_Cpp) $(Common_C_Cpp_Flags) $(SGX_Include_Paths)

### Project Settings ###
COMMON_INCLUDE_FLAGS := -I./src
Enclave_C_Flags += $(COMMON_INCLUDE_FLAGS)
Enclave_Cpp_Flags += $(COMMON_INCLUDE_FLAGS)
### Project Settings ###

### Phony targets ###
.PHONY: checks all clean enclave_api
### Phony targets ###

### some checking before building any targets
checks:
ifndef PROJECT_ROOT_DIR
	$(error PROJECT_ROOT_DIR is not set. Please set to secure serverless inference project root directory)
endif

all: checks lib/libtflm_t.a

### Edger8r related sourcs ###
SRC_EDL := $(PROJECT_ROOT_DIR)/include/embed-worker/inference-runtime/tflm/Enclave.edl
EDL_SEARCH_FLAGS := --search-path $(PROJECT_ROOT_DIR) --search-path $(PROJECT_ROOT_DIR)/include
src/trusted/Enclave_t.h: $(SGX_EDGER8R) $(SRC_EDL)
	$(SGX_EDGER8R) --header-only --trusted $(SRC_EDL) --trusted-dir src/trusted --search-path $(SGX_SDK)/include $(EDL_SEARCH_FLAGS)
	@echo "GEN => $@"

enclave_api: src/trusted/Enclave_t.h
### Edger8r related sourcs ###

src/trusted/tflm_patch.o: src/trusted/patch.cpp src/trusted/Enclave_t.h
	$(CXX) $(Enclave_Cpp_Flags) -Isrc/common -c $< -o $@
	@echo "CXX   <=  $<"

Enclave_Objects := src/trusted/tflm_patch.o

lib/libtflm_t.a: $(Enclave_Objects)
	cp $(DEP_LOCAL_DIR)/lib/libtflm.a $@
	ar r $@ $^
	@echo "LINK => $@"

### Clean command ###
clean:
	@rm -f src/trusted/*.o src/trusted/Enclave_t.h
	@rm -f lib/libtflm_t.a

