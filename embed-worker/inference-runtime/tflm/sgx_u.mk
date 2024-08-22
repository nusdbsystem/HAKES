### Project Settings ###
PROJECT_ROOT_DIR := $(shell readlink -f ../..)
DEP_LOCAL_DIR := deps
### Intel(R) SGX SDK Settings ###
SGX_SDK ?= /opt/intel/sgxsdk
# SGXSSL_DIR ?= /opt/intel/sgxssl
SGX_MODE ?= HW
SGX_ARCH ?= x64
SGX_DEBUG ?= 0
SGX_PRERELEASE ?= 1
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
# $(error Cannot set SGX_DEBUG and SGX_PRERELEASE at the same time!!)
$(info Cannot set SGX_DEBUG and SGX_PRERELEASE at the same time!!")
SGX_PRERELEASE = 0
endif
endif

ifeq ($(SGX_DEBUG), 1)
        SGX_COMMON_CFLAGS += -O0 -g -DSGX_DEBUG
else
        SGX_COMMON_CFLAGS += -O2 -march=native
endif

ifneq ($(SGX_MODE), HW)
	Urts_Library_Name := sgx_urts_sim
else
	Urts_Library_Name := sgx_urts
endif

ifeq ($(SGX_MODE), HW)
ifneq ($(SGX_DEBUG), 1)
ifneq ($(SGX_PRERELEASE), 1)
Build_Mode = HW_RELEASE
endif
endif
endif

SGX_COMMON_FLAGS += -Wall -Wextra -Winit-self -Wpointer-arith -Wreturn-type \
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

APP_DCAP_LIBS := -lsgx_dcap_ql -lsgx_dcap_quoteverify

SGX_Include_Paths := -I$(SGX_SDK)/include

Flags_Just_For_C := -Wjump-misses-init -Wstrict-prototypes \
										-Wunsuffixed-float-constants
Common_C_Cpp_Flags := $(SGX_COMMON_CFLAGS) $(SGX_COMMON_FLAGS) -fPIC -Wno-attributes -DUSE_SGX

App_C_Flags := $(Flags_Just_For_C) $(Common_C_Cpp_Flags) $(SGX_Include_Paths)
App_Cpp_Flags := $(Common_C_Cpp_Flags) $(SGX_Include_Paths)
App_Link_Flags := -L$(SGX_LIBRARY_PATH)	-l$(Urts_Library_Name)
### Intel(R) SGX SDK Settings ###

### Project Settings ###
COMMON_INCLUDE_FLAGS = -I./src
App_C_Flags += $(COMMON_INCLUDE_FLAGS)
App_Cpp_Flags += $(COMMON_INCLUDE_FLAGS)
### Project Settings ###

### Project Settings ###

### Phony targets ###
.PHONY: all clean enclave_api

all: lib/libtflm_u.a

### Sources ###
## Edger8r related sources ##
SRC_EDL := $(PROJECT_ROOT_DIR)/include/embed-worker/inference-runtime/tflm/Enclave.edl
EDL_SEARCH_FLAGS := --search-path $(PROJECT_ROOT_DIR) --search-path $(PROJECT_ROOT_DIR)/include
src/untrusted/Enclave_u.h: $(SGX_EDGER8R) $(SRC_EDL)
	$(SGX_EDGER8R) --header-only --untrusted $(SRC_EDL) --untrusted-dir src/untrusted --search-path $(SGX_SDK)/include $(EDL_SEARCH_FLAGS)
	@echo "GEN => $@"

enclave_api: src/untrusted/Enclave_u.h
## Edger8r related sources ##

src/untrusted/ocalls.o: src/untrusted/ocalls.cpp src/untrusted/Enclave_u.h
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<" 

lib/libtflm_u.a: src/untrusted/ocalls.o
	ar rcs $@ $<
	@echo "LINK => $@"
### Sources ###

### Clean command ###
clean:
	@rm -f src/untrusted/*.o
	@rm -f src/untrusted/Enclave_u.h
	@rm -f lib/libtflm_u.a
