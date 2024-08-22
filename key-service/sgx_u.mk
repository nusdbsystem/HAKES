### Project Settings ###
PROJECT_ROOT_DIR ?= $(shell readlink -f .)
HAKES_EMBED_ROOT ?= $(PROJECT_ROOT_DIR)/..
SGX_RA_TLS_DIR ?= $(HAKES_EMBED_ROOT)/deps/install/ratls
### Intel(R) SGX SDK Settings ###
SGX_SDK ?= /opt/intel/sgxsdk
SGXSSL_DIR ?= /opt/intel/sgxssl
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

APP_DCAP_LIBS := -lsgx_dcap_ql -lsgx_dcap_quoteverify
### Intel(R) SGX SDK Settings ###

### Project Settings ###
Common_C_Cpp_Flags := $(SGX_COMMON_CFLAGS) -fPIC -Wno-attributes -I.
Common_C_Cpp_Flags += -Wall -Wextra -Winit-self -Wpointer-arith -Wreturn-type \
                    -Waddress -Wsequence-point -Wformat-security \
                    -Wmissing-include-dirs -Wfloat-equal -Wundef -Wshadow \
                    -Wcast-align -Wcast-qual -Wconversion -Wredundant-decls
Common_C_Flags := -Wjump-misses-init -Wstrict-prototypes \
										-Wunsuffixed-float-constants
SGX_RA_TLS_Extra_Flags := -DWOLFSSL_SGX 


# Three configuration modes - Debug, prerelease, release
#   Debug - Macro DEBUG enabled.
#   Prerelease - Macro NDEBUG and EDEBUG enabled.
#   Release - Macro NDEBUG enabled.
ifeq ($(SGX_DEBUG), 1)
        Common_C_Cpp_Flags += -DDEBUG -UNDEBUG -UEDEBUG
else ifeq ($(SGX_PRERELEASE), 1)
        Common_C_Cpp_Flags += -DNDEBUG -DEDEBUG -UDEBUG
else
        Common_C_Cpp_Flags += -DNDEBUG -UEDEBUG -UDEBUG
endif


COMMON_INCLUDE_FLAGS = -I. -I${PROJECT_ROOT_DIR}/include -I${HAKES_EMBED_ROOT} -I$(HAKES_EMBED_ROOT)/ratls-channel/common -I./src/untrusted

App_C_Cpp_Flags := $(Common_C_Cpp_Flags) $(SGX_RA_TLS_Extra_Flags) -I$(SGX_SDK)/include $(COMMON_INCLUDE_FLAGS) -I$(SGX_RA_TLS_DIR)/include

### Project Settings ###

### Linking setting ###
SGX_RA_TLS_LINK_FLAGS := -L$(SGX_RA_TLS_DIR)/lib -lratls_attester_u -lratls_challenger -lratls_common_u \
	-l:libcurl-wolfssl.a -l:libwolfssl.a

App_Link_Flags := $(SGX_RA_TLS_LINK_FLAGS) -L$(SGX_LIBRARY_PATH)	-l$(Urts_Library_Name) $(APP_DCAP_LIBS) \
	-lpthread -lz -lm -lcrypto

## Add sgx_uae_service library to link ##
ifneq ($(SGX_MODE), HW)
	App_Link_Flags += -lsgx_uae_service_sim
else
	App_Link_Flags += -lsgx_uae_service
endif

# ## Add sgx ssl library
App_Link_Flags += -L$(SGXSSL_DIR)/lib64 -lsgx_usgxssl
### Linking setting ###

### Phony targets ###
.PHONY: all clean enclave_api

### Build all ###
ifeq ($(Build_Mode), HW_RELEASE)
all: key_server
	@echo "Build app [$(Build_Mode)|$(SGX_ARCH)] success!"
	@echo
	@echo "*********************************************************************************************************************************************************"
	@echo "PLEASE NOTE: In this mode, please sign the Worker_Enclave.so first using Two Step Sign mechanism before you run the app to launch and access the enclave."
	@echo "*********************************************************************************************************************************************************"
	@echo

else
all: key_server
endif

### Sources ###
## Edger8r related sources ##
src/untrusted/Enclave_u.c: $(SGX_EDGER8R) ${PROJECT_ROOT_DIR}/include/key-service/trusted/Enclave.edl
	@echo Entering ./untrusted
	cd ./src/untrusted && $(SGX_EDGER8R) --untrusted ${PROJECT_ROOT_DIR}/include/key-service/trusted/Enclave.edl --search-path ${PROJECT_ROOT_DIR}/include/ --search-path ${PROJECT_ROOT_DIR}/include/untrusted --search-path $(SGX_SDK)/include --search-path $(SGXSSL_DIR)/include --search-path $(SGX_RA_TLS_DIR)/include $(Enclave_Search_Dirs)
	@echo "GEN  =>  $@"

src/untrusted/Enclave_u.o: src/untrusted/Enclave_u.c
	$(CC) $(Common_C_Flags) $(App_C_Cpp_Flags) -c $< -o $@
	@echo "CC   <=  $<"

enclave_api: src/untrusted/Enclave_u.c
## Edger8r related sources ##

## build files needed from other directory
src/untrusted/base64.o: ${HAKES_EMBED_ROOT}/utils/base64.cpp
	$(CXX) $(App_C_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/untrusted/fileutil.o: ${HAKES_EMBED_ROOT}/utils/fileutil.cpp
	$(CXX) $(App_C_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/untrusted/channel_server.o: ${HAKES_EMBED_ROOT}/ratls-channel/untrusted/channel_server.cpp
	$(CXX) $(App_C_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/untrusted/fs_store.o: ${HAKES_EMBED_ROOT}/store-client/fs_store.cpp
	$(CXX) $(App_C_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/untrusted/fs_store.o: ${HAKES_EMBED_ROOT}/store-client/fs_store.cpp
	$(CXX) $(App_C_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

## build files needed from other directory

src/untrusted/%.o: src/untrusted/%.cpp src/untrusted/Enclave_u.o
	$(CXX) $(App_C_Cpp_Flags) -c $< -o $@
	@echo "CXX  <=  $<"

App_Cpp_Objects := src/untrusted/base64.o src/untrusted/fileutil.o src/untrusted/channel_server.o src/untrusted/fs_store.o src/untrusted/ocalls.o

libuntrusted_worker.a: src/untrusted/Enclave_u.o $(App_Cpp_Objects)
	ar -rcs $@ $^
	@echo "LINK => $@"

## Build worker app ##
key_server: src/untrusted/keyservice_server.o libuntrusted_worker.a
	$(CXX) $< -o $@ -L. -l:libuntrusted_worker.a $(App_Link_Flags)
	@echo "LINK =>  $@"
### Sources ###

### Clean command ###
clean:
	rm -f key_server src/untrusted/keyservice_server.o $(App_Cpp_Objects) src/untrusted/Enclave_u.* libuntrusted_worker.a
