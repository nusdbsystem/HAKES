### Project Settings ###
PROJECT_ROOT_DIR ?= $(shell readlink -f .)
HAKES_ROOT ?= $(PROJECT_ROOT_DIR)/..
DEPS_INSTALL_DIR = $(HAKES_ROOT)/deps/install
SGX_RA_TLS_DIR ?= $(DEPS_INSTALL_DIR)/ratls
LIBUV_DIR = $(DEPS_INSTALL_DIR)/libuv
LLHTTP_DIR = $(DEPS_INSTALL_DIR)/llhttp
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
Common_C_Cpp_Flags := $(SGX_COMMON_CFLAGS) $(SGX_COMMON_FLAGS) -fPIC -Wno-attributes

App_C_Flags := $(Flags_Just_For_C) $(Common_C_Cpp_Flags) $(SGX_Include_Paths)
App_Cpp_Flags := $(Common_C_Cpp_Flags) $(SGX_Include_Paths)
App_Link_Flags := -L$(SGX_LIBRARY_PATH)	-l$(Urts_Library_Name)
### Intel(R) SGX SDK Settings ###

### Project Settings ###
COMMON_INCLUDE_FLAGS = -I. -I${PROJECT_ROOT_DIR}/include -I${HAKES_ROOT} -I$(HAKES_ROOT)/ratls-channel/common -I./src/untrusted
SGX_RA_TLS_Extra_Flags := -I$(SGX_RA_TLS_DIR)/include -DWOLFSSL_SGX 
App_C_Flags += $(SGX_RA_TLS_Extra_Flags) $(COMMON_INCLUDE_FLAGS) 
App_Cpp_Flags += $(SGX_RA_TLS_Extra_Flags) $(COMMON_INCLUDE_FLAGS) 

### Project Settings ###

### Linking setting ###
SGX_RA_TLS_LINK_FLAGS := -L$(SGX_RA_TLS_DIR)/lib -lratls_attester_u -lratls_challenger -lratls_common_u \
	-l:libcurl-wolfssl.a -l:libwolfssl.a

App_Link_Flags += $(SGX_RA_TLS_LINK_FLAGS) $(APP_DCAP_LIBS) -lpthread -lz -lm -lcrypto

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
all: app embed_server
	@echo "Build app [$(Build_Mode)|$(SGX_ARCH)] success!"
	@echo
	@echo "*********************************************************************************************************************************************************"
	@echo "PLEASE NOTE: In this mode, please sign the Worker_Enclave.so first using Two Step Sign mechanism before you run the app to launch and access the enclave."
	@echo "*********************************************************************************************************************************************************"
	@echo

else
all: app embed_server
endif

### Sources ###
## Edger8r related sources ##
src/untrusted/Enclave_u.c: $(SGX_EDGER8R) ${PROJECT_ROOT_DIR}/include/embed-worker/trusted/Enclave.edl
	@echo Entering ./untrusted
	cd ./src/untrusted && $(SGX_EDGER8R) --untrusted ${PROJECT_ROOT_DIR}/include/embed-worker/trusted/Enclave.edl --search-path ${PROJECT_ROOT_DIR}/include/ --search-path $(SGX_SDK)/include --search-path $(SGXSSL_DIR)/include --search-path $(SGX_RA_TLS_DIR)/include --search-path ${HAKES_ROOT} --search-path $(INFERENCE_RT_INCLUDE_DIR) $(Enclave_Search_Dirs)
	@echo "GEN  =>  $@"

src/untrusted/Enclave_u.o: src/untrusted/Enclave_u.c
	$(CC) $(App_C_Flags) -c $< -o $@
	@echo "CC   <=  $<"

enclave_api: src/untrusted/Enclave_u.c
## Edger8r related sources ##

## build files needed from other directory
src/untrusted/base64.o: ${HAKES_ROOT}/utils/base64.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/untrusted/fileutil.o: ${HAKES_ROOT}/utils/fileutil.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/untrusted/hexutil.o: ${HAKES_ROOT}/utils/hexutil.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/untrusted/json.o: ${HAKES_ROOT}/utils/json.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/untrusted/channel_client_ocalls.o: ${HAKES_ROOT}/ratls-channel/untrusted/channel_client_ocalls.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/untrusted/embed.o: ${HAKES_ROOT}/message/embed.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/untrusted/fs_store.o: ${HAKES_ROOT}/store-client/fs_store.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

## build files needed from other directory

src/untrusted/%.o: src/untrusted/%.cpp src/untrusted/Enclave_u.o
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX  <=  $<"

App_Cpp_Objects := src/untrusted/base64.o src/untrusted/fileutil.o src/untrusted/hexutil.o src/untrusted/json.o src/untrusted/fs_store.o src/untrusted/embed.o src/untrusted/worker_u.o src/untrusted/ocalls.o src/untrusted/channel_client_ocalls.o

libuntrusted_worker.a: src/untrusted/Enclave_u.o $(App_Cpp_Objects)
	ar -rcs $@ $^
	@echo "LINK => $@"

## Build worker app ##
app: src/untrusted/app.o libuntrusted_worker.a
	$(CXX) $< -o $@ -L. -l:libuntrusted_worker.a $(App_Link_Flags) -L$(INFERENCE_RT_LINK_DIR) -l:libuntrusted_inference_rt.a
	@echo "LINK =>  $@"

## Build embed server ##

Server_Additional_Include_Flags := -Iserver -I$(LIBUV_DIR)/include -I$(LLHTTP_DIR)/include -I$(HAKES_ROOT)/server
Server_Additional_Link_Flags := -L$(LIBUV_DIR)/lib -l:libuv_a.a -L$(LLHTTP_DIR)/lib -l:libllhttp.a -lrt -ldl

server/sgx/service.o: $(HAKES_ROOT)/server/service.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

server/sgx/server.o: $(HAKES_ROOT)/server/server.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

server/sgx/embed_worker.o: server/embed_worker.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

server/sgx/main.o : server/sgx/main.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

Server_Objects := server/sgx/service.o server/sgx/server.o server/sgx/embed_worker.o server/sgx/main.o

embed_server: $(Server_Objects) libuntrusted_worker.a
	$(CXX) $(Server_Objects) -o $@ -L. -l:libuntrusted_worker.a $(App_Link_Flags) -L$(INFERENCE_RT_LINK_DIR) -l:libuntrusted_inference_rt.a $(Server_Additional_Link_Flags)
### Sources ###

### Clean command ###
clean:
	rm -f app src/untrusted/*.o $(App_Cpp_Objects) src/untrusted/Enclave_u.* libuntrusted_worker.a server/sgx/*.o embed_server
