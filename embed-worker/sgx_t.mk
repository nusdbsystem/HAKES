### Project Settings ###
PROJECT_ROOT_DIR ?= $(shell readlink -f .)
HAKES_ROOT ?= $(PROJECT_ROOT_DIR)/..
SGX_RA_TLS_DIR ?= $(HAKES_ROOT)/deps/install/ratls

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
$(info "Cannot set SGX_DEBUG and SGX_PRERELEASE at the same time!!)
SGX_PRERELEASE = 0
endif
endif

ifeq ($(SGX_DEBUG), 1)
        SGX_COMMON_CFLAGS += -O0 -g -DSGX_DEBUG
else
        SGX_COMMON_CFLAGS += -O2 -march=native
endif

ifneq ($(SGX_MODE), HW)
	Trts_Library_Name := sgx_trts_sim
	Service_Library_Name := sgx_tservice_sim
else
	Trts_Library_Name := sgx_trts
	Service_Library_Name := sgx_tservice
endif

ifeq ($(SGX_MODE), HW)
ifneq ($(SGX_DEBUG), 1)
ifneq ($(SGX_PRERELEASE), 1)
Build_Mode = HW_RELEASE
endif
endif
endif

SGX_COMMON_FLAGS += -Wall -Wextra -Wchar-subscripts -Wno-coverage-mismatch \
										-Winit-self -Wpointer-arith -Wreturn-type \
                    -Waddress -Wsequence-point -Wformat-security \
                    -Wmissing-include-dirs -Wfloat-equal -Wundef -Wshadow \
                    -Wcast-align -Wcast-qual -Wconversion -Wredundant-decls

# customize suppressed warnings
SGX_COMMON_FLAGS += -Wno-sign-compare -Wno-unused-variable -Wno-comment -Wno-unused-function -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-type-limits -Wno-shadow

DCAP_TVL_LIB = sgx_dcap_tvl
Crypto_Library_Name := sgx_tcrypto

SGX_Include_Paths := -I$(SGX_SDK)/include -I$(SGX_SDK)/include/tlibc \
						 -I$(SGX_SDK)/include/libcxx -I$(SGXSSL_DIR)/include

Flags_Just_For_C := -Wno-implicit-function-declaration -std=c11 -Wjump-misses-init -Wstrict-prototypes \
										-Wunsuffixed-float-constants
Flags_Just_For_Cpp := -Wnon-virtual-dtor -std=c++11 -nostdinc++
Common_C_Cpp_Flags := $(SGX_COMMON_CFLAGS) $(SGX_COMMON_FLAGS) -nostdinc -fvisibility=hidden -fpie -fstack-protector -fno-builtin -fno-builtin-printf
SGX_RA_TLS_Extra_Flags := -DSGX_SDK -DWOLFSSL_SGX -DWOLFSSL_SGX_ATTESTATION -DUSER_TIME -DWOLFSSL_CERT_EXT

Enclave_C_Flags := $(Flags_Just_For_C) $(Common_C_Cpp_Flags) $(SGX_RA_TLS_Extra_Flags) $(SGX_Include_Paths) 
Enclave_Cpp_Flags := $(Flags_Just_For_Cpp) $(Common_C_Cpp_Flags) $(SGX_RA_TLS_Extra_Flags) -DSGXCLIENT $(SGX_Include_Paths)
### Intel(R) SGX SDK Settings ###
### Project Settings ###

COMMON_INCLUDE_FLAGS = -Isrc/trusted -I${PROJECT_ROOT_DIR}/include -I$(PROJECT_ROOT_DIR)/include/embed-worker/trusted -I${HAKES_ROOT} -I$(SGX_RA_TLS_DIR)/include
Enclave_C_Flags += $(COMMON_INCLUDE_FLAGS)
Enclave_Cpp_Flags += $(COMMON_INCLUDE_FLAGS)

SGXSSL_Link_Flags := -L$(SGXSSL_DIR)/lib64 -Wl,--whole-archive -lsgx_tsgxssl -Wl,--no-whole-archive -lsgx_tsgxssl_crypto -lsgx_pthread

SGX_RA_TLS_LIB=$(SGX_RA_TLS_DIR)/lib

Enclave_Link_Flags := $(SGX_COMMON_CFLAGS) \
	-Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles \
	-L$(SGX_LIBRARY_PATH) \
	-L$(SGX_RA_TLS_LIB) -lratls_ext -lratls_attester_t -lratls_challenger_t \
	-lratls_common_t -lwolfssl.sgx.static.lib \
	-L$(INFERENCE_RT_LINK_DIR) -l:libtrusted_inference_rt.a \
	$(SGXSSL_Link_Flags) \
	-Wl,--whole-archive -l$(DCAP_TVL_LIB) -l$(Trts_Library_Name) -Wl,--no-whole-archive \
	-Wl,--start-group -lsgx_tstdc -lsgx_tcxx -l$(Crypto_Library_Name) \
	-l$(Service_Library_Name) -Wl,--end-group \
	-Wl,-Bstatic -Wl,-Bsymbolic \
	-Wl,-pie,-eenclave_entry -Wl,--export-dynamic \
	-Wl,--defsym,__ImageBase=0 \
	-Wl,--version-script=src/trusted/Enclave.lds
### Project Settings ###

### Phony targets ###
.PHONY: all clean enclave_api

### Build all ###
ifeq ($(Build_Mode), HW_RELEASE)
all: Worker_Enclave.so
	@echo "Build enclave Server_Enclave.so [$(Build_Mode)|$(SGX_ARCH)] success!"
	@echo
	@echo "*********************************************************************************************************************************************************"
	@echo "PLEASE NOTE: In this mode, please sign the Server_Enclave.so first using Two Step Sign mechanism before you run the app to launch and access the enclave."
	@echo "*********************************************************************************************************************************************************"
	@echo
else
all: Worker_Enclave.signed.so
endif

### Edger8r related sourcs ###
src/trusted/Enclave_t.c: $(SGX_EDGER8R) ${PROJECT_ROOT_DIR}/include/embed-worker/trusted/Enclave.edl
	@echo Entering ./src/trusted
	cd ./src/trusted && $(SGX_EDGER8R) --trusted ${PROJECT_ROOT_DIR}/include/embed-worker/trusted/Enclave.edl --search-path $(PROJECT_ROOT_DIR)/include --search-path $(SGX_SDK)/include --search-path $(SGXSSL_DIR)/include --search-path $(SGX_RA_TLS_DIR)/include --search-path $(HAKES_ROOT) --search-path $(INFERENCE_RT_INCLUDE_DIR) $(Enclave_Search_Dirs)
	@echo "GEN  =>  $@"

src/trusted/Enclave_t.o: src/trusted/Enclave_t.c
	$(CC) $(Enclave_C_Flags) -c $< -o $@
	@echo "CC   <=  $<"

enclave_api: src/trusted/Enclave_t.c
### Edger8r related sourcs ###

## build files needed from other directory
src/trusted/base64.o: ${HAKES_ROOT}/utils/base64.cpp
	$(CXX) ${Enclave_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/trusted/json.o: ${HAKES_ROOT}/utils/json.cpp
	$(CXX) ${Enclave_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/trusted/tcrypto_ext.o: ${HAKES_ROOT}/utils/tcrypto_ext.cpp
	$(CXX) ${Enclave_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/trusted/channel_client.o: $(HAKES_ROOT)/ratls-channel/common/channel_client.cpp
	$(CXX) $(Enclave_Cpp_Flags) $() -c $< -o $@
	@echo "CXX  <=  $<"

src/trusted/keyservice_worker.o: ${HAKES_ROOT}/message/keyservice_worker.cpp
	$(CXX) ${Enclave_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"
## build files needed from other directory

src/trusted/ecalls.o: src/trusted/ecalls.cpp
	$(CXX) $(Enclave_Cpp_Flags) -c $< -o $@
	@echo "CXX  <=  $<"

### Enclave Image ###
Enclave_Cpp_Objects := src/trusted/base64.o src/trusted/json.o src/trusted/tcrypto_ext.o src/trusted/channel_client.o src/trusted/keyservice_worker.o src/trusted/ecalls.o

Worker_Enclave.so: src/trusted/Enclave_t.o $(Enclave_Cpp_Objects)
	$(CXX) $^ -o $@ $(Enclave_Link_Flags)
	@echo "LINK =>  $@"

### Signing ###
Worker_Enclave.signed.so: Worker_Enclave.so
	$(SGX_ENCLAVE_SIGNER) sign -key src/trusted/Enclave_private.pem -enclave Worker_Enclave.so -out $@ -config src/trusted/Enclave.config.xml
	@echo "SIGN =>  $@"
### Sources ###

### Clean command ###
clean:
	rm -f Worker_Enclave.* src/trusted/Enclave_t.*  $(Enclave_Cpp_Objects)
