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
### Intel(R) SGX SDK Settings ###

### Project Settings ###
SGX_Include_Paths := -I$(SGX_SDK)/include -I$(SGX_SDK)/include/tlibc \
						 -I$(SGX_SDK)/include/libcxx -I$(SGXSSL_DIR)/include

Flags_Just_For_C := -Wno-implicit-function-declaration -std=c11
Flags_Just_For_Cpp := -Wnon-virtual-dtor -std=c++11 -nostdinc++
Common_C_Cpp_Flags := $(SGX_COMMON_CFLAGS) $(SGX_COMMON_FLAGS) -nostdinc -fvisibility=hidden -fpie -fstack-protector -fno-builtin -fno-builtin-printf
Common_C_Flags := -Wjump-misses-init -Wstrict-prototypes \
										-Wunsuffixed-float-constants
SGX_RA_TLS_Extra_Flags := -DSGX_SDK -DWOLFSSL_SGX -DWOLFSSL_SGX_ATTESTATION -DUSER_TIME -DWOLFSSL_CERT_EXT


COMMON_INCLUDE_FLAGS = -I. -I${PROJECT_ROOT_DIR}/include -I$(PROJECT_ROOT_DIR)/include/key-service/trusted -I${HAKES_EMBED_ROOT}

Enclave_C_Flags := $(Flags_Just_For_C) $(Common_C_Cpp_Flags) $(Common_C_Flags) $(SGX_RA_TLS_Extra_Flags) $(SGX_Include_Paths) -I$(SGX_RA_TLS_DIR)/include $(COMMON_INCLUDE_FLAGS)

Enclave_Cpp_Flags := $(Flags_Just_For_Cpp) $(Common_C_Cpp_Flags) $(SGX_RA_TLS_Extra_Flags) -DSGXCLIENT $(SGX_Include_Paths) -I$(SGX_RA_TLS_DIR)/include $(COMMON_INCLUDE_FLAGS)

Crypto_Library_Name := sgx_tcrypto
SGXSSL_Link_Flags := -L$(SGXSSL_DIR)/lib64 -Wl,--whole-archive -lsgx_tsgxssl -Wl,--no-whole-archive -lsgx_tsgxssl_crypto -lsgx_pthread

SGX_RA_TLS_LIB=$(SGX_RA_TLS_DIR)/lib

Enclave_Link_Flags := $(SGX_COMMON_CFLAGS) \
	-Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles \
	-L$(SGX_LIBRARY_PATH) \
	-L$(SGX_RA_TLS_LIB) -lratls_ext -lratls_attester_t -lratls_challenger_t \
	-lratls_common_t -lwolfssl.sgx.static.lib \
	-Wl,--whole-archive -l$(DCAP_TVL_LIB) -l$(Trts_Library_Name) -Wl,--no-whole-archive \
	$(SGXSSL_Link_Flags) \
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
all: KeyServer_Enclave.so
	@echo "Build enclave Server_Enclave.so [$(Build_Mode)|$(SGX_ARCH)] success!"
	@echo
	@echo "*********************************************************************************************************************************************************"
	@echo "PLEASE NOTE: In this mode, please sign the Server_Enclave.so first using Two Step Sign mechanism before you run the app to launch and access the enclave."
	@echo "*********************************************************************************************************************************************************"
	@echo
else
all: KeyServer_Enclave.signed.so
endif

### Edger8r related sourcs ###
src/trusted/Enclave_t.c: $(SGX_EDGER8R) ${PROJECT_ROOT_DIR}/include/key-service/trusted/Enclave.edl
	@echo Entering ./src/trusted
	cd ./src/trusted && $(SGX_EDGER8R) --trusted ${PROJECT_ROOT_DIR}/include/key-service/trusted/Enclave.edl --search-path . --search-path $(SGX_SDK)/include --search-path $(SGXSSL_DIR)/include --search-path $(SGX_RA_TLS_DIR)/include $(Enclave_Search_Dirs)
	@echo "GEN  =>  $@"

src/trusted/Enclave_t.o: src/trusted/Enclave_t.c
	$(CC) $(Enclave_C_Flags) -c $< -o $@
	@echo "CC   <=  $<"

enclave_api: src/trusted/Enclave_t.c
### Edger8r related sourcs ###

## build files needed from other directory
src/trusted/base64.o: ${HAKES_EMBED_ROOT}/utils/base64.cpp
	$(CXX) ${Enclave_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/trusted/hexutil.o: ${HAKES_EMBED_ROOT}/utils/hexutil.cpp
	$(CXX) ${Enclave_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/trusted/json.o: ${HAKES_EMBED_ROOT}/utils/json.cpp
	$(CXX) ${Enclave_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/trusted/seal.o: ${HAKES_EMBED_ROOT}/utils/seal_t.cpp
	$(CXX) ${Enclave_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/trusted/tcrypto_ext.o: ${HAKES_EMBED_ROOT}/utils/tcrypto_ext.cpp
	$(CXX) ${Enclave_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/trusted/keyservice_user.o: ${HAKES_EMBED_ROOT}/message/keyservice_user.cpp
	$(CXX) ${Enclave_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"
src/trusted/keyservice_worker.o: ${HAKES_EMBED_ROOT}/message/keyservice_worker.cpp
	$(CXX) ${Enclave_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"
## build files needed from other directory

src/trusted/ecalls.o: src/trusted/ecalls.cpp
	$(CXX) $(Enclave_Cpp_Flags) -c $< -o $@
	@echo "CXX  <=  $<"

### Enclave Image ###
Enclave_Cpp_Objects := src/trusted/ecalls.o src/trusted/base64.o src/trusted/hexutil.o src/trusted/json.o src/trusted/seal.o src/trusted/tcrypto_ext.o src/trusted/keyservice_user.o src/trusted/keyservice_worker.o

KeyServer_Enclave.so: src/trusted/Enclave_t.o $(Enclave_Cpp_Objects)
	$(CXX) $^ -o $@ $(Enclave_Link_Flags)
	@echo "LINK =>  $@"

### Signing ###
KeyServer_Enclave.signed.so: KeyServer_Enclave.so
	$(SGX_ENCLAVE_SIGNER) sign -key src/trusted/Enclave_private.pem -enclave KeyServer_Enclave.so -out $@ -config src/trusted/Enclave.config.xml
	@echo "SIGN =>  $@"
### Sources ###

### Clean command ###
clean:
	rm -f KeyServer_Enclave.* src/trusted/Enclave_t.*  $(Enclave_Cpp_Objects)
