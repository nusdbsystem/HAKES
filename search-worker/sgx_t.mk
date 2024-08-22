### Project Settings ###
PROJECT_ROOT_DIR ?= $(shell readlink -f .)
HAKES_ROOT_DIR ?= $(shell readlink -f ..)
SGX_RA_TLS_DIR ?= $(HAKES_ROOT_DIR)/deps/install/ratls

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
# $(error Cannot set SGX_DEBUG and SGX_PRERELEASE at the same time!!)
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

# customize suppressed warnings
SGX_COMMON_FLAGS += -Wno-sign-compare -Wno-unused-variable -Wno-comment -Wno-unused-function -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-type-limits -Wno-shadow

DCAP_TVL_LIB = sgx_dcap_tvl
Crypto_Library_Name := sgx_tcrypto

SGX_Include_Paths := -I$(SGX_SDK)/include -I$(SGX_SDK)/include/tlibc \
						 -I$(SGX_SDK)/include/libcxx -I$(SGXSSL_DIR)/include

Flags_Just_For_C := -Wno-implicit-function-declaration -std=c11 -Wjump-misses-init \
	-Wstrict-prototypes -Wunsuffixed-float-constants
Flags_Just_For_Cpp := -Wnon-virtual-dtor -std=c++11 -nostdinc++
Common_C_Cpp_Flags := $(SGX_COMMON_CFLAGS) $(SGX_COMMON_FLAGS) -nostdinc -fvisibility=hidden -fpie -fstack-protector -fno-builtin -fno-builtin-printf -fopenmp -DUSE_SGX
Common_C_Flags := -Wjump-misses-init -Wstrict-prototypes \
										-Wunsuffixed-float-constants
SGX_RA_TLS_Extra_Flags := -DSGX_SDK -DWOLFSSL_SGX -DWOLFSSL_SGX_ATTESTATION -DUSER_TIME -DWOLFSSL_CERT_EXT

Enclave_C_Flags := $(Flags_Just_For_C) $(Common_C_Cpp_Flags) $(SGX_Include_Paths)
Enclave_Cpp_Flags := $(Flags_Just_For_Cpp) $(Common_C_Cpp_Flags) $(SGX_Include_Paths)
### Intel(R) SGX SDK Settings ###
### Project Settings ###

COMMON_INCLUDE_FLAGS = -I. -I${PROJECT_ROOT_DIR}/include -I$(PROJECT_ROOT_DIR)/include/search-worker/trusted -I$(PROJECT_ROOT_DIR)/include/search-worker/trusted/intrinsics -I${HAKES_ROOT_DIR} -I./src/trusted
Enclave_C_Flags += $(COMMON_INCLUDE_FLAGS) $(SGX_RA_TLS_Extra_Flags) -I$(SGX_RA_TLS_DIR)/include
Enclave_Cpp_Flags += -DSGXCLIENT $(COMMON_INCLUDE_FLAGS) $(SGX_RA_TLS_Extra_Flags) -I$(SGX_RA_TLS_DIR)/include
SGXSSL_Link_Flags := -L$(SGXSSL_DIR)/lib64 -Wl,--whole-archive -lsgx_tsgxssl -Wl,--no-whole-archive -lsgx_tsgxssl_crypto -lsgx_pthread
SGX_RA_TLS_LIB=$(SGX_RA_TLS_DIR)/lib
### Project Settings ###

Enclave_Link_Flags := $(SGX_COMMON_CFLAGS) \
	-Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles \
	-L$(SGX_LIBRARY_PATH) \
	-L$(SGX_RA_TLS_LIB) -lratls_ext -lratls_attester_t -lratls_challenger_t \
	-lratls_common_t -lwolfssl.sgx.static.lib \
	$(SGXSSL_Link_Flags) \
	-Wl,--whole-archive -l$(DCAP_TVL_LIB) -l$(Trts_Library_Name) -Wl,--no-whole-archive \
	-Wl,--whole-archive -lsgx_tcmalloc -Wl,--no-whole-archive \
	-Wl,--start-group -lsgx_tstdc -lsgx_pthread -lsgx_omp -lsgx_tcxx -l$(Crypto_Library_Name) \
	-l$(Service_Library_Name) -Wl,--end-group \
	-Wl,-Bstatic -Wl,-Bsymbolic \
	-Wl,-pie,-eenclave_entry -Wl,--export-dynamic \
	-Wl,--defsym,__ImageBase=0 -Wl,--gc-sections \
	-Wl,--version-script=src/trusted/Enclave.lds

### Project Settings ###

### Phony targets ###
.PHONY: all clean enclave_api

### Build all ###
ifeq ($(Build_Mode), HW_RELEASE)
all: Enclave.so
	@echo "Build enclave Server_Enclave.so [$(Build_Mode)|$(SGX_ARCH)] success!"
	@echo
	@echo "*********************************************************************************************************************************************************"
	@echo "PLEASE NOTE: In this mode, please sign the Server_Enclave.so first using Two Step Sign mechanism before you run the app to launch and access the enclave."
	@echo "*********************************************************************************************************************************************************"
	@echo
else
all: Enclave.signed.so
endif

### Edger8r related sourcs ###
SRC_EDL := ${PROJECT_ROOT_DIR}/include/search-worker/trusted/Enclave.edl
EDL_SEARCH_FLAGS := --search-path $(PROJECT_ROOT_DIR)/include --search-path ${PROJECT_ROOT_DIR}/include/trusted --search-path $(SGXSSL_DIR)/include --search-path $(SGX_RA_TLS_DIR)/include --search-path ${HAKES_ROOT_DIR}
src/trusted/Enclave_t.c: $(SGX_EDGER8R) $(SRC_EDL)
	$(SGX_EDGER8R) --trusted $(SRC_EDL) --trusted-dir ./src/trusted --search-path $(SGX_SDK)/include  $(EDL_SEARCH_FLAGS) $(Enclave_Search_Dirs)
	@echo "GEN  =>  $@"

src/trusted/Enclave_t.o: src/trusted/Enclave_t.c
	$(CC) $(Enclave_C_Flags) -c $< -o $@
	@echo "CC   <=  $<"

enclave_api: src/trusted/Enclave_t.c
### Edger8r related sourcs ###

## build files needed from other directory

build_dir:
	@echo "Creating build directory"
	mkdir -p src/trusted/index-build/blas
	mkdir -p src/trusted/index-build/ext
	mkdir -p src/trusted/index-build/impl
	mkdir -p src/trusted/index-build/invlists
	mkdir -p src/trusted/index-build/utils

## index srcs
src/trusted/index-build/%.o: index/%.cpp build_dir
	@echo "GEN1 $< from $@"
	$(CXX) ${Enclave_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/trusted/index-build/blas/%.o: index/blas/%.c build_dir
	@echo "GEN2 $< from $@"
	$(CXX) ${Enclave_Cpp_Flags} -c $< -o $@ -I${PROJECT_ROOT_DIR}/index/blas
	@echo "CXX <= $<"
## index srcs

## other srcs
src/trusted/base64.o: ${HAKES_ROOT_DIR}/utils/base64.cpp
	$(CXX) ${Enclave_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/trusted/hexutil.o: ${HAKES_ROOT_DIR}/utils/hexutil.cpp
	$(CXX) ${Enclave_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/trusted/json.o: ${HAKES_ROOT_DIR}/utils/json.cpp
	$(CXX) ${Enclave_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/trusted/tcrypto_ext.o: ${HAKES_ROOT_DIR}/utils/tcrypto_ext.cpp
	$(CXX) ${Enclave_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/trusted/channel_client.o: $(HAKES_ROOT_DIR)/ratls-channel/common/channel_client.cpp
	$(CXX) $(Enclave_Cpp_Flags) $() -c $< -o $@
	@echo "CXX  <=  $<"

src/trusted/keyservice_worker.o: ${HAKES_ROOT_DIR}/message/keyservice_worker.cpp
	$(CXX) ${Enclave_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/trusted/searchservice.o: ${HAKES_ROOT_DIR}/message/searchservice.cpp
	$(CXX) ${Enclave_Cpp_Flags} -c $< -o $@ -I${PROJECT_ROOT_DIR}/index/blas
	@echo "CXX <= $<"
## other srcs

## build files needed from other directory

src/trusted/worker.o: src/common/worker.cc
	$(CXX) $(Enclave_Cpp_Flags) -c $< -o $@
	@echo "CXX  <=  $<"

# ## build files needed from other directory

src/trusted/ecalls.o: src/trusted/ecalls.cc
	$(CXX) $(Enclave_Cpp_Flags) -c $< -o $@
	@echo "CXX  <=  $<"

### Enclave Image ###
Enclave_Cpp_Objects := src/trusted/ecalls.o src/trusted/worker.o \
	src/trusted/base64.o \
	src/trusted/hexutil.o \
	src/trusted/json.o \
	src/trusted/tcrypto_ext.o \
	src/trusted/channel_client.o \
	src/trusted/keyservice_worker.o \
	src/trusted/searchservice.o \
	src/trusted/index-build/ext/BlockInvertedListsL.o \
	src/trusted/index-build/ext/HakesIndex.o \
	src/trusted/index-build/ext/IdMap.o \
	src/trusted/index-build/ext/index_io_ext.o \
	src/trusted/index-build/ext/IndexFlatCodesL.o \
	src/trusted/index-build/ext/IndexFlatL.o \
	src/trusted/index-build/ext/IndexIVFFastScanL.o \
	src/trusted/index-build/ext/IndexIVFL.o \
	src/trusted/index-build/ext/IndexIVFPQFastScanL.o \
	src/trusted/index-build/ext/IndexRefineL.o \
	src/trusted/index-build/impl/AuxIndexStructures.o \
	src/trusted/index-build/impl/CodePacker.o \
	src/trusted/index-build/impl/IDSelector.o \
	src/trusted/index-build/impl/io.o \
	src/trusted/index-build/impl/kmeans1d.o \
	src/trusted/index-build/impl/pq4_fast_scan_search_1.o \
	src/trusted/index-build/impl/pq4_fast_scan_search_qbs.o \
	src/trusted/index-build/impl/pq4_fast_scan.o \
	src/trusted/index-build/impl/ProductQuantizer.o \
	src/trusted/index-build/invlists/DirectMap.o \
	src/trusted/index-build/invlists/InvertedLists.o \
	src/trusted/index-build/utils/distances_simd.o \
	src/trusted/index-build/utils/distances.o \
	src/trusted/index-build/utils/Heap.o \
	src/trusted/index-build/utils/quantize_lut.o \
	src/trusted/index-build/utils/random.o \
	src/trusted/index-build/utils/sorting.o \
	src/trusted/index-build/utils/utils.o \
	src/trusted/index-build/blas/sgemm.o \
	src/trusted/index-build/blas/lsame.o \
	src/trusted/index-build/Clustering.o \
	src/trusted/index-build/Index.o \
	src/trusted/index-build/IndexFlat.o \
	src/trusted/index-build/IndexFlatCodes.o \
	src/trusted/index-build/IndexIVF.o \
	src/trusted/index-build/IndexIVFPQ.o \
	src/trusted/index-build/IndexPQ.o \
	src/trusted/index-build/VectorTransform.o

Enclave.so: src/trusted/Enclave_t.o $(Enclave_Cpp_Objects)
	$(CXX) $^ -o $@ $(Enclave_Link_Flags)
	@echo "LINK =>  $@"

### Signing ###
Enclave.signed.so: Enclave.so
	$(SGX_ENCLAVE_SIGNER) sign -key src/trusted/Enclave_private.pem -enclave Enclave.so -out $@ -config src/trusted/Enclave.config.xml
	@echo "SIGN =>  $@"
### Sources ###

### Clean command ###
clean:
	rm -rf src/trusted/index-build
	rm -f Enclave.* src/trusted/Enclave_t.*  $(Enclave_Cpp_Objects)
