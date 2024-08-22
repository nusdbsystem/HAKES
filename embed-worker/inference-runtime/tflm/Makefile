PROJECT_ROOT_DIR := $(shell readlink -f ../..)
HAKES_EMBED_ROOT := $(shell readlink -f ../../..)
TFLM_LOCAL_PATH ?= $(HAKES_EMBED_ROOT)/deps/install/tflm

SGX_SDK ?= /opt/intel/sgxsdk
SGX_MODE ?= HW
SGX_ARCH ?= x64
SGX_DEBUG ?= 0

.PHONY: all install clean mrproper enclave_api no_sgx sgx_u sgx_t 

### Install tflm to the current directory ###
DEP_LOCAL_DIR := deps

deps:
	$(MAKE) -ef tflm_deps.mk tflm_deps
	@echo Installing tflm headers locally
	mkdir -p $(DEP_LOCAL_DIR)/include
	mkdir -p $(DEP_LOCAL_DIR)/lib
	cp -r $(TFLM_LOCAL_PATH)/include/* $(DEP_LOCAL_DIR)/include
	cp -r $(TFLM_LOCAL_PATH)/lib/* $(DEP_LOCAL_DIR)/lib
	@echo tflm headers installed
### Install tflm to the current directory ###

enclave_api:
	$(MAKE) -ef sgx_u.mk enclave_api SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG)
	$(MAKE) -ef sgx_t.mk enclave_api SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG)

lib: 
	mkdir -p $@

no_sgx: deps lib
	$(MAKE) -ef no_sgx.mk all 

sgx_u: deps lib
	$(MAKE) -ef sgx_u.mk all SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG)

sgx_t: deps lib
	$(MAKE) -ef sgx_t.mk all SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG)

all: sgx_u sgx_t no_sgx

clean:
	$(MAKE) -ef no_sgx.mk clean
	$(MAKE) -ef sgx_u.mk clean
	$(MAKE) -ef sgx_t.mk clean
	@rm -rf lib

mrproper: clean
	@rm -rf deps
