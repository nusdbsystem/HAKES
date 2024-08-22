### Project Settings ###
DEP_TFLM_DIR ?= $(shell readlink -f ../../tflm)
HAKES_EMBED_ROOT ?= $(shell readlink -f ../../../..)
### Intel(R) SGX SDK Settings ###
SGX_SDK ?= /opt/intel/sgxsdk
SGX_MODE ?= HW
SGX_ARCH ?= x64
SGX_DEBUG ?= 0
### Intel(R) SGX SDK Settings ###

.PHONY: all check install clean mrproper sgx_u sgx_t enclave_api

check:
ifndef PROJECT_ROOT_DIR
	$(error PROJECT_ROOT_DIR is not set. Please set to secure serverless inference project root directory)
endif

enclave_api:
	$(MAKE) -ef sgx_u.mk enclave_api SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG)
	$(MAKE) -ef sgx_t.mk enclave_api SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG)

no_sgx:
	$(MAKE) -ef no_sgx.mk all

sgx_u:
	$(MAKE) -ef sgx_u.mk all SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG)

sgx_t:
	$(MAKE) -ef sgx_t.mk all SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG)

all: check sgx_u sgx_t no_sgx

INSTALL_PREFIX ?= ./install
install:
	install -d $(INSTALL_PREFIX)/include/tflm-default
	install -d $(INSTALL_PREFIX)/lib/tflm-default
	cp lib*inference_rt.a $(INSTALL_PREFIX)/lib/tflm-default
	cp Enclave.edl $(INSTALL_PREFIX)/include/tflm-default/InferenceRT_Enclave.edl

clean:
	$(MAKE) -ef no_sgx.mk clean
	$(MAKE) -ef sgx_u.mk clean
	$(MAKE) -ef sgx_t.mk clean
