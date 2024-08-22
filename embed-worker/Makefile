SGX_SDK ?= /opt/intel/sgxsdk
SGX_MODE ?= HW
SGX_ARCH ?= x64
SGX_DEBUG ?= 0

### Choose the inference runtime ###
# INFERENCERT ?= TFLM_DEFAULT
# INFERENCERT ?= TVMCRT_DEFAULT
# INFERENCERT ?= TVMCRT_IMAGEEMBED
INFERENCERT ?= TVMCRT_BERT
### Choose the inference runtime ###

INSTALL ?= install
INSTALL_PREFIX ?= $(abspath ./install)
INSTALL_LIB_DIR = $(INSTALL_PREFIX)/lib
INSTALL_INCLUDE_DIR = $(INSTALL_PREFIX)/include
INSTALL_BIN_DIR = $(INSTALL_PREFIX)/bin

ifeq ($(INFERENCERT), TFLM_DEFAULT)
	INFERENCE_RT_INCLUDE_DIR ?= $(INSTALL_PREFIX)/include/tflm-default 
	INFERENCE_RT_LINK_DIR ?= $(INSTALL_PREFIX)/lib/tflm-default 
else ifeq ($(INFERENCERT), TVMCRT_DEFAULT)
	INFERENCE_RT_INCLUDE_DIR ?= $(INSTALL_PREFIX)/include/tvm-default 
	INFERENCE_RT_LINK_DIR ?= $(INSTALL_PREFIX)/lib/tvm-default 
else ifeq ($(INFERENCERT), TVMCRT_IMAGEEMBED)
	INFERENCE_RT_INCLUDE_DIR ?= $(INSTALL_PREFIX)/include/tvm-image-embed 
	INFERENCE_RT_LINK_DIR ?= $(INSTALL_PREFIX)/lib/tvm-image-embed
else ifeq ($(INFERENCERT), TVMCRT_BERT)
	INFERENCE_RT_INCLUDE_DIR ?= $(INSTALL_PREFIX)/include/tvm-bert 
	INFERENCE_RT_LINK_DIR ?= $(INSTALL_PREFIX)/lib/tvm-bert
else
	$(error "Invalid inference runtime")
endif

.PHONY: all install clean mrproper sgx_u sgx_t enclave_api inference_rt

inference_rt:
	echo $(MODULE_OBJS_DIR)
	echo $(shell readlink -f $(MODULE_OBJS_DIR))
	$(MAKE) -C inference-runtime all INFERENCERT=$(INFERENCERT) MODULE_OBJS_DIR=$(shell readlink -f $(MODULE_OBJS_DIR))
	$(INSTALL) -d $(INSTALL_PREFIX) 
	$(MAKE) -C inference-runtime install INFERENCERT=$(INFERENCERT) INSTALL_PREFIX=$(INSTALL_PREFIX)

inference_rt_no_sgx:
	echo $(MODULE_OBJS_DIR)
	echo $(shell readlink -f $(MODULE_OBJS_DIR))
	$(MAKE) -C inference-runtime inference_rt_no_sgx INFERENCERT=$(INFERENCERT) MODULE_OBJS_DIR=$(shell readlink -f $(MODULE_OBJS_DIR))
	$(INSTALL) -d $(INSTALL_PREFIX) 
	$(MAKE) -C inference-runtime install INFERENCERT=$(INFERENCERT) INSTALL_PREFIX=$(INSTALL_PREFIX)

enclave_api:
	$(MAKE) -ef sgx_u.mk enclave_api SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG) INFERENCE_RT_INCLUDE_DIR=$(INFERENCE_RT_INCLUDE_DIR)
	$(MAKE) -ef sgx_t.mk enclave_api SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG) INFERENCE_RT_INCLUDE_DIR=$(INFERENCE_RT_INCLUDE_DIR)

no_sgx: inference_rt_no_sgx
	$(MAKE) -ef no_sgx.mk all INFERENCE_RT_INCLUDE_DIR=$(INFERENCE_RT_INCLUDE_DIR) INFERENCE_RT_LINK_DIR=$(INFERENCE_RT_LINK_DIR)

sgx_u: inference_rt
	$(MAKE) -ef sgx_u.mk all SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG) INFERENCE_RT_INCLUDE_DIR=$(INFERENCE_RT_INCLUDE_DIR) INFERENCE_RT_LINK_DIR=$(INFERENCE_RT_LINK_DIR)

sgx_t: inference_rt
	$(MAKE) -ef sgx_t.mk all SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG) INFERENCE_RT_INCLUDE_DIR=$(INFERENCE_RT_INCLUDE_DIR) INFERENCE_RT_LINK_DIR=$(INFERENCE_RT_LINK_DIR)

all: sgx_u sgx_t no_sgx

install:
	$(INSTALL) -d $(INSTALL_INCLUDE_DIR)
	cp -r include/* $(INSTALL_INCLUDE_DIR)
	$(INSTALL) -d $(INSTALL_LIB_DIR)
	if [ -f *.signed.so ]; then $(INSTALL) -C -m 664 *.signed.so $(INSTALL_LIB_DIR); fi;
	if [ -f *.a ]; then $(INSTALL) -C -m 644 *.a $(INSTALL_LIB_DIR); fi;
	$(INSTALL) -d $(INSTALL_BIN_DIR)
	$(INSTALL) -C -m 755 embed_server* $(INSTALL_BIN_DIR)

clean:
	$(MAKE) -C inference-runtime INFERENCERT=$(INFERENCERT) clean
	$(MAKE) -ef no_sgx.mk clean
	$(MAKE) -ef sgx_u.mk clean
	$(MAKE) -ef sgx_t.mk clean

mrproper: clean
	$(MAKE) -C inference-runtime INFERENCERT=$(INFERENCERT) mrproper
	rm -rf ./install

# MODULE_OBJS_DIR=../examples/tvm-mb/mobilenet1.0/ make all
# MODULE_OBJS_DIR=../examples/tvm-mb/mobilenet1.0/ make clean