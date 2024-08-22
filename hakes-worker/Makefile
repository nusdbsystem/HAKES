SGX_SDK ?= /opt/intel/sgxsdk
SGX_MODE ?= HW
SGX_ARCH ?= x64
SGX_DEBUG ?= 0

INSTALL ?= install
INSTALL_PREFIX ?= $(abspath ./install)
INSTALL_LIB_DIR = $(INSTALL_PREFIX)/lib
INSTALL_INCLUDE_DIR = $(INSTALL_PREFIX)/include/hakes-worker
INSTALL_BIN_DIR = $(INSTALL_PREFIX)/bin

.PHONY: all install clean mrproper sgx_u sgx_t enclave_api inference_rt

enclave_api:
	$(MAKE) -ef sgx_u.mk enclave_api SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG) 
	$(MAKE) -ef sgx_t.mk enclave_api SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG) 

no_sgx: inference_rt
	$(MAKE) -ef no_sgx.mk all

sgx_u: inference_rt
	$(MAKE) -ef sgx_u.mk all SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG)

sgx_t: inference_rt
	$(MAKE) -ef sgx_t.mk all SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG)

all: sgx_u sgx_t no_sgx

install:
	$(INSTALL) -d $(INSTALL_INCLUDE_DIR)
	$(INSTALL) -d $(INSTALL_INCLUDE_DIR)/common
	$(INSTALL) -d $(INSTALL_INCLUDE_DIR)/trusted
	$(INSTALL) -d $(INSTALL_INCLUDE_DIR)/untrusted
	$(INSTALL) -C -m 644 include/hakes-worker/common/* $(INSTALL_INCLUDE_DIR)/common
	$(INSTALL) -C -m 644 include/hakes-worker/trusted/*.edl $(INSTALL_INCLUDE_DIR)/trusted
	$(INSTALL) -C -m 644 include/hakes-worker/untrusted/* $(INSTALL_INCLUDE_DIR)/untrusted
	$(INSTALL) -d $(INSTALL_LIB_DIR)
	if [ -f *.signed.so ]; then $(INSTALL) -C -m 664 *.signed.so $(INSTALL_LIB_DIR); fi;
	if [ -f *.a ]; then $(INSTALL) -C -m 644 *.a $(INSTALL_LIB_DIR); fi
	$(INSTALL) -d $(INSTALL_BIN_DIR)
	$(INSTALL) -C -m 755 hakes_server* $(INSTALL_BIN_DIR)

clean:
	$(MAKE) -ef no_sgx.mk clean
	$(MAKE) -ef sgx_u.mk clean
	$(MAKE) -ef sgx_t.mk clean

mrproper: clean
	rm -rf ./install
