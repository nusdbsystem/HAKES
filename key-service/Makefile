SGX_SDK ?= /opt/intel/sgxsdk
SGX_MODE ?= HW
SGX_ARCH ?= x64
SGX_DEBUG ?= 0

INSTALL ?= install
INSTALL_PREFIX ?= ./install
INSTALL_LIB_DIR = $(INSTALL_PREFIX)/lib
INSTALL_INCLUDE_DIR = $(INSTALL_PREFIX)/include
INSTALL_BIN_DIR = $(INSTALL_PREFIX)/bin

.PHONY: all install clean mrproper sgx_u sgx_t enclave_api

enclave_api:
	$(MAKE) -ef sgx_u.mk enclave_api SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG) Enclave_Search_Dirs="$(Enclave_Search_Dirs)"
	$(MAKE) -ef sgx_t.mk enclave_api SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG) Enclave_Search_Dirs="$(Enclave_Search_Dirs)"

sgx_u:
	$(MAKE) -ef sgx_u.mk all SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG) Enclave_Search_Dirs="$(Enclave_Search_Dirs)"

sgx_t:
	$(MAKE) -ef sgx_t.mk all SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG) Enclave_Search_Dirs="$(Enclave_Search_Dirs)"

all: sgx_u sgx_t

install:
	$(INSTALL) -d $(INSTALL_INCLUDE_DIR)
	cp -r include/* $(INSTALL_INCLUDE_DIR)
	$(INSTALL) -d $(INSTALL_LIB_DIR)
	$(INSTALL) -C -m 664 *.signed.so $(INSTALL_LIB_DIR)
	$(INSTALL) -C -m 644 *.a $(INSTALL_LIB_DIR)
	$(INSTALL) -d $(INSTALL_BIN_DIR)
	$(INSTALL) -C -m 755 key_server $(INSTALL_BIN_DIR)

clean:
	$(MAKE) -ef sgx_u.mk clean
	$(MAKE) -ef sgx_t.mk clean

mrproper: clean
	rm -rf ./install
