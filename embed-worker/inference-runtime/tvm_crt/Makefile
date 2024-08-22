SGX_SDK ?= /opt/intel/sgxsdk
SGX_MODE ?= HW
SGX_ARCH ?= x64
SGX_DEBUG ?= 0

.PHONY: all install clean mrproper enclave_api no_sgx sgx_u sgx_t 

### Install tvm crt to the current directory ###
deps:
	@echo Installing tvm crt sources locally
	mkdir -p build && cd build && cmake ../crt_install && make standalone_crt && cd ..
	mkdir -p deps && cp -r build/standalone_crt deps/ && cd deps/standalone_crt && patch -p1 < ../../crt_install/tvm-0.9.patch && patch -p1 < ../../crt_install/tvm-0.9-tls.patch && patch -p1 < ../../crt_install/min-asan.patch && patch -p1 < ../../crt_install/memset-bug-fix.patch && cd ../..
	rm -rf build
	@echo tvm crt sources headers installed
### Install tvm crt to the current directory ###

enclave_api:
	$(MAKE) -ef sgx_u.mk enclave_api SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG) Enclave_Search_Dirs="$(Enclave_Search_Dirs)"
	$(MAKE) -ef sgx_t.mk enclave_api SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG) Enclave_Search_Dirs="$(Enclave_Search_Dirs)"

lib: 
	mkdir -p $@

no_sgx: deps lib
	$(MAKE) -ef no_sgx.mk all 

sgx_u: deps lib
	$(MAKE) -ef sgx_u.mk all SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG) Enclave_Search_Dirs="$(Enclave_Search_Dirs)"

sgx_t: deps lib
	$(MAKE) -ef sgx_t.mk all SGX_MODE=$(SGX_MODE) SGX_DEBUG=$(SGX_DEBUG) Enclave_Search_Dirs="$(Enclave_Search_Dirs)"

all: sgx_u sgx_t no_sgx

clean:
	$(MAKE) -ef no_sgx.mk clean
	$(MAKE) -ef sgx_u.mk clean
	$(MAKE) -ef sgx_t.mk clean
	@rm -rf lib
