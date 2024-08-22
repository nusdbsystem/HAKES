### Project Settings ###
PROJECT_ROOT_DIR := $(shell readlink -f ../../..)
DEP_TVMCRT_DIR := $(shell readlink -f ../../tvm_crt)
### Project Settings ###

### Phony targets ###
.PHONY: all clean enclave_api

all: libuntrusted_inference_rt.a

### Sources ###
## Edger8r related sources ##
## Edger8r related sources ##

libuntrusted_inference_rt.a:
	@echo "Creating untrusted part of inference runtime"
	cp $(DEP_TVMCRT_DIR)/lib/libtvm_u.a $@
	@echo "Created untrusted part of inference runtime"
### Sources ###

### Clean command ###
clean:
	@echo "Cleaning untrusted part of inference runtime"
	@rm -f libuntrusted_inference_rt.a
