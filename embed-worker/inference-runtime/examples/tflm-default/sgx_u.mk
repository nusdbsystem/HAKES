### Project Settings ###
PROJECT_ROOT_DIR := $(shell readlink -f ../../..)
DEP_TFLM_DIR ?= $(shell readlink -f ../../tflm)
### Project Settings ###

### Phony targets ###
.PHONY: all clean enclave_api

all: libuntrusted_inference_rt.a

### Sources ###
## Edger8r related sources ##
enclave_api:
	@echo "Enclave API generated"
## Edger8r related sources ##

libuntrusted_inference_rt.a:
	@echo "Creating untrusted part of inference runtime"
	cp $(DEP_TFLM_DIR)/lib/libtflm_u.a $@
	@echo "Created untrusted part of inference runtime"
### Sources ###

### Clean command ###
clean:
	@echo "Cleaning untrusted part of inference runtime"
	@rm -f libuntrusted_inference_rt.a
