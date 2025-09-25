### Project Settings ###
PROJECT_ROOT_DIR ?= $(shell readlink -f ..)
HAKES_EMBED_ROOT ?= $(shell readlink -f ../..)

## Dependency path ##
DEP_TFLM_DIR := $(shell readlink -f tflm)
DEP_TVMCRT_DIR := $(shell readlink -f tvm_crt)
### Project Settings ###

### Choose the inference runtime ###
# INFERENCERT ?= TFLM_DEFAULT
# INFERENCERT ?= TVMCRT_DEFAULT
INFERENCERT ?= TVMCRT_IMAGEEMBED
### Choose the inference runtime ###

## pick the dependencies for inference runtime ##
ifeq ($(INFERENCERT), TFLM_DEFAULT)
	DEP_RT_DIR := $(DEP_TFLM_DIR)
	RT_DIR := examples/tflm-default
	RT_BUILD_OPTIONS := DEP_TFLM_DIR=$(DEP_TFLM_DIR)
else ifeq ($(INFERENCERT), TVMCRT_DEFAULT)
	DEP_RT_DIR := $(DEP_TVMCRT_DIR)
	RT_DIR := examples/tvm-default
	MODULE_OBJS_DIR ?= $(DEP_TVMCRT_DIR)/module
	RT_BUILD_OPTIONS := DEP_TVMCRT_DIR=$(DEP_TVMCRT_DIR) \
		MODULE_OBJS_DIR=$(shell readlink -f $(MODULE_OBJS_DIR))
else ifeq ($(INFERENCERT), TVMCRT_IMAGEEMBED)
	DEP_RT_DIR := $(DEP_TVMCRT_DIR)
	RT_DIR := examples/tvm-image-embed
	MODULE_OBJS_DIR ?= $(DEP_TVMCRT_DIR)/module
	RT_BUILD_OPTIONS := DEP_TVMCRT_DIR=$(DEP_TVMCRT_DIR) \
		MODULE_OBJS_DIR=$(shell readlink -f $(MODULE_OBJS_DIR))
else ifeq ($(INFERENCERT), TVMCRT_BERT)
	DEP_RT_DIR := $(DEP_TVMCRT_DIR)
	RT_DIR := examples/tvm-bert
	MODULE_OBJS_DIR ?= $(DEP_TVMCRT_DIR)/module
	RT_BUILD_OPTIONS := DEP_TVMCRT_DIR=$(DEP_TVMCRT_DIR) \
		MODULE_OBJS_DIR=$(shell readlink -f $(MODULE_OBJS_DIR))
endif

.PHONY: all inference_rt clean install mrproper

all: inference_rt

inference_rt:
	$(MAKE) -C $(DEP_RT_DIR) build PROJECT_ROOT_DIR=$(PROJECT_ROOT_DIR) HAKES_EMBED_ROOT=$(HAKES_EMBED_ROOT)
	$(MAKE) -C $(RT_DIR) build PROJECT_ROOT_DIR=$(PROJECT_ROOT_DIR) HAKES_EMBED_ROOT=$(HAKES_EMBED_ROOT) $(RT_BUILD_OPTIONS)

# install
INSTALL_PREFIX ?= $(abspath ../install)
install: 
	@echo Installing inference runtime
	install -d $(INSTALL_PREFIX)
	cp -r $(PROJECT_ROOT_DIR)/include $(INSTALL_PREFIX)
	$(MAKE) -C $(RT_DIR) install INSTALL_PREFIX=$(INSTALL_PREFIX)
	@echo inference runtime installed

clean:
	$(MAKE) -C $(DEP_RT_DIR) clean
	$(MAKE) -C $(RT_DIR) clean
	@rm -rf lib

mrproper: clean
	$(MAKE) -C $(DEP_RT_DIR) mrproper
	[ -e install ] && rm -rf install || :
