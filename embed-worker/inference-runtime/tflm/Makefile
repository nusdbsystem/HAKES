PROJECT_ROOT_DIR := $(shell readlink -f ../..)
HAKES_EMBED_ROOT := $(shell readlink -f ../../..)
TFLM_LOCAL_PATH ?= $(HAKES_EMBED_ROOT)/deps/install/tflm

.PHONY: all install clean mrproper build

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

lib: 
	mkdir -p $@

build: deps lib
	$(MAKE) -ef build.mk all 

all: build

clean:
	$(MAKE) -ef build.mk clean
	@rm -rf lib

mrproper: clean
	@rm -rf deps
