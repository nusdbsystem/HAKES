### Project Settings ###
DEP_TFLM_DIR ?= $(shell readlink -f ../../tflm)
HAKES_EMBED_ROOT ?= $(shell readlink -f ../../../..)

.PHONY: all check install clean mrproper build

check:
ifndef PROJECT_ROOT_DIR
	$(error PROJECT_ROOT_DIR is not set. Please set to secure serverless inference project root directory)
endif

build:
	$(MAKE) -ef build.mk all

all: check build

INSTALL_PREFIX ?= ./install
install:
	install -d $(INSTALL_PREFIX)/include/tflm-default
	install -d $(INSTALL_PREFIX)/lib/tflm-default
	cp lib*inference_rt.a $(INSTALL_PREFIX)/lib/tflm-default

clean:
	$(MAKE) -ef build.mk clean
