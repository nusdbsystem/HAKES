.PHONY: all check install clean mrproper build

check:
ifndef PROJECT_ROOT_DIR
	$(error PROJECT_ROOT_DIR is not set. Please set to secure serverless inference project root directory)
endif
ifndef MODULE_OBJS_DIR
	$(error MODULE_OBJS_DIR is not set. Please set to secure serverless inference project tflm dependency directory)
endif

build:
	$(MAKE) -ef build.mk all MODULE_OBJS_DIR=$(MODULE_OBJS_DIR)

all: check build

INSTALL_PREFIX ?= ./install
install:
	install -d $(INSTALL_PREFIX)/include/tvm-bert
	install -d $(INSTALL_PREFIX)/lib/tvm-bert
	cp lib*inference_rt.a $(INSTALL_PREFIX)/lib/tvm-bert

clean:
	$(MAKE) -ef build.mk clean
