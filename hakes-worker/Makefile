INSTALL ?= install
INSTALL_PREFIX ?= $(abspath ./install)
INSTALL_LIB_DIR = $(INSTALL_PREFIX)/lib
INSTALL_INCLUDE_DIR = $(INSTALL_PREFIX)/include/hakes-worker
INSTALL_BIN_DIR = $(INSTALL_PREFIX)/bin

.PHONY: all install clean mrproper

build:
	$(MAKE) -ef build.mk all

all: build

install:
	$(INSTALL) -d $(INSTALL_INCLUDE_DIR)
	$(INSTALL) -C -m 644 include/hakes-worker/* $(INSTALL_INCLUDE_DIR)
	$(INSTALL) -d $(INSTALL_LIB_DIR)
	if [ -f *.a ]; then $(INSTALL) -C -m 644 *.a $(INSTALL_LIB_DIR); fi
	$(INSTALL) -d $(INSTALL_BIN_DIR)
	$(INSTALL) -C -m 755 hakes_server* $(INSTALL_BIN_DIR)

clean:
	$(MAKE) -ef build.mk clean

mrproper: clean
	rm -rf ./install
