INSTALL ?= install
INSTALL_PREFIX ?= ./install
INSTALL_LIB_DIR = $(INSTALL_PREFIX)/lib
INSTALL_INCLUDE_DIR = $(INSTALL_PREFIX)/include/search-worker
INSTALL_BIN_DIR = $(INSTALL_PREFIX)/bin

.PHONY: all install clean mrproper

build: 
	@echo "Creating build directory"
	mkdir -p src/index-build
	mkdir -p src/index-build/blas
	mkdir -p src/index-build/ext
	mkdir -p src/index-build/impl
	mkdir -p src/index-build/invlists
	mkdir -p src/index-build/utils
	$(MAKE) -ef build.mk all 

all: build 

install:
	$(INSTALL) -d $(INSTALL_INCLUDE_DIR)
	$(INSTALL) -C -m 644 include/search-worker/* $(INSTALL_INCLUDE_DIR)
	$(INSTALL) -d $(INSTALL_LIB_DIR)
	if [ -f *.a ]; then $(INSTALL) -C -m 644 *.a $(INSTALL_LIB_DIR); fi
	$(INSTALL) -d $(INSTALL_BIN_DIR)
	$(INSTALL) -C -m 755 search_server* $(INSTALL_BIN_DIR)

clean:
	$(MAKE) -ef build.mk clean

mrproper: clean
	rm -rf ./install
