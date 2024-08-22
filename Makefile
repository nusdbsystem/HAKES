# used to mainly build dependencies
PROJECT_ROOT_DIR := $(shell readlink -f .)

DEPS_INSTALL_DIR = $(PROJECT_ROOT_DIR)/deps/install

LIBUV_DIR = $(PROJECT_ROOT_DIR)/deps/libuv
LLHTTP_DIR = $(PROJECT_ROOT_DIR)/deps/llhttp

RATLS_DIR = $(PROJECT_ROOT_DIR)/deps/ratls

.PHONY: all deps mrproper preparation

preparation:
	git submodule update --init --recursive

# make sure ra_tls_options.c is already created
ratls_deps:
	cd $(RATLS_DIR) && make clean \
	&& make option/ra_tls_options.c && mkdir -p deps && make -j4 deps \
	&& make ratls_libs \
	&& make INSTALL_PREFIX=$(PROJECT_ROOT_DIR)/deps/install/ratls install \
	&& cd $(PROJECT_ROOT_DIR)

ratls_deps_clean:
	make -C $(RATLS_DIR) mrproper

server_deps:
	mkdir -p $(DEPS_INSTALL_DIR)
	cd $(LIBUV_DIR) && mkdir -p build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=$(DEPS_INSTALL_DIR)/libuv && make && make install && cd $(PROJECT_ROOT_DIR)
	mkdir -p $(DEPS_INSTALL_DIR)/llhttp/include && mkdir -p $(DEPS_INSTALL_DIR)/llhttp/lib
	npm --version
	cd $(LLHTTP_DIR) && npm install && make && PREFIX=$(DEPS_INSTALL_DIR)/llhttp make install && cd $(PROJECT_ROOT_DIR)

server_deps_clean:
	make -C $(LLHTTP_DIR) clean
	rm -rf $(LLHTTP_DIR)/node_modules
	rm -rf $(LIBUV_DIR)/build

deps: ratls_deps server_deps

all: preparation deps

clean: 
	@echo "nothing"

mrproper: clean server_deps_clean
	rm -rf ${PROJECT_ROOT_DIR}/deps/install
