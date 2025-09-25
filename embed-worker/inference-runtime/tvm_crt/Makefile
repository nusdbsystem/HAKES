.PHONY: all install clean mrproper build 

### Install tvm crt to the current directory ###
deps:
	@echo Installing tvm crt sources locally
	mkdir -p build && cd build && cmake ../crt_install && make standalone_crt && cd ..
	mkdir -p deps && cp -r build/standalone_crt deps/ && cd deps/standalone_crt && patch -p1 < ../../crt_install/tvm-0.9.patch && patch -p1 < ../../crt_install/tvm-0.9-tls.patch && patch -p1 < ../../crt_install/min-asan.patch && patch -p1 < ../../crt_install/memset-bug-fix.patch && cd ../..
	rm -rf build
	@echo tvm crt sources headers installed
### Install tvm crt to the current directory ###

lib: 
	mkdir -p $@

build: deps lib
	$(MAKE) -ef build.mk all 

all: build

clean:
	$(MAKE) -ef build.mk clean
	@rm -rf lib
