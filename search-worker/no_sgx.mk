### Project Settings ###
PROJECT_ROOT_DIR ?= $(shell readlink -f .)
HAKES_ROOT_DIR ?= $(shell readlink -f ..)
DEPS_INSTALL_DIR = $(HAKES_ROOT_DIR)/deps/install
LIBUV_DIR = $(DEPS_INSTALL_DIR)/libuv
LLHTTP_DIR = $(DEPS_INSTALL_DIR)/llhttp
MKL_LIBRARY_PATH ?= $(HOME)/intel/oneapi/mkl/2024.0/lib/

WARNING_IGNORE = -Wno-sign-compare -Wno-unused-variable -Wno-comment -Wno-unused-function -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-type-limits
COMMON_FLAGS = -fpic -fopenmp -ftree-vectorize -Wall -Wextra $(WARNING_IGNORE)
ifeq ($(DEBUG), 1)
	COMMON_FLAGS += -march=native -O2 -g -fsanitize=address
else
	COMMON_FLAGS += -Ofast -march=native
endif

App_Cpp_Flags = -std=c++17 $(COMMON_FLAGS)
COMMON_INCLUDE_FLAGS = -I. -I${PROJECT_ROOT_DIR}/include -I${HAKES_ROOT_DIR}
App_Cpp_Flags += $(COMMON_INCLUDE_FLAGS)

App_Link_Flags = -lrt -pthread -lm -lcrypto -lssl -fopenmp -ldl -L$(MKL_LIBRARY_PATH) -lmkl_rt

.PHONY: all clean build_dir

all: build_dir app_no_sgx search_server_no_sgx app3_no_sgx index_test

build_dir:
	@echo "Creating build directory"
	mkdir -p src/no-sgx/index-build
	mkdir -p src/no-sgx/index-build/blas
	mkdir -p src/no-sgx/index-build/ext
	mkdir -p src/no-sgx/index-build/impl
	mkdir -p src/no-sgx/index-build/invlists
	mkdir -p src/no-sgx/index-build/utils

## index srcs
src/no-sgx/index-build/%.o: index/%.cpp build_dir
	@echo "GEN1 $< from $@"
	$(CXX) ${App_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/no-sgx/index-build/blas/%.o: index/blas/%.c build_dir
	@echo "GEN1 $< from $@"
	$(CXX) ${COMMON_FLAGS} -c $< -o $@ -I${PROJECT_ROOT_DIR}/index/blas
	@echo "CXX <= $<"

## index srcs

## other srcs
src/no-sgx/io.o: ${HAKES_ROOT_DIR}/utils/io.cpp
	$(CXX) ${App_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/no-sgx/fileutil.o: ${HAKES_ROOT_DIR}/utils/fileutil.cpp
	$(CXX) ${App_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/no-sgx/hexutil.o: ${HAKES_ROOT_DIR}/utils/hexutil.cpp
	$(CXX) ${App_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/no-sgx/json.o: ${HAKES_ROOT_DIR}/utils/json.cpp
	$(CXX) ${App_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/no-sgx/searchservice.o: ${HAKES_ROOT_DIR}/message/searchservice.cpp
	$(CXX) ${App_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"
## other srcs

src/no-sgx/worker.o: src/common/worker.cc
	$(CXX) ${App_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/no-sgx/%.o: src/no-sgx/%.cc
	@echo "GEN2 $< from $@"
	$(CXX) ${App_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

Objects := src/no-sgx/worker.o \
	src/no-sgx/io.o \
	src/no-sgx/fileutil.o \
	src/no-sgx/hexutil.o \
	src/no-sgx/json.o \
	src/no-sgx/searchservice.o \
	src/no-sgx/index-build/ext/BlockInvertedListsL.o \
	src/no-sgx/index-build/ext/HakesIndex.o \
	src/no-sgx/index-build/ext/HakesFlatIndex.o \
	src/no-sgx/index-build/ext/IdMap.o \
	src/no-sgx/index-build/ext/index_io_ext.o \
	src/no-sgx/index-build/ext/IndexFlatCodesL.o \
	src/no-sgx/index-build/ext/IndexFlatL.o \
	src/no-sgx/index-build/ext/IndexIVFFastScanL.o \
	src/no-sgx/index-build/ext/IndexIVFL.o \
	src/no-sgx/index-build/ext/IndexIVFPQFastScanL.o \
	src/no-sgx/index-build/ext/IndexRefineL.o \
	src/no-sgx/index-build/ext/IndexScalarQuantizerL.o \
	src/no-sgx/index-build/impl/AuxIndexStructures.o \
	src/no-sgx/index-build/impl/CodePacker.o \
	src/no-sgx/index-build/impl/IDSelector.o \
	src/no-sgx/index-build/impl/io.o \
	src/no-sgx/index-build/impl/kmeans1d.o \
	src/no-sgx/index-build/impl/pq4_fast_scan_search_1.o \
	src/no-sgx/index-build/impl/pq4_fast_scan_search_qbs.o \
	src/no-sgx/index-build/impl/pq4_fast_scan.o \
	src/no-sgx/index-build/impl/ProductQuantizer.o \
	src/no-sgx/index-build/impl/ScalarQuantizer.o \
	src/no-sgx/index-build/invlists/DirectMap.o \
	src/no-sgx/index-build/invlists/InvertedLists.o \
	src/no-sgx/index-build/utils/distances_simd.o \
	src/no-sgx/index-build/utils/distances.o \
	src/no-sgx/index-build/utils/Heap.o \
	src/no-sgx/index-build/utils/partitioning.o \
	src/no-sgx/index-build/utils/quantize_lut.o \
	src/no-sgx/index-build/utils/random.o \
	src/no-sgx/index-build/utils/sorting.o \
	src/no-sgx/index-build/utils/utils.o \
	src/no-sgx/index-build/Clustering.o \
	src/no-sgx/index-build/Index.o \
	src/no-sgx/index-build/IndexFlat.o \
	src/no-sgx/index-build/IndexFlatCodes.o \
	src/no-sgx/index-build/IndexIVF.o \
	src/no-sgx/index-build/IndexIVFPQ.o \
	src/no-sgx/index-build/IndexPQ.o \
	src/no-sgx/index-build/VectorTransform.o

libworker.a: ${Objects}
	ar -rcs $@ $^
	@echo "LINK => $@"

app_no_sgx: test/no-sgx/app.cc libworker.a
	$(CXX) ${App_Cpp_Flags} $< -L. -l:libworker.a -o test/no-sgx/$@ ${COMMON_LINK_FLAGS} $(App_Link_Flags)

app3_no_sgx: test/no-sgx/app3.cc libworker.a
	$(CXX) ${App_Cpp_Flags} $< -L. -l:libworker.a -o test/no-sgx/$@ ${COMMON_LINK_FLAGS} $(App_Link_Flags)

index_test: test/no-sgx/index_test.cpp libworker.a
	$(CXX) ${App_Cpp_Flags} $< -L. -l:libworker.a -o test/no-sgx/$@ ${COMMON_LINK_FLAGS} $(App_Link_Flags)

## Build search server ##

Server_Additional_Include_Flags := -Iserver -I$(LIBUV_DIR)/include -I$(LLHTTP_DIR)/include -I$(HAKES_ROOT_DIR)/server
Server_Additional_Link_Flags := -L$(LIBUV_DIR)/lib -l:libuv_a.a -L$(LLHTTP_DIR)/lib -l:libllhttp.a -lrt -ldl

ifeq ($(DEBUG), 1)
	Server_Additional_Link_Flags += -g -fsanitize=address
endif

server/no-sgx/service.o: $(HAKES_ROOT_DIR)/server/service.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

server/no-sgx/server.o: $(HAKES_ROOT_DIR)/server/server.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

server/no-sgx/search_worker.o: server/search_worker.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

server/no-sgx/main.o : server/no-sgx/main.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

Server_Objects := server/no-sgx/service.o server/no-sgx/server.o server/no-sgx/search_worker.o server/no-sgx/main.o

search_server_no_sgx: $(Server_Objects) libworker.a
	$(CXX) $(Server_Objects) -o $@ -L. -l:libworker.a $(App_Link_Flags) $(Server_Additional_Link_Flags)

clean:
	rm -rf src/no-sgx/index-build
	rm -f app_no_sgx ${Objects} libworker.a server/no-sgx/*.o search_server_no_sgx app3_no_sgx
