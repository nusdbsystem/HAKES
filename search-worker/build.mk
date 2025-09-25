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
	COMMON_FLAGS += -march=native -O2 -g
else
	COMMON_FLAGS += -Ofast -march=native
endif

ifeq ($(ASAN), 1)
	COMMON_FLAGS += -fsanitize=address
endif

App_Cpp_Flags = -std=c++17 $(COMMON_FLAGS)
COMMON_INCLUDE_FLAGS = -I. -I${PROJECT_ROOT_DIR}/include -I${HAKES_ROOT_DIR}
App_Cpp_Flags += $(COMMON_INCLUDE_FLAGS)

App_Link_Flags = -lrt -pthread -lm -lcrypto -lssl -fopenmp -ldl -L$(MKL_LIBRARY_PATH) -lmkl_rt

.PHONY: all clean

all: app search_server app3 index_test worker_test

## index srcs
src/index-build/%.o: index/%.cpp
	@echo "GEN1 $< from $@"
	$(CXX) ${App_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/index-build/blas/%.o: index/blas/%.c
	@echo "GEN1 $< from $@"
	$(CXX) ${COMMON_FLAGS} -c $< -o $@ -I${PROJECT_ROOT_DIR}/index/blas
	@echo "CXX <= $<"

## index srcs

## other srcs
src/io.o: ${HAKES_ROOT_DIR}/utils/io.cpp
	$(CXX) ${App_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/fileutil.o: ${HAKES_ROOT_DIR}/utils/fileutil.cpp
	$(CXX) ${App_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/hexutil.o: ${HAKES_ROOT_DIR}/utils/hexutil.cpp
	$(CXX) ${App_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/json.o: ${HAKES_ROOT_DIR}/utils/json.cpp
	$(CXX) ${App_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/searchservice.o: ${HAKES_ROOT_DIR}/message/searchservice.cpp
	$(CXX) ${App_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/checkpoint.o: src/checkpoint.cpp
	$(CXX) ${App_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

## other srcs

src/worker.o: src/worker.cpp
	$(CXX) ${App_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

src/%.o: src/%.cpp
	@echo "GEN2 $< from $@"
	$(CXX) ${App_Cpp_Flags} -c $< -o $@
	@echo "CXX <= $<"

Objects := src/worker.o \
	src/io.o \
	src/fileutil.o \
	src/hexutil.o \
	src/json.o \
	src/checkpoint.o \
	src/searchservice.o \
	src/index-build/ext/BlockInvertedListsL.o \
	src/index-build/ext/HakesIndex.o \
	src/index-build/ext/HakesFlatIndex.o \
	src/index-build/ext/IdMap.o \
	src/index-build/ext/index_io_ext.o \
	src/index-build/ext/IndexFlatCodesL.o \
	src/index-build/ext/IndexFlatL.o \
	src/index-build/ext/IndexIVFFastScanL.o \
	src/index-build/ext/IndexIVFL.o \
	src/index-build/ext/IndexIVFPQFastScanL.o \
	src/index-build/ext/IndexRefineL.o \
	src/index-build/ext/IndexScalarQuantizerL.o \
	src/index-build/impl/AuxIndexStructures.o \
	src/index-build/impl/CodePacker.o \
	src/index-build/impl/IDSelector.o \
	src/index-build/impl/io.o \
	src/index-build/impl/kmeans1d.o \
	src/index-build/impl/pq4_fast_scan_search_1.o \
	src/index-build/impl/pq4_fast_scan_search_qbs.o \
	src/index-build/impl/pq4_fast_scan.o \
	src/index-build/impl/ProductQuantizer.o \
	src/index-build/impl/ScalarQuantizer.o \
	src/index-build/invlists/DirectMap.o \
	src/index-build/invlists/InvertedLists.o \
	src/index-build/utils/distances_simd.o \
	src/index-build/utils/distances.o \
	src/index-build/utils/Heap.o \
	src/index-build/utils/partitioning.o \
	src/index-build/utils/quantize_lut.o \
	src/index-build/utils/random.o \
	src/index-build/utils/sorting.o \
	src/index-build/utils/utils.o \
	src/index-build/Clustering.o \
	src/index-build/Index.o \
	src/index-build/IndexFlat.o \
	src/index-build/IndexFlatCodes.o \
	src/index-build/IndexIVF.o \
	src/index-build/IndexIVFPQ.o \
	src/index-build/IndexPQ.o \
	src/index-build/VectorTransform.o

libworker.a: ${Objects}
	ar -rcs $@ $^
	@echo "LINK => $@"

app: test/app.cpp libworker.a
	$(CXX) ${App_Cpp_Flags} $< -L. -l:libworker.a -o test/$@ ${COMMON_LINK_FLAGS} $(App_Link_Flags)

app3: test/app3.cpp libworker.a
	$(CXX) ${App_Cpp_Flags} $< -L. -l:libworker.a -o test/$@ ${COMMON_LINK_FLAGS} $(App_Link_Flags)

index_test: test/index_test.cpp libworker.a
	$(CXX) ${App_Cpp_Flags} $< -L. -l:libworker.a -o test/$@ ${COMMON_LINK_FLAGS} $(App_Link_Flags)

worker_test: test/worker_test.cpp libworker.a
	$(CXX) ${App_Cpp_Flags} $< -L. -l:libworker.a -o test/$@ ${COMMON_LINK_FLAGS} $(App_Link_Flags)

## Build search server ##

Server_Additional_Include_Flags := -Iserver -I$(LIBUV_DIR)/include -I$(LLHTTP_DIR)/include -I$(HAKES_ROOT_DIR)/server
Server_Additional_Link_Flags := -L$(LIBUV_DIR)/lib -l:libuv_a.a -L$(LLHTTP_DIR)/lib -l:libllhttp.a -lrt -ldl

ifeq ($(DEBUG), 1)
	Server_Additional_Link_Flags += -g
endif

ifeq ($(ASAN), 1)
	Server_Additional_Link_Flags += -fsanitize=address
endif

server/service.o: $(HAKES_ROOT_DIR)/server/service.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

server/server.o: $(HAKES_ROOT_DIR)/server/server.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

server/search_worker.o: server/search_worker.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

server/main.o : server/main.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

Server_Objects := server/service.o server/server.o server/search_worker.o server/main.o

search_server: $(Server_Objects) libworker.a
	$(CXX) $(Server_Objects) -o $@ -L. -l:libworker.a $(App_Link_Flags) $(Server_Additional_Link_Flags)

clean:
	rm -rf src/index-build
	rm -f search_server
	rm -f test/app ${Objects} libworker.a server/*.o test/search_server test/app3 test/index_test test/worker_test
