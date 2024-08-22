### Project Settings ###
PROJECT_ROOT_DIR ?= $(shell readlink -f .)
HAKES_ROOT ?= $(shell readlink -f ..)
DEPS_INSTALL_DIR = $(HAKES_ROOT)/deps/install
LIBUV_DIR = $(DEPS_INSTALL_DIR)/libuv
LLHTTP_DIR = $(DEPS_INSTALL_DIR)/llhttp

WARNING_IGNORE = -Wno-sign-compare -Wno-unused-variable -Wno-comment -Wno-unused-function -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-type-limits
COMMON_FLAGS = -O2 -march=native -ftree-vectorize -Wall -Wextra $(WARNING_IGNORE)
App_Cpp_Flags = $(COMMON_FLAGS)
COMMON_INCLUDE_FLAGS = -I. -I$(PROJECT_ROOT_DIR)/include -I${HAKES_ROOT} -I./src/no_sgx
App_Cpp_Flags += $(COMMON_INCLUDE_FLAGS)

App_Link_Flags = -lrt -pthread -lm -lcrypto -lssl
### Project Settings ###

.PHONY: all clean build_dir

all: app_no_sgx hakes_server_no_sgx

## build files needed from other directory
src/no-sgx/base64.o: $(HAKES_ROOT)/utils/base64.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/no-sgx/fileutil.o: $(HAKES_ROOT)/utils/fileutil.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/no-sgx/hexutil.o: $(HAKES_ROOT)/utils/hexutil.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/no-sgx/http.o: $(HAKES_ROOT)/utils/http.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/no-sgx/json.o: $(HAKES_ROOT)/utils/json.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/no-sgx/client_req.o: $(HAKES_ROOT)/message/client_req.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/no-sgx/embed.o: $(HAKES_ROOT)/message/embed.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/no-sgx/searchservice.o: $(HAKES_ROOT)/message/searchservice.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/no-sgx/kvservice.o: $(HAKES_ROOT)/message/kvservice.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

## build files needed from other directory

src/no-sgx/config.o: src/common/config.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/no-sgx/search_result_agg.o: src/common/search_result_agg.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/no-sgx/workerimpl.o: src/common/workerimpl.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

## embed endpoint
src/no-sgx/endpoint.o: $(HAKES_ROOT)/embed-endpoint/endpoint.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/no-sgx/openai_endpoint.o: $(HAKES_ROOT)/embed-endpoint/openai_endpoint.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/no-sgx/huggingface_endpoint.o: $(HAKES_ROOT)/embed-endpoint/huggingface_endpoint.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"
## embed endpoint

src/no-sgx/%.o: src/no-sgx/%.cpp
	@echo "GEN2 $< from $@"
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

Objects := src/no-sgx/workerimpl.o \
	src/no-sgx/config.o \
	src/no-sgx/search_result_agg.o \
	src/no-sgx/data_manager_impl.o \
	src/no-sgx/base64.o \
	src/no-sgx/fileutil.o \
	src/no-sgx/hexutil.o \
	src/no-sgx/http.o \
	src/no-sgx/json.o \
	src/no-sgx/client_req.o \
	src/no-sgx/embed.o \
	src/no-sgx/searchservice.o \
	src/no-sgx/kvservice.o
# endpoint object files
Objects += src/no-sgx/endpoint.o \
	src/no-sgx/openai_endpoint.o \
	src/no-sgx/huggingface_endpoint.o

libhakes_worker.a: $(Objects)
	ar -rcs $@ $^
	@echo "LINK => $@"

app_no_sgx: src/no-sgx/app.cpp libhakes_worker.a
	$(CXX) $(App_Cpp_Flags) $< -o $@ -L. -l:libhakes_worker.a $(App_Link_Flags)

# ## Build embed server ##

Server_Additional_Include_Flags := -Iserver -I$(LIBUV_DIR)/include -I$(LLHTTP_DIR)/include -I$(HAKES_ROOT)/server
Server_Additional_Link_Flags := -L$(LIBUV_DIR)/lib -l:libuv_a.a -L$(LLHTTP_DIR)/lib -l:libllhttp.a -lrt -ldl -lcurl

server/no-sgx/service.o: $(HAKES_ROOT)/server/service.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

server/no-sgx/server.o: $(HAKES_ROOT)/server/server.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

server/no-sgx/hakes_worker.o: server/hakes_worker.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

server/no-sgx/main.o : server/no-sgx/main.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

Server_Objects := server/no-sgx/service.o server/no-sgx/server.o server/no-sgx/hakes_worker.o server/no-sgx/main.o

hakes_server_no_sgx: $(Server_Objects) libhakes_worker.a
	$(CXX) $(Server_Objects) -o $@ -L. -l:libhakes_worker.a $(App_Link_Flags) $(Server_Additional_Link_Flags)

clean:
	@rm -f app_no_sgx ${Objects} libhakes_worker.a server/no-sgx/*.o hakes_server_no_sgx
