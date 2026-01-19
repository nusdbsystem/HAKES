### Project Settings ###
PROJECT_ROOT_DIR ?= $(shell readlink -f .)
HAKES_ROOT ?= $(shell readlink -f ..)
DEPS_INSTALL_DIR = $(HAKES_ROOT)/deps/install
LIBUV_DIR = $(DEPS_INSTALL_DIR)/libuv
LLHTTP_DIR = $(DEPS_INSTALL_DIR)/llhttp

WARNING_IGNORE = -Wno-sign-compare -Wno-unused-variable -Wno-comment -Wno-unused-function -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-type-limits
COMMON_FLAGS = -O2 -march=native -ftree-vectorize -Wall -Wextra $(WARNING_IGNORE)
App_Cpp_Flags = $(COMMON_FLAGS)
COMMON_INCLUDE_FLAGS = -I. -I$(PROJECT_ROOT_DIR)/gateway/include -I${HAKES_ROOT} -I$(HAKES_ROOT)/hakes-server/common/server -I$(HAKES_ROOT)/server/common -I$(HAKES_ROOT)/server/embedder
App_Cpp_Flags += $(COMMON_INCLUDE_FLAGS)

App_Link_Flags = -lrt -pthread -lm -lcrypto -lssl
### Project Settings ###

.PHONY: all clean build_dir

all: app hakes_server

## build files needed from other directory
src/base64.o: $(HAKES_ROOT)/server/common/utils/base64.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/fileutil.o: $(HAKES_ROOT)/server/common/utils/fileutil.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/hexutil.o: $(HAKES_ROOT)/server/common/utils/hexutil.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/http.o: $(HAKES_ROOT)/server/common/utils/http.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/json.o: $(HAKES_ROOT)/server/common/utils/json.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/client_req.o: $(HAKES_ROOT)/hakes-server/common/server/message/client_req.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/embed.o: $(HAKES_ROOT)/hakes-server/common/server/message/embed.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/searchservice.o: $(HAKES_ROOT)/hakes-server/common/server/message/searchservice.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/kvservice.o: $(HAKES_ROOT)/hakes-server/common/server/message/kvservice.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

## build files needed from other directory

src/config.o: gateway/src/config.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/search_result_agg.o: gateway/src/search_result_agg.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/workerimpl.o: gateway/src/workerimpl.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

## embed endpoint
src/endpoint.o: $(HAKES_ROOT)/server/embedder/endpoints/endpoint.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/openai_endpoint.o: $(HAKES_ROOT)/server/embedder/endpoints/openai_endpoint.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/huggingface_endpoint.o: $(HAKES_ROOT)/server/embedder/endpoints/huggingface_endpoint.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

src/ollama_endpoint.o: $(HAKES_ROOT)/server/embedder/endpoints/ollama_endpoint.cpp
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"
## embed endpoint

src/%.o: gateway/src/%.cpp
	@echo "GEN2 $< from $@"
	$(CXX) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

Objects := src/workerimpl.o \
	src/config.o \
	src/search_result_agg.o \
	src/data_manager_impl.o \
	src/base64.o \
	src/fileutil.o \
	src/hexutil.o \
	src/http.o \
	src/json.o \
	src/client_req.o \
	src/embed.o \
	src/searchservice.o \
	src/kvservice.o
# endpoint object files
Objects += src/endpoint.o \
	src/openai_endpoint.o \
	src/huggingface_endpoint.o \
	src/ollama_endpoint.o

libhakes_worker.a: $(Objects)
	ar -rcs $@ $^
	@echo "LINK => $@"

app: gateway/src/app.cpp libhakes_worker.a
	$(CXX) $(App_Cpp_Flags) $< -o $@ -L. -l:libhakes_worker.a $(App_Link_Flags)

# ## Build embed server ##

Server_Additional_Include_Flags := -Igateway/server -I$(LIBUV_DIR)/include -I$(LLHTTP_DIR)/include -I$(HAKES_ROOT)/hakes-server/common/server
Server_Additional_Link_Flags := -L$(LIBUV_DIR)/lib -l:libuv_a.a -L$(LLHTTP_DIR)/lib -l:libllhttp.a -lrt -ldl -lcurl

server/service.o: $(HAKES_ROOT)/hakes-server/common/server/service.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

server/server.o: $(HAKES_ROOT)/hakes-server/common/server/server.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

server/hakes_worker.o: gateway/server/hakes_worker.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

server/main.o : gateway/server/main.cpp
	$(CXX) $(App_Cpp_Flags) $(Server_Additional_Include_Flags) -c $< -o $@
	@echo "CXX <= $<"

Server_Objects := server/service.o server/server.o server/hakes_worker.o server/main.o

hakes_server: $(Server_Objects) libhakes_worker.a
	$(CXX) $(Server_Objects) -o $@ -L. -l:libhakes_worker.a $(App_Link_Flags) $(Server_Additional_Link_Flags)

clean:
	@rm -f app ${Objects} libhakes_worker.a server/*.o hakes_server
