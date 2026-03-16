### Project Settings ###
PROJECT_ROOT_DIR ?= $(shell readlink -f .)
HAKES_ROOT ?= $(shell readlink -f ../..)

# Detect MongoDB installation from environment variables
# Priority: MONGOCXX_DIR env var -> extract from CPATH -> extract from LD_LIBRARY_PATH -> /usr/local
ifndef MONGOCXX_DIR
  ifdef CPATH
    # Try to find MongoDB headers in CPATH
    MONGOCXX_DIR := $(shell for path in $(subst :, ,$(CPATH)); do if [ -d $$path/mongocxx ]; then echo $$path/..; break; fi; done)
  endif
  ifndef MONGOCXX_DIR
    ifdef LD_LIBRARY_PATH
      # Try to find MongoDB libraries in LD_LIBRARY_PATH
      MONGOCXX_DIR := $(shell for path in $(subst :, ,$(LD_LIBRARY_PATH)); do if [ -f $$path/libmongocxx.so ] || [ -f $$path/libmongocxx.a ]; then echo $$path/..; break; fi; done)
    endif
  endif
  ifndef MONGOCXX_DIR
    # Default to /usr/local
    MONGOCXX_DIR := /usr/local
  endif
endif

BSONCXX_DIR ?= $(MONGOCXX_DIR)
MONGOC_DIR ?= $(MONGOCXX_DIR)

WARNING_IGNORE = -Wno-sign-compare -Wno-unused-variable -Wno-comment -Wno-unused-function -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-type-limits
COMMON_FLAGS = -O2 -march=native -ftree-vectorize -Wall -Wextra $(WARNING_IGNORE)
Store_Cpp_Flags = $(COMMON_FLAGS) -std=c++17
COMMON_INCLUDE_FLAGS = -I. -I$(PROJECT_ROOT_DIR) -I$(HAKES_ROOT)
Store_Cpp_Flags += $(COMMON_INCLUDE_FLAGS)

# MongoDB C++ driver paths
MONGOCXX_CFLAGS := -I$(MONGOCXX_DIR)/include/mongocxx/v_noabi
MONGOCXX_LIBS := -L$(MONGOCXX_DIR)/lib -lmongocxx
BSONCXX_CFLAGS := -I$(BSONCXX_DIR)/include/bsoncxx/v_noabi
BSONCXX_LIBS := -L$(BSONCXX_DIR)/lib -lbsoncxx
MONGOC_CFLAGS := -I$(MONGOC_DIR)/include/mongoc-2.1.2 -I$(MONGOC_DIR)/include/mongoc-2.1.2/mongoc
MONGOC_LIBS := -L$(MONGOC_DIR)/lib -lmongoc-static-1
BSON_CFLAGS := -I$(BSONCXX_DIR)/include/bson-2.1.2 -I$(BSONCXX_DIR)/include/bson-2.1.2/bson
BSON_LIBS := -L$(BSONCXX_DIR)/lib -lbson-static-1

Store_Cpp_Flags += $(MONGOCXX_CFLAGS) $(BSONCXX_CFLAGS) $(MONGOC_CFLAGS) $(BSON_CFLAGS)
Store_Link_Flags = $(MONGOCXX_LIBS) $(BSONCXX_LIBS) $(MONGOC_LIBS) $(BSON_LIBS) -lpthread

### Project Settings ###

.PHONY: all clean build_dir

all: libhakes_store.a

# MongoDB store implementation
build/mongodb.o: infra/mongodb/mongodb.cpp infra/mongodb/mongodb.h repo/store.h
	@mkdir -p build
	$(CXX) $(Store_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"

libhakes_store.a: build/mongodb.o
	ar -rcs $@ $^
	@echo "LINK => $@"

clean:
	@rm -f build/*.o libhakes_store.a
	@rm -rf build/