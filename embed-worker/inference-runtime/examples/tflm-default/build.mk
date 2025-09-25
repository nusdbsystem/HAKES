### Project Settings ###
PROJECT_ROOT_DIR ?= $(shell readlink -f ../../..)
HAKES_EMBED_ROOT ?= $(shell readlink -f ../../../..)
DEP_TFLM_DIR ?= $(shell readlink -f ../../tflm)

App_C_Flags := -Wall -Wextra -O2 -march=native -fPIC
App_Cpp_Flags := -Wall -Wextra -O2 -march=native -fPIC

COMMON_INCLUDE_FLAGS :=	-I./src -I$(HAKES_EMBED_ROOT) -I$(PROJECT_ROOT_DIR)/include -I$(DEP_TFLM_DIR)/deps/include -I$(DEP_TFLM_DIR)/deps/include/tflm
ADDITIONAL_FLAGS := -fno-rtti -fno-exceptions -fno-threadsafe-statics -fno-unwind-tables -ffunction-sections -fdata-sections -fmessage-length=0 -DTF_LITE_STATIC_MEMORY -Wdouble-promotion
App_C_Flags += $(COMMON_INCLUDE_FLAGS) $(ADDITIONAL_FLAGS)
App_Cpp_Flags += $(COMMON_INCLUDE_FLAGS) $(ADDITIONAL_FLAGS)
### Project Settings ###

.PHONY: all install clean

all: libinference_rt.a

src/tflm_default.o: src/tflm_default.cpp
	$(CXX) $(App_Cpp_Flags) -Isrc -c $< -o $@
	@echo "CXX   <=  $<"

src/patch.o: src/patch.cpp
	$(CXX) $(App_Cpp_Flags) -Isrc -c $< -o $@
	@echo "CXX   <=  $<"

Objects := src/tflm_default.o  src/patch.o

libinference_rt.a: $(Objects)
	@echo "Creating inference runtime"
	cp $(DEP_TFLM_DIR)/lib/libtflm.a $@
	ar r $@ $^
	@echo "Created inference runtime"

clean:
	@rm -f src/*.o libinference_rt.a
