HAKES_EMBED_ROOT := $(shell readlink -f ../../..)
DEPS_INSTALL_DIR = $(HAKES_EMBED_ROOT)/deps/install

### TFLM
# many variables defined to facilitate local install of tflm
TFLM = tflm
TFLM_DIR = $(HAKES_EMBED_ROOT)/deps/$(TFLM)
TFLM_LOCAL_INCLUDE_DIR = $(DEPS_INSTALL_DIR)/$(TFLM)/include
TFLM_LOCAL_LIB_DIR = $(DEPS_INSTALL_DIR)/$(TFLM)/lib

TFLM_DEP_RELATIVE_PATH := $(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/
TFLM_BUILD_TARGET_PATH = $(TFLM_DIR)/tensorflow/lite/micro/tools/make/gen
TFLM_RELEASE_LIB = $(TFLM_BUILD_TARGET_PATH)/linux_x86_64_release/lib/libtensorflow-microlite.a

# for micro and subdir of micro no recursive install
TFLM_INSTALL_HEADER_PATHS := tensorflow/lite/c \
												tensorflow/lite/core \
												tensorflow/lite/kernels \
												tensorflow/lite/schema

TFLM_INSTALL_NONRECURSIVE_HEADER_PATHS := tensorflow/lite \
												tensorflow/lite/micro \
												tensorflow/lite/micro/kernels \
												tensorflow/lite/micro/memory_planner

TFLM_DEP_FLATBUFFERS_HEADER_PATH := $(TFLM_DEP_RELATIVE_PATH)/flatbuffers/include/flatbuffers
TFLM_DEP_GEMMLOWP_HEADER_PATH := $(TFLM_DEP_RELATIVE_PATH)/gemmlowp
TFLM_DEP_RUY_HEADER_PATH := $(TFLM_DEP_RELATIVE_PATH)/ruy/ruy
TFLM_DEP_LOCAL_PATHS := flatbuffers gemmlowp ruy
### TFLM

.PHONY: tflm_deps tflm_deps_clean

tflm_deps:
	cd $(TFLM_DIR) && make -f tensorflow/lite/micro/tools/make/Makefile clean \
	&& rm -rf tensorflow/lite/micro/tools/make/downloads/ \
	&& make BUILD_TYPE=release -f tensorflow/lite/micro/tools/make/Makefile microlite -j10 \
	&& cd $(HAKES_EMBED_ROOT)
	@echo Installing tflm headers locally
	install -d $(TFLM_LOCAL_INCLUDE_DIR)/$(TFLM)
	for header_dir in $(TFLM_INSTALL_HEADER_PATHS); do \
		install -d $(TFLM_LOCAL_INCLUDE_DIR)/$(TFLM)/$$header_dir; \
		for subdir in `find $(TFLM_DIR)/$$header_dir -type d -not -path '*/.*'`; do \
			dest=$${subdir#"$(TFLM_DIR)"}; \
			install -d $(TFLM_LOCAL_INCLUDE_DIR)/$(TFLM)/$$dest; \
		done; \
		for header in `find $(TFLM_DIR)/$$header_dir -type f -name *.h -not -path '*/.*'`; do \
			dest=$${header#"$(TFLM_DIR)"}; \
			install -C -m 644 $$header $(TFLM_LOCAL_INCLUDE_DIR)/$(TFLM)/$$dest; \
		done \
	done
	for header_dir in $(TFLM_INSTALL_NONRECURSIVE_HEADER_PATHS); do \
		install -d $(TFLM_LOCAL_INCLUDE_DIR)/$(TFLM)/$$header_dir; \
		for header in `find $(TFLM_DIR)/$$header_dir -maxdepth 1 -type f -name *.h -not -path '*/.*'`; do \
			dest=$${header#"$(TFLM_DIR)"}; \
			install -C -m 644 $$header $(TFLM_LOCAL_INCLUDE_DIR)/$(TFLM)/$$dest; \
		done \
	done
	for header_dir in $(TFLM_DEP_LOCAL_PATHS); do \
		install -d $(TFLM_LOCAL_INCLUDE_DIR)/$$header_dir; \
	done
	for subdir in `find $(TFLM_DEP_FLATBUFFERS_HEADER_PATH) -type d -not -path '*/.*'`; do \
		dest=$${subdir#"$(TFLM_DEP_FLATBUFFERS_HEADER_PATH)"}; \
		install -d $(TFLM_LOCAL_INCLUDE_DIR)/flatbuffers/$$dest; \
	done
	for header in `find $(TFLM_DEP_FLATBUFFERS_HEADER_PATH) -type f -name *.h -not -path '*/.*'`; do \
		dest=$${header#"$(TFLM_DEP_FLATBUFFERS_HEADER_PATH)"}; \
		install -C -m 644 $$header $(TFLM_LOCAL_INCLUDE_DIR)/flatbuffers/$$dest; \
	done
	for subdir in `find $(TFLM_DEP_GEMMLOWP_HEADER_PATH) -type d -not -path '*/.*'`; do \
		dest=$${subdir#"$(TFLM_DEP_GEMMLOWP_HEADER_PATH)"}; \
		install -d $(TFLM_LOCAL_INCLUDE_DIR)/gemmlowp/$$dest; \
	done
	for header in `find $(TFLM_DEP_GEMMLOWP_HEADER_PATH) -type f -name *.h -not -path '*/.*'`; do \
		dest=$${header#"$(TFLM_DEP_GEMMLOWP_HEADER_PATH)"}; \
		install -C -m 644 $$header $(TFLM_LOCAL_INCLUDE_DIR)/gemmlowp/$$dest; \
	done
	for subdir in `find $(TFLM_DEP_RUY_HEADER_PATH) -type d -not -path '*/.*'`; do \
		dest=$${subdir#"$(TFLM_DEP_RUY_HEADER_PATH)"}; \
		install -d $(TFLM_LOCAL_INCLUDE_DIR)/ruy/$$dest; \
	done
	for header in `find $(TFLM_DEP_RUY_HEADER_PATH) -type f -name *.h -not -path '*/.*'`; do \
		dest=$${header#"$(TFLM_DEP_RUY_HEADER_PATH)"}; \
		install -C -m 644 $$header $(TFLM_LOCAL_INCLUDE_DIR)/ruy/$$dest; \
	done	
	@echo tflm headers installed
	@echo Installing tflm library locally
	install -d $(TFLM_LOCAL_LIB_DIR)
	install -C -m 644 $(TFLM_RELEASE_LIB) $(TFLM_LOCAL_LIB_DIR)/libtflm.a
	@echo tflm library installed

# buggy tflm overall Makefile clean and clean_downloads commands
# here we directly remove the folders
tflm_deps_clean:
	cd $(TFLM_DIR) \
	&& rm -rf tensorflow/lite/micro/tools/make/gen \
	&& rm -rf tensorflow/lite/micro/tools/make/downloads \
	&& cd $(HAKES_EMBED_ROOT)
