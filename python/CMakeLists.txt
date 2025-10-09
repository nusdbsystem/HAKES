cmake_minimum_required(VERSION 3.13)
project(hakes)

# Install pybind11 manually for version control.
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v2.10.1
)
FetchContent_MakeAvailable(pybind11)

# list(APPEND CMAKE_MODULE_PATH "./cmake")

find_package(OpenMP REQUIRED)
set(MKL_THREADING gnu_thread)
find_package(MKL CONFIG REQUIRED)

set(BINDING_FILES
    bindings/bindings.cpp
)

set(MODULE_NAME
    _hakes
)

set(SRC_HEADER_PATH
    ../search-worker/include
    ..
)

set(SRC_LIB_PATH
    ../search-worker
)

set(SRC_LIBS
    libworker.a
    OpenMP::OpenMP_CXX
)

# add_subdirectory(pybind11)
pybind11_add_module(${MODULE_NAME} ${BINDING_FILES})
target_include_directories(${MODULE_NAME} PUBLIC ${SRC_HEADER_PATH})
target_link_directories(${MODULE_NAME} PUBLIC ${SRC_LIB_PATH})
target_link_libraries(${MODULE_NAME} PUBLIC ${SRC_LIBS})
target_compile_features(${MODULE_NAME} PRIVATE cxx_std_17)

target_link_libraries(${MODULE_NAME} PRIVATE MKL::MKL)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(${MODULE_NAME}
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})