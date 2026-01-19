cmake_minimum_required(VERSION 3.2)
project(standalone_crt)

get_filename_component(TVM_PROJECT_ROOT "../../../../deps/tvm" ABSOLUTE)
# get_filename_component(TVM_PROJECT_ROOT "/home/xunf/c/project/secure-serverless-inference/deps/tvm" ABSOLUTE)
set(CMAKE_CURRENT_SOURCE_DIR ${TVM_PROJECT_ROOT})
set(USE_MICRO ON)

# include(${PROJECT_ROOT}/cmake/utils/Utils.cmake)
# include(${PROJECT_ROOT}/cmake/modules/StandaloneCrt.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/Utils.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/StandaloneCrt.cmake)
