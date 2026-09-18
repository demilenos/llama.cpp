cmake_minimum_required(VERSION 3.20)

# Minimal extraction of the existing Maple W2A8/XMX core for Prism PTQ1 bring-up.
# No Level4/MoE/tools/tests are built here.
if(NOT WIN32)
  message(FATAL_ERROR "Maple PTQ1 XMX bridge currently supports Windows only")
endif()
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Clang")
  message(FATAL_ERROR "Maple PTQ1 XMX requires Intel oneAPI DPC++/icx/icpx")
endif()

set(MAPLE_PRECISE_FLAGS -O3 -fno-fast-math)
if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  set(MAPLE_PRECISE_FLAGS /O2 /fp:precise)
endif()

find_path(MAPLE_LEVEL_ZERO_INCLUDE_DIR
  level_zero/ze_api.h
  HINTS "$ENV{LEVEL_ZERO_V1_SDK_PATH}/include" "$ENV{LEVEL_ZERO_SDK_PATH}/include")
find_library(MAPLE_LEVEL_ZERO_LIBRARY
  ze_loader
  HINTS "$ENV{LEVEL_ZERO_V1_SDK_PATH}/lib" "$ENV{LEVEL_ZERO_SDK_PATH}/lib")
if(NOT MAPLE_LEVEL_ZERO_INCLUDE_DIR OR NOT MAPLE_LEVEL_ZERO_LIBRARY)
  message(FATAL_ERROR "Set LEVEL_ZERO_V1_SDK_PATH/LEVEL_ZERO_SDK_PATH or MAPLE_LEVEL_ZERO_INCLUDE_DIR + MAPLE_LEVEL_ZERO_LIBRARY")
endif()

add_library(maple-w2-xmx STATIC
  src/maple_w2a8.cpp
  src/maple_w2a8_grouped.cpp)
target_include_directories(maple-w2-xmx PUBLIC include)
target_compile_options(maple-w2-xmx PUBLIC -fsycl PRIVATE ${MAPLE_PRECISE_FLAGS} -fsycl-device-code-split=per_kernel)
target_link_options(maple-w2-xmx PUBLIC -fsycl -fsycl-device-code-split=per_kernel)

add_library(maple-ptq1-ggml-bridge STATIC integration/ggml_ptq1_bridge.cpp)
target_include_directories(maple-ptq1-ggml-bridge
  PUBLIC include
  PRIVATE
    "${CMAKE_SOURCE_DIR}/ggml/include"
    "${CMAKE_SOURCE_DIR}/ggml/src"
    "${CMAKE_SOURCE_DIR}/ggml/src/ggml-vulkan"
    "${MAPLE_LEVEL_ZERO_INCLUDE_DIR}")
target_link_libraries(maple-ptq1-ggml-bridge
  PUBLIC maple-w2-xmx
  PRIVATE "${MAPLE_LEVEL_ZERO_LIBRARY}" dxgi d3d12)
target_compile_definitions(maple-ptq1-ggml-bridge PUBLIC NOMINMAX)
target_compile_options(maple-ptq1-ggml-bridge PRIVATE ${MAPLE_PRECISE_FLAGS})
