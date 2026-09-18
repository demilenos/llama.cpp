include_guard(GLOBAL)
# Level4 zero-copy: portable contracts + native API doubles; no GPU claim.
add_executable(maple-level4-zc-contract-tests "${MAPLE_L4_ROOT}/tests/level4_zerocopy_contract_tests.cpp")
target_include_directories(maple-level4-zc-contract-tests PRIVATE "${MAPLE_L4_ROOT}/include")
add_test(NAME maple-level4-zc-contract-tests COMMAND maple-level4-zc-contract-tests)
add_executable(maple-level4-zc-native-mock-tests
    "${MAPLE_L4_ROOT}/tests/level4_zerocopy_native_tests.cpp"
    "${MAPLE_L4_ROOT}/integration/level4_zerocopy_win32.cpp"
    "${MAPLE_L4_ROOT}/src/maple_level4.cpp")
target_include_directories(maple-level4-zc-native-mock-tests PRIVATE
    "${MAPLE_L4_ROOT}/tests/zerocopy_mock" "${MAPLE_L4_ROOT}/tests/level4_mock"
    "${MAPLE_L4_ROOT}/include" "${MAPLE_L4_ROOT}/integration")
target_compile_definitions(maple-level4-zc-native-mock-tests PRIVATE MAPLE_ZC_NATIVE_MOCK=1 MAPLE_LEVEL4_SCALAR_TEST=1)
# Host API doubles exercise ownership and scalar control flow, not optimization.
# Avoid spending minutes optimizing the scalar fake-SYCL lambda instantiations.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(maple-level4-zc-native-mock-tests PRIVATE -O0)
endif()
add_test(NAME maple-level4-zc-native-mock-tests COMMAND maple-level4-zc-native-mock-tests)

option(MAPLE_LEVEL4_ZERO_COPY "Build Windows Vulkan/LevelZero zero-copy transport" OFF)
if(MAPLE_LEVEL4_ZERO_COPY)
    if(NOT WIN32 OR NOT TARGET maple-level4)
        message(FATAL_ERROR "Zero-copy native target needs Windows and MAPLE_BUILD_SYCL")
    endif()
    find_package(Vulkan REQUIRED)
    find_path(MAPLE_LEVEL_ZERO_INCLUDE_DIR level_zero/ze_api.h HINTS "$ENV{LEVEL_ZERO_V1_SDK_PATH}/include" "$ENV{LEVEL_ZERO_SDK_PATH}/include")
    find_library(MAPLE_LEVEL_ZERO_LIBRARY ze_loader HINTS "$ENV{LEVEL_ZERO_V1_SDK_PATH}/lib" "$ENV{LEVEL_ZERO_SDK_PATH}/lib")
    if(NOT MAPLE_LEVEL_ZERO_INCLUDE_DIR OR NOT MAPLE_LEVEL_ZERO_LIBRARY)
        message(FATAL_ERROR "Set MAPLE_LEVEL_ZERO_INCLUDE_DIR and MAPLE_LEVEL_ZERO_LIBRARY (ze_loader.lib)")
    endif()
    add_library(maple-level4-zerocopy STATIC "${MAPLE_L4_ROOT}/integration/level4_zerocopy_win32.cpp")
    target_include_directories(maple-level4-zerocopy PUBLIC "${MAPLE_L4_ROOT}/integration" "${MAPLE_LEVEL_ZERO_INCLUDE_DIR}")
    target_link_libraries(maple-level4-zerocopy PUBLIC maple-level4 Vulkan::Vulkan "${MAPLE_LEVEL_ZERO_LIBRARY}" dxgi d3d12)
    target_compile_definitions(maple-level4-zerocopy PUBLIC VK_USE_PLATFORM_WIN32_KHR NOMINMAX)
    target_compile_options(maple-level4-zerocopy PRIVATE -O3 -fno-fast-math)
    add_executable(maple-level4-zerocopy-probe "${MAPLE_L4_ROOT}/tools/level4_zerocopy_probe.cpp")
    target_link_libraries(maple-level4-zerocopy-probe PRIVATE maple-level4-zerocopy)
endif()
