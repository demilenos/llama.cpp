llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

set(LLAMA_CORE_SOURCES
    llama.cpp
    llama-adapter.cpp
    llama-arch.cpp
    llama-batch.cpp
    llama-chat.cpp
    llama-context.cpp
    llama-cparams.cpp
    llama-grammar.cpp
    llama-graph.cpp
    llama-hparams.cpp
    llama-impl.cpp
    llama-io.cpp
    llama-kv-cache.cpp
    llama-kv-cache-iswa.cpp
    llama-kv-cache-dsa.cpp
    llama-kv-cache-dsa-iswa.cpp
    llama-kv-cache-msa.cpp
    llama-kv-cache-dsv4.cpp
    llama-memory.cpp
    llama-memory-hybrid.cpp
    llama-memory-hybrid-iswa.cpp
    llama-memory-hybrid-idx.cpp
    llama-memory-recurrent.cpp
    llama-mmap.cpp
    llama-model-loader.cpp
    llama-model-saver.cpp
    llama-model.cpp
    llama-quant.cpp
    llama-sampler.cpp
    llama-vocab.cpp
    unicode-data.cpp
    unicode.cpp
)

add_library(llama
            ../include/llama.h
            ${LLAMA_CORE_SOURCES}
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 16
)

# exclude non-model sources from unity build
set_source_files_properties(${LLAMA_CORE_SOURCES} ../include/llama.h unicode.h
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

configure_file(llama-version.h.in ${CMAKE_CURRENT_BINARY_DIR}/llama-version.h @ONLY)

target_include_directories(llama PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)
if (LLAMA_MAPLE_LEVEL4)
    if (NOT WIN32 OR NOT TARGET maple-level4-zerocopy OR NOT GGML_VULKAN)
        message(FATAL_ERROR "Maple Level4 runtime requires Windows, GGML_VULKAN and MAPLE_LEVEL4_ZERO_COPY")
    endif()
    add_library(llama-maple-runtime STATIC llama-maple-level4.cpp)
    target_include_directories(llama-maple-runtime PRIVATE . ../ggml/include ../ggml/src)
    target_link_libraries(llama-maple-runtime PRIVATE maple-level4 ggml "${MAPLE_LEVEL_ZERO_LIBRARY}" dxgi d3d12)
    target_include_directories(llama-maple-runtime PRIVATE "${MAPLE_LEVEL_ZERO_INCLUDE_DIR}")
    set_target_properties(llama-maple-runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if (MSVC)
        target_compile_options(llama-maple-runtime PRIVATE /EHsc /fp:precise)
    else()
        target_compile_options(llama-maple-runtime PRIVATE -fexceptions -fno-fast-math)
    endif()
    target_compile_definitions(llama PRIVATE LLAMA_MAPLE_LEVEL4)
    target_link_libraries(llama PRIVATE $<LINK_ONLY:llama-maple-runtime>)
    target_link_options(llama PRIVATE -fsycl)
endif()

if (LLAMA_PTQ1_XMX)
    if (NOT WIN32 OR NOT TARGET maple-w2-xmx OR NOT GGML_VULKAN)
        message(FATAL_ERROR "PTQ1 XMX runtime requires Windows, GGML_VULKAN and Maple SYCL")
    endif()
    find_path(LLAMA_PTQ1_LEVEL_ZERO_INCLUDE_DIR level_zero/ze_api.h
        HINTS "$ENV{LEVEL_ZERO_V1_SDK_PATH}/include" "$ENV{LEVEL_ZERO_SDK_PATH}/include")
    find_library(LLAMA_PTQ1_LEVEL_ZERO_LIBRARY ze_loader
        HINTS "$ENV{LEVEL_ZERO_V1_SDK_PATH}/lib" "$ENV{LEVEL_ZERO_SDK_PATH}/lib")
    if (NOT LLAMA_PTQ1_LEVEL_ZERO_INCLUDE_DIR OR NOT LLAMA_PTQ1_LEVEL_ZERO_LIBRARY)
        message(FATAL_ERROR "PTQ1 XMX requires Level Zero SDK headers and ze_loader.lib")
    endif()

    add_library(llama-ptq1-xmx-runtime STATIC llama-ptq1-xmx.cpp)
    target_include_directories(llama-ptq1-xmx-runtime PRIVATE
        .
        ../ggml/include
        ../ggml/src
        ../maple-w2-xmx-v0.4/include
        "${LLAMA_PTQ1_LEVEL_ZERO_INCLUDE_DIR}")
    target_link_libraries(llama-ptq1-xmx-runtime PRIVATE
        maple-w2-xmx
        ggml
        "${LLAMA_PTQ1_LEVEL_ZERO_LIBRARY}"
        dxgi
        d3d12)
    set_target_properties(llama-ptq1-xmx-runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if (MSVC)
        target_compile_options(llama-ptq1-xmx-runtime PRIVATE /EHsc /fp:precise)
    else()
        target_compile_options(llama-ptq1-xmx-runtime PRIVATE -fexceptions -fno-fast-math)
    endif()
    target_compile_definitions(llama PRIVATE LLAMA_PTQ1_XMX)
    target_link_libraries(llama PRIVATE $<LINK_ONLY:llama-ptq1-xmx-runtime>)
    target_link_options(llama PRIVATE -fsycl)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
