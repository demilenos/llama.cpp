#pragma once

#include <cstddef>
#include <cstdint>

struct ggml_tensor;

bool ggml_vk_hybrid_supported(const ggml_tensor * node);
bool ggml_vk_hybrid_try(void * backend_ctx, ggml_tensor * node);
bool ggml_vk_hybrid_read_tensors(void * backend_ctx, const ggml_tensor * const * tensors, void * const * data, const size_t * sizes, int count);

struct ggml_vk_hybrid_external_span {
    uint64_t allocation_id;
    void * handle;
    size_t allocation_size;
    size_t offset;
    size_t size;
};

bool ggml_vk_hybrid_get_external_span(const ggml_tensor * tensor, size_t offset, size_t size, ggml_vk_hybrid_external_span * span);

// Internal C++ bridge ABI. Vulkan objects stay in the backend.
#include "ggml.h"
#include <memory>
struct ggml_vk_external_lease;
struct ggml_vk_external_span {
    uint64_t allocation_id, generation;
    size_t allocation_size, bind_offset, tensor_offset, bytes;
    void * memory_handle; // Caller closes the duplicated NT handle.
    uint32_t vendor, device, node_mask;
    bool luid_valid;
    uint8_t luid[8], device_uuid[16];
    std::shared_ptr<void> owner;
};
struct ggml_vk_external_descriptor {
    uint32_t magic = 0x4d4c345a;
    uint32_t abi_version = 1;
};
struct ggml_vk_external_api {
    uint32_t abi_version;
    bool (*get_span)(ggml_vk_external_lease *, const ggml_tensor *, ggml_vk_external_span *);
    bool (*release_to_external)(ggml_vk_external_lease *);
};
using ggml_vk_external_executor = bool (*)(ggml_tensor *, void *, ggml_vk_external_lease *, const ggml_vk_external_api *);
using ggml_vk_external_register = bool (*)(ggml_custom_op_t, ggml_vk_external_executor, uint32_t);
