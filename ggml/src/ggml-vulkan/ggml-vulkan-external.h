#pragma once

#include <cstddef>
#include <cstdint>

struct ggml_tensor;
struct ggml_vk_external_lease;

struct ggml_vk_external_span {
    uint64_t allocation_id = 0;
    uint64_t generation = 0;
    size_t allocation_size = 0;
    size_t tensor_offset = 0;
    size_t bytes = 0;
    void * memory_handle = nullptr; // duplicated Win32 HANDLE; caller closes it
    uint32_t vendor = 0;
    uint32_t device = 0;
    uint32_t node_mask = 0;
    bool luid_valid = false;
    uint8_t luid[8] = {};
    uint8_t device_uuid[16] = {};
};

struct ggml_vk_external_api {
    uint32_t abi_version = 1;
    bool (*get_span)(ggml_vk_external_lease *, const ggml_tensor *, ggml_vk_external_span *) = nullptr;
    bool (*release_to_external)(ggml_vk_external_lease *) = nullptr;
};
