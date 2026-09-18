#pragma once

#include "ggml.h"

#include <cstdint>
#include <memory>

// Graph-owned descriptor for the Vulkan Level4 executor.
// Weight tensors are model-owned and immutable; boundary tensors are node inputs.
struct llama_maple_level4_params {
    uint32_t magic = 0x4d4c345a;
    uint32_t abi_version = 1;
    ggml_tensor * o = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * next_attn_norm = nullptr;
    ggml_tensor * router = nullptr;

    uint32_t hidden = 0;
    uint32_t attention = 0;
    uint32_t q_width = 0;
    uint32_t kv_width = 0;
    uint32_t ffn = 0;
    uint32_t experts = 0;
    uint32_t topk = 0;
    float epsilon = 0.0f;
    float clamp = 0.0f;
    int current_layer = -1;
    int next_layer = -1;
    unsigned token_tile = 1;

    // Graph-owned packed weights and imported allocation leases.
    std::shared_ptr<void> runtime;
};

void llama_maple_level4_compute(ggml_tensor * dst, int ith, int nth, void * userdata);

void llama_maple_level4_register();
