// SPDX-License-Identifier: MIT
#pragma once

struct ggml_tensor;
struct ggml_vk_external_lease;
struct ggml_vk_external_api;

bool maple_ptq1_xmx_supported(const ggml_tensor * node);
bool maple_ptq1_xmx_execute(
    ggml_tensor * node,
    ggml_vk_external_lease * lease,
    const ggml_vk_external_api * api);
