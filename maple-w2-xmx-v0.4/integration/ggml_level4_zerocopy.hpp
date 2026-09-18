// SPDX-License-Identifier: MIT
#pragma once
#include "ggml_level4_contract.hpp"
#include "level4_zerocopy_win32.hpp"
namespace maple_w2::level4::zc {
// tensor_offset is already the FINAL offset of this tensor inside VkBuffer,
// including view offsets resolved by ggml's actual allocator. Never add t->view_offs
// again here, and never reinterpret t->data / vkGetBufferDeviceAddress as USM.
struct TensorBinding {
    const ggml_tensor* tensor=nullptr;
    std::shared_ptr<Buffer> storage;
    size_t tensor_offset=0;
    ReadView read(uint32_t tokens,uint32_t width)const {
        if(!storage||tensor_offset>=storage->bytes())throw std::invalid_argument("unresolved ggml shared tensor");
        const size_t capacity=(storage->bytes()-tensor_offset)/sizeof(float);
        auto*ptr=storage->at<float>(tensor_offset,capacity);
        auto v=ggml_read_view(tensor,ptr,capacity,tokens,width);
        if(!token_contiguous(v,tokens,width))throw std::invalid_argument("ggml boundary is strided: export alone cannot remove a layout copy");
        return v;
    }
    WriteView write(uint32_t tokens,uint32_t width)const {
        const auto r=read(tokens,width);
        WriteView v{const_cast<float*>(r.data),r.elements,r.head_dim,r.token_stride,r.head_stride,r.lane_stride};
        check_write(v,tokens,width);return v;
    }
};
// This adapter changes storage binding only. It DOES NOT install a graph matcher
// or blindly skip nodes. Existing Level4 GraphSemantics validation still applies.
} // namespace
