// SPDX-License-Identifier: MIT
#pragma once
#include "ggml.h"
#include "level4_bridge.hpp"
namespace maple_w2::level4 {
// Pure metadata adapter. No tensor_get, no H2D/D2H and no raw VkDeviceAddress casts.
// The caller resolves tensor offset + VkBuffer binding offset in a cached IMPORT.
inline ReadView ggml_read_view(const ggml_tensor*t,const float*imported,size_t allocation_elements,
                               uint32_t tokens,uint32_t width) {
    if(!t||t->type!=GGML_TYPE_F32||t->ne[3]!=1||!imported)
        throw std::invalid_argument("Level4 boundary requires imported F32 ggml tensor");
    for(int i=0;i<3;++i)if(t->nb[i]%4)throw std::invalid_argument("non-F32 byte stride");
    ReadView v;v.data=imported;v.elements=allocation_elements;v.lane_stride=t->nb[0]/4;
    if(t->ne[0]==width&&t->ne[1]==tokens&&t->ne[2]==1){v.head_dim=width;v.token_stride=t->nb[1]/4;v.head_stride=t->nb[1]/4;}
    else if(t->ne[0]>0&&t->ne[1]>0&&t->ne[0]<=width&&uint64_t(t->ne[0])*uint64_t(t->ne[1])==width&&t->ne[2]==tokens){v.head_dim=uint32_t(t->ne[0]);v.head_stride=t->nb[1]/4;v.token_stride=t->nb[2]/4;}
    else throw std::invalid_argument("unsupported ggml logical [D,Q] or [Dh,H,Q] boundary");
    (void)required_elements(v,tokens,width);return v;
}
inline WriteView ggml_write_view(const ggml_tensor*t,float*imported,size_t allocation_elements,
                                 uint32_t tokens,uint32_t width) {
    auto r=ggml_read_view(t,imported,allocation_elements,tokens,width);
    WriteView v{imported,r.elements,r.head_dim,r.token_stride,r.head_stride,r.lane_stride};check_write(v,tokens,width);return v;
}
inline bool is_plain_o_projection(const ggml_tensor*node,const ggml_tensor*raw_fa,const ggml_tensor*wo) {
    return node&&raw_fa&&wo&&node->op==GGML_OP_MUL_MAT&&node->src[0]==wo&&node->src[1]==raw_fa&&
           wo->type==GGML_TYPE_TQ2_0&&node->type==GGML_TYPE_F32;
}
// The caller's graph optimizer must prove all skipped intermediate nodes have no
// outside consumers, recognize both sequential residuals, and preserve any final
// output-row selection. GraphSemantics defaults false intentionally. Never enable
// a region by its tensor NAME alone. This header does NOT install a ggml op hook.
} // namespace
