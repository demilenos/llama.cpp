// SPDX-License-Identifier: MIT
#pragma once
#include "maple_moe.hpp"
#include "level4_contract.hpp"
namespace maple_w2::level4 {
struct Weights {
    const uint8_t *o=nullptr,*q=nullptr,*k=nullptr,*v=nullptr;
    const uint8_t *gate=nullptr,*up=nullptr,*down=nullptr;
    const float *ffn_norm=nullptr,*next_attn_norm=nullptr,*router=nullptr;
    // O/FFN belong to layer L; QKV and next_attn_norm belong to layer L+1.
    // Bootstrap uses layer 0's QKV/attention norm. Terminal does not read QKV.
    Layout dense_layout=Layout::s2tile8,moe_layout=Layout::s2tile8;
    int32_t current_layer=-1,next_layer=-1; // required provenance of bound weight sets
};
struct Bindings {
    ReadView residual, attention; // attention MUST be pre-O, not build_attn(...wo)'s result
    WriteView hidden,q,k,v;       // raw Q/K/V: no QK-norm/RoPE/KV mutation here
};
struct Options {
    MoeOptions moe{};
    // New QKV/O activation rounding is independent of the tested MoE A8 path.
    // A8 is opt-in for attention projections until model quality is measured.
    DensePrecision o_precision=DensePrecision::a16,qkv_precision=DensePrecision::a16;
    maple_w2::Options o_kernel{},qkv_kernel{};
    bool diagnostic_stage_waits=false; // never enabled by default
};
struct Workspace {
    uint8_t* base=nullptr;size_t capacity=0;Plan plan;
    template<class T>T*get(const std::string&n)const {
        const auto&s=plan.at(n);if(sizeof(T)!=s.element_bytes)throw std::logic_error("workspace element size");
        return s.count?reinterpret_cast<T*>(base+s.offset):nullptr;
    }
    A8Workspace a8(const std::string&n)const {return {get<int8_t>(n+".q"),get<float>(n+".scales"),get<int32_t>(n+".invalid"),false};}
    MoeWorkspace moe()const;
};
Workspace bind_workspace(void* device_arena,size_t capacity,Plan plan);
void validate(const Weights&,const Bindings&,const Options&,const Workspace&);
struct Stage {const char* name="";sycl::event event;uint64_t parents=0;};
struct Run {
    std::array<Stage,56> stages{};size_t count=0;sycl::event done;int32_t* status=nullptr;
    MoeScheduleDecision dispatch{};
    uint64_t add(const char*,const sycl::event&,uint64_t parents=0);
};
// Zero internal tensor allocations, copies to/from HOST, imports, or mandatory waits.
// All views must live in this queue's context. Caller manages external ownership
// release/acquire. The final event covers every submitted producer and status write.
// One arena per in-flight request (or explicitly order reuse with dependencies).
Run enqueue(sycl::queue&,const Weights&,const Bindings&,const Options&,const Workspace&,
            const std::vector<sycl::event>& dependencies={});
// Small public kernels support real GPU stage-local validation with the SAME inputs.
sycl::event enqueue_residual_norm(sycl::queue&,const float*a,const float*b,const float*gamma,
    float*residual,float*normalized,int32_t*bad,uint32_t tokens,uint32_t width,float eps,
    const std::vector<sycl::event>&);
sycl::event enqueue_router_logits(sycl::queue&,const float*x,const float*w,float*logits,
    uint32_t tokens,uint32_t width,uint32_t experts,const std::vector<sycl::event>&);
sycl::event enqueue_router_select(sycl::queue&,const float*logits,int32_t*ids,float*routes,
    int32_t*bad,uint32_t tokens,uint32_t experts,uint32_t topk,const std::vector<sycl::event>&);
} // namespace
