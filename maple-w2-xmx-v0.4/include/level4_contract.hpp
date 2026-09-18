// SPDX-License-Identifier: MIT
#pragma once
#include "tq2_reference.hpp"
#include "expert_token_tiles.hpp"
#include "expert_tiles_reference.hpp"
#include <array>
#include <map>
namespace maple_w2::level4 {
// This is an attention-to-attention island, NOT a complete SYCL model backend.
enum class Entry { bootstrap, advance, terminal };
enum class DensePrecision { a16, a8 };
struct Config {
    uint32_t tokens=1, hidden=2048, attention=2048, q_width=2048, kv_width=512;
    uint32_t ffn=512, experts=256, topk=8;
    Entry entry=Entry::advance;
    float ffn_epsilon=1e-6f, next_epsilon=1e-6f;
};
inline size_t plus_size(size_t a,size_t b) {
    if(b>SIZE_MAX-a)throw std::overflow_error("Level4 size overflow");
    return a+b;
}
inline size_t aligned(size_t n) { return plus_size(n,63)&~size_t(63); }
inline bool has_post(Entry e) { return e!=Entry::bootstrap; }
inline bool has_qkv(Entry e) { return e!=Entry::terminal; }
inline void check_config(const Config&c) {
    if(c.entry!=Entry::bootstrap&&c.entry!=Entry::advance&&c.entry!=Entry::terminal)
        throw std::invalid_argument("unknown Level4 entry");
    if(!c.tokens||c.tokens>2048||!c.hidden||c.hidden%256||!c.attention||c.attention%256||
       !c.ffn||c.ffn%256||!c.experts||c.experts>256||!c.topk||c.topk>8||c.topk>c.experts||
       !c.q_width||c.q_width%8||!c.kv_width||c.kv_width%8)
        throw std::invalid_argument("Level4 requires Q1..2048, K256, M8, E<=256, topk<=8");
    if(!std::isfinite(c.ffn_epsilon)||c.ffn_epsilon<=0||!std::isfinite(c.next_epsilon)||c.next_epsilon<=0)
        throw std::invalid_argument("RMSNorm epsilon must come from model metadata and be positive");
    (void)nbytes({c.hidden,c.ffn,c.experts});
    (void)nbytes({c.ffn,c.hidden,c.experts});
    (void)nbytes({c.attention,c.hidden,1});
    if(c.hidden>UINT32_MAX/4||c.attention>UINT32_MAX/4||c.ffn>UINT32_MAX/4)
        throw std::invalid_argument("Level4 gather offset overflow");
}
struct GraphSemantics {
    // Host graph recognizer must positively establish this contract.
    bool raw_fa_before_o=false, raw_qkv_before_qk_norm=false;
    bool sequential_residual=false, maple_clamped_swiglu=false;
    bool weights_validated=false, same_device_context=false;
    bool active_lora=false, control_vector=false, projection_bias=false;
    bool extra_scale_or_rotation=false, graph_has_external_interior_consumers=false;
    bool final_row_selection_inside=false;
};
inline const char* reject_graph(const Config&c,const GraphSemantics&s) {
    if(!s.weights_validated)return "weights not validated at load";
    if(!s.same_device_context)return "no verified same-device/context imported storage";
    if(has_post(c.entry)&&!s.raw_fa_before_o)return "attention binding already includes O projection";
    if(has_qkv(c.entry)&&!s.raw_qkv_before_qk_norm)return "output binding must precede Q/K norm and RoPE";
    if(has_post(c.entry)&&(!s.sequential_residual||!s.maple_clamped_swiglu))return "unsupported FFN/residual graph";
    if(s.active_lora||s.control_vector||s.projection_bias||s.extra_scale_or_rotation)
        return "unmaterialized adapter/cvec/bias/scale/rotation";
    if(s.graph_has_external_interior_consumers)return "interior tensor has an external graph consumer";
    if(s.final_row_selection_inside)return "last-layer output row selection needs an explicit boundary";
    return nullptr;
}
// F32 views with arbitrary token/head/lane strides; all strides are FLOAT elements.
// Supports actual FA head-major layouts without a host conversion.
struct ReadView {
    const float* data=nullptr; size_t elements=0;
    uint32_t head_dim=0; size_t token_stride=0,head_stride=0,lane_stride=1;
};
struct WriteView {
    float* data=nullptr; size_t elements=0;
    uint32_t head_dim=0; size_t token_stride=0,head_stride=0,lane_stride=1;
};
template<class V>inline size_t offset(const V&v,size_t t,size_t c) {
    return t*v.token_stride+(c/v.head_dim)*v.head_stride+(c%v.head_dim)*v.lane_stride;
}
template<class V>inline size_t required_elements(const V&v,uint32_t tokens,uint32_t width) {
    if(!v.data||!v.head_dim||width%v.head_dim||!v.lane_stride||!tokens||!width)
        throw std::invalid_argument("invalid Level4 F32 boundary view");
    auto last=plus_size(checked_mul(tokens-1,v.token_stride),checked_mul(width/v.head_dim-1,v.head_stride));
    last=plus_size(last,checked_mul(v.head_dim-1,v.lane_stride));
    if(plus_size(last,1)>v.elements)throw std::invalid_argument("Level4 boundary view exceeds allocation");
    return last+1;
}
inline ReadView contiguous_read(const float*p,uint32_t q,uint32_t w) {return {p,checked_mul(q,w),w,w,w,1};}
inline WriteView contiguous_write(float*p,uint32_t q,uint32_t w) {return {p,checked_mul(q,w),w,w,w,1};}
// Validate a write view has no repeated addresses, including Q=1 head strides.
inline void check_write(const WriteView&v,uint32_t q,uint32_t w) {
    (void)required_elements(v,q,w);
    const size_t lane_span=plus_size(checked_mul(v.head_dim-1,v.lane_stride),1);
    const size_t heads=w/v.head_dim;
    // Also accept head-major [heads,Q,D] and token-major [Q,heads,D].
    const bool token_major=(heads<=1||v.head_stride>=lane_span) &&
       (q<=1||v.token_stride>=plus_size(checked_mul(heads-1,v.head_stride),lane_span));
    const bool head_major=(q<=1||v.token_stride>=lane_span) &&
       (heads<=1||v.head_stride>=plus_size(checked_mul(q-1,v.token_stride),lane_span));
    if(!token_major&&!head_major)throw std::invalid_argument("overlapping output view strides");
}
// Constant-time proof for disjoint Q/K/V slices in a token-interleaved QKV buffer.
inline bool disjoint_token_slices(const WriteView&a,uint32_t aw,const WriteView&b,uint32_t bw) {
    auto row_contiguous=[](const WriteView&v,uint32_t width){return v.lane_stride==1 && (v.head_dim==width||v.head_stride==v.head_dim);};
    if(!row_contiguous(a,aw)||!row_contiguous(b,bw)||a.token_stride!=b.token_stride||!a.token_stride)return false;
    uintptr_t ap=reinterpret_cast<uintptr_t>(a.data),bp=reinterpret_cast<uintptr_t>(b.data);
    uint32_t left=aw,right=bw;if(bp<ap){std::swap(ap,bp);std::swap(left,right);}
    if((bp-ap)%sizeof(float))return false;
    size_t gap=(bp-ap)/sizeof(float);
    return gap>=left && gap<=a.token_stride && right<=a.token_stride-gap;
}
struct Span {size_t offset=0,count=0,element_bytes=0;size_t bytes()const{return checked_mul(count,element_bytes);} };
struct Plan {
    Config config;std::map<std::string,Span> spans;size_t bytes=0;
    uint32_t gate_split=1,down_split=1,o_split=1,qkv_split=1,gate_tile=1,down_tile=1;
    void add(const std::string&n,size_t count,size_t size) {
        bytes=aligned(bytes);if(!spans.emplace(n,Span{bytes,count,size}).second)throw std::logic_error("duplicate region");
        bytes=plus_size(bytes,checked_mul(count,size));
    }
    const Span&at(const std::string&n)const{return spans.at(n);}
};
inline void check_split(uint32_t k,uint32_t s) {
    if((s!=1&&s!=2&&s!=4&&s!=8)||(k/256)%s)throw std::invalid_argument("invalid Level4 split-K");
}
inline Plan make_plan(Config c,uint32_t gs=1,uint32_t ds=1,uint32_t os=1,uint32_t qs=1,
                      uint32_t gt=1,uint32_t dt=1) {
    check_config(c);check_split(c.hidden,gs);check_split(c.ffn,ds);check_split(c.attention,os);check_split(c.hidden,qs);
    check_tokens_per_tile(gt);check_tokens_per_tile(dt);
    Plan p{c,{},0,gs,ds,os,qs,gt,dt};
    const size_t q=c.tokens, x=checked_mul(q,c.hidden),jobs=checked_mul(q,c.topk),h=checked_mul(jobs,c.ffn),d=checked_mul(jobs,c.hidden);
    auto f=[&](const char*n,size_t count){p.add(n,count,4);};
    auto a8=[&](const std::string&n,size_t count){p.add(n+".q",count,1);p.add(n+".scales",count/32,4);p.add(n+".invalid",count/32,4);};
    f("residual",x);f("ids_zero",q);f("status",q);f("norm_bad",q);f("next_bad",q);f("router_bad",q);
    f("dense_o_status",q);f("dense_q_status",q);f("dense_kv_status",q);
    if(has_post(c.entry)) {
        f("attention",checked_mul(q,c.attention));f("o",x);f("ff_residual",x);f("ff_norm",x);f("moe_out",x);
        f("router_logits",checked_mul(q,c.experts));f("routes",jobs);f("ids",jobs);
        f("gate",h);f("up",h);f("hidden",h);f("down",d);f("gate_status",jobs);f("down_status",jobs);
        f("hidden_invalid",h);f("output_invalid",x);
        f("gate_split",gs>1?checked_mul(checked_mul(h,gs),2):0);f("down_split",ds>1?checked_mul(d,ds):0);
        f("o_split",os>1?checked_mul(x,os):0);
        a8("input_a8",x);a8("hidden_a8",h);a8("o_a8",checked_mul(q,c.attention));
        f("counts",c.experts+1);f("offsets",c.experts+2);f("sorted_to_job",jobs);
        const size_t gemm_cap=expert_tile_capacity(jobs,c.experts,4);
        for(auto key:{".expert",".begin",".rows"})p.add(std::string("gemm_tiles")+key,gemm_cap,4);
        p.add("gemm_tiles.count",1,4);p.add("gemm_tiles.invalid",c.experts+1,4);
        for(const auto&item:std::array<std::pair<const char*,uint32_t>,2>{{{"gate_tiles",gt},{"down_tiles",dt}}}) {
            const size_t cap=item.second>1?expert_token_tile_capacity(jobs,c.experts,item.second):0;
            for(auto key:{".first",".length",".expert"})p.add(std::string(item.first)+key,cap,4);
            p.add(std::string(item.first)+".total",cap?1:0,4);
        }
    }
    if(has_qkv(c.entry)) {
        f("next_norm",x);f("q",checked_mul(q,c.q_width));f("k",checked_mul(q,c.kv_width));f("v",checked_mul(q,c.kv_width));
        f("q_split",qs>1?checked_mul(checked_mul(q,c.q_width),qs):0);
        f("kv_split",qs>1?checked_mul(checked_mul(checked_mul(q,c.kv_width),qs),2):0);a8("qkv_a8",x);
    }
    f("hidden_out",x);p.bytes=aligned(p.bytes);return p;
}
// FP32 full softmax then top-k renormalization, deterministic lower-ID ties.
// Not a promise of ggml bitwise router parity: compare route IDs during integration.
inline void route_reference(const float*logits,uint32_t e,uint32_t topk,int32_t*ids,float*routes) {
    float mx=-std::numeric_limits<float>::infinity();
    for(uint32_t i=0;i<e;++i){if(!std::isfinite(logits[i]))throw std::invalid_argument("nonfinite router logits");mx=std::max(mx,logits[i]);}
    float sum=0;for(uint32_t i=0;i<e;++i)sum+=std::exp(logits[i]-mx);
    std::vector<uint32_t> order(e);for(uint32_t i=0;i<e;++i)order[i]=i;
    std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return logits[a]>logits[b];});
    float z=0;for(uint32_t k=0;k<topk;++k){ids[k]=int32_t(order[k]);routes[k]=std::exp(logits[order[k]]-mx)/sum;z+=routes[k];}
    for(uint32_t k=0;k<topk;++k)routes[k]/=z+1e-20f;
}
} // namespace
