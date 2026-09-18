// SPDX-License-Identifier: MIT
#pragma once
#include "level4_contract.hpp"
#include "moe_reference.hpp"
namespace maple_w2::level4 {
inline std::vector<float> norm_reference(const std::vector<float>&a,const std::vector<float>&b,
 const std::vector<float>&g,uint32_t width,float eps) {
    if(!width||a.empty()||a.size()%width||(!b.empty()&&a.size()!=b.size())||g.size()!=width)
        throw std::invalid_argument("reference norm sizes");
    std::vector<float> y(a.size());
    for(size_t t=0;t<a.size()/width;++t){double sum=0;for(uint32_t k=0;k<width;++k){float z=a[t*width+k]+(b.empty()?0.f:b[t*width+k]);sum+=double(z)*z;}
        float inv=1.f/std::sqrt(float(sum/width)+eps);for(uint32_t k=0;k<width;++k)y[t*width+k]=((a[t*width+k]+(b.empty()?0.f:b[t*width+k]))*inv)*g[k];}
    return y;
}
inline std::vector<float> router_logits_reference(const std::vector<float>&x,const std::vector<float>&w,uint32_t width,uint32_t experts) {
    if(!width||x.size()%width||w.size()!=size_t(width)*experts)throw std::invalid_argument("reference router sizes");
    std::vector<float>y(x.size()/width*experts);for(size_t t=0;t<x.size()/width;++t)for(uint32_t e=0;e<experts;++e){double s=0;for(uint32_t k=0;k<width;++k)s+=double(x[t*width+k])*w[size_t(e)*width+k];y[t*experts+e]=float(s);}return y;
}
inline std::vector<uint8_t> synthetic_weight(Shape s,uint32_t seed) {
    // Deterministic nonuniform scales. Only code 0,1,2; no all-row-scale assumption.
    std::vector<uint8_t>w(nbytes(s));uint32_t state=seed;
    auto next=[&](){state=state*1664525u+1013904223u;return state;};
    for(size_t b=0;b<w.size();b+=66){for(uint32_t k=0;k<256;++k)set_native_code(w.data()+b,k,(next()>>16)%3);
        put16(w.data()+b+64,float_to_half(float(1+(next()%7))/1024.f));}
    return w;
}
} // namespace
