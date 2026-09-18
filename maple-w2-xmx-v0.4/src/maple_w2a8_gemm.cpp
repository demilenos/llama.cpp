// SPDX-License-Identifier: MIT
#include "maple_w2a8_gemm.hpp"
#include "dg2_validation.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <iostream>
namespace maple_w2 {
namespace es=sycl::ext::intel::esimd;
namespace xmx=sycl::ext::intel::esimd::xmx;
template<int R,bool Pair> class Tq2MultiTokenG32;
template<int R> class MultiTokenS2S8Probe;

template<int R,bool Pair>
static sycl::event launch_gemm(sycl::queue&q,DeviceProblem p,A8Options opt,A8Workspace a,
    ExpertTilesWorkspace tiles,const std::vector<sycl::event>&deps) {
    const size_t jobs=size_t(p.tokens)*p.topk;
    const uint32_t nr=p.shape.m/8,nk=p.shape.k/256;
    const size_t capacity=expert_tile_capacity(jobs,p.shape.experts,R);
    const size_t logical=capacity*nr;
    const size_t global=((logical+opt.kernel.local_size-1)/opt.kernel.local_size)*opt.kernel.local_size;
    return q.submit([&](sycl::handler&h) {
        h.depends_on(deps);
        h.parallel_for<Tq2MultiTokenG32<R,Pair>>(
            sycl::nd_range<1>{sycl::range<1>(global),sycl::range<1>(opt.kernel.local_size)},
            [=](sycl::nd_item<1>it) SYCL_ESIMD_KERNEL {
                const size_t id=it.get_global_linear_id();if(id>=logical)return;
                const uint32_t rt=uint32_t(id%nr);const size_t ti=id/nr;
                const int32_t nt=*tiles.count;
                if(nt<0 || size_t(nt)>capacity) {if(id==0)p.status[0]=2;return;}
                if(ti>=size_t(nt))return; // no host readback, unused launch slots exit here
                const int32_t eid=tiles.expert[ti],start=tiles.begin[ti],rows=tiles.rows[ti];
                if(eid<0||uint32_t(eid)>p.shape.experts||start<0||rows<1||rows>R||size_t(start)+rows>jobs) {
                    if(id==0)p.status[0]=2;return; // caller also validates planner and full coverage
                }
                es::simd<int32_t,R> original=-1;
                es::simd<int32_t,R> active=0;
                #pragma unroll
                for(int r=0;r<R;++r) {
                    if(r<rows) {
                        const int32_t j=opt.job_order[size_t(start)+r];
                        if(j>=0 && size_t(j)<jobs) {
                            original[r]=j;
                            const bool valid_expert=uint32_t(eid)<p.shape.experts;
                            const bool same_id=valid_expert && p.ids[j]==eid;
                            active[r]=same_id?1:0;
                            if(!same_id && rt==0)p.status[j]=valid_expert?2:1;
                        } else if(rt==0)p.status[size_t(start)+r]=2;
                    }
                }
                es::simd<float,R*8> acc0=0.f,acc1=0.f;
                if(uint32_t(eid)<p.shape.experts) {
                    const size_t tb=(size_t(eid)*nr+rt)*nk*528;
                    es::simd<uint32_t,8> so(0,2);
                    es::simd<uint32_t,16> bo(0,4);
                    es::simd<uint32_t,32> ao(0,1);
                    for(uint32_t kb=0;kb<nk;++kb) {
                        es::simd<uint16_t,8> ds0=es::gather<uint16_t,8>(reinterpret_cast<const uint16_t*>(p.w0+tb+size_t(kb)*528+512),so);
                        es::simd<sycl::half,8> h0=ds0.template bit_cast_view<sycl::half>();
                        es::simd<float,8> dw0=es::convert<float>(h0),dw1=0.f;
                        if constexpr(Pair) {
                            es::simd<uint16_t,8> ds1=es::gather<uint16_t,8>(reinterpret_cast<const uint16_t*>(p.w1+tb+size_t(kb)*528+512),so);
                            es::simd<sycl::half,8> h1=ds1.template bit_cast_view<sycl::half>();dw1=es::convert<float>(h1);
                        }
                        for(uint32_t kg=0;kg<256;kg+=32) {
                            // B: K32 x N8, 2 bits/weight = 64 bytes. Loaded ONCE
                            // for R activation rows. No expanded global weight tensor.
                            const size_t wb=tb+size_t(kb)*528+size_t(kg/32)*64;
                            es::simd<uint32_t,16> b0=es::gather<uint32_t,16>(reinterpret_cast<const uint32_t*>(p.w0+wb),bo),b1=0u;
                            if constexpr(Pair)b1=es::gather<uint32_t,16>(reinterpret_cast<const uint32_t*>(p.w1+wb),bo);
                            // Src2 A is row-major [R,32]. Padded rows use all-zero A;
                            // no pointer for a padded row is dereferenced.
                            es::simd<int8_t,R*32> av=0;
                            es::simd<float,R> da=0.f;
                            #pragma unroll
                            for(int r=0;r<R;++r) {
                                if(active[r]) {
                                    const size_t j=size_t(original[r]),xr=p.per_selection?j:j/p.topk;
                                    const size_t ax=xr*p.shape.k+kb*256+kg;
                                    av.template select<32,1>(r*32)=es::gather<int8_t,32>(a.q+ax,ao);
                                    da[r]=a.scales[ax/32];
                                }
                            }
                            es::simd<int32_t,R*8> i0=0,i1=0;
                            i0=xmx::dpas<8,R,int32_t,int32_t,uint32_t,int8_t,
                                xmx::dpas_argument_type::s2,xmx::dpas_argument_type::s8>(i0,b0,av);
                            if constexpr(Pair)i1=xmx::dpas<8,R,int32_t,int32_t,uint32_t,int8_t,
                                xmx::dpas_argument_type::s2,xmx::dpas_argument_type::s8>(i1,b1,av);
                            // G32 is NOT changed to G128/G256. Every token row has
                            // its OWN activation scale, every 8-output block its dw.
                            // Preserve v0.4's K/group order and FP32 expression.
                            #pragma unroll
                            for(int r=0;r<R;++r) {
                                es::simd<int32_t,8> z0=i0.template select<8,1>(r*8);
                                es::simd<float,8> f0=acc0.template select<8,1>(r*8);
                                f0+=es::convert<float>(z0)*(dw0*float(da[r]));
                                acc0.template select<8,1>(r*8)=f0;
                                if constexpr(Pair) {
                                    es::simd<int32_t,8> z1=i1.template select<8,1>(r*8);
                                    es::simd<float,8> f1=acc1.template select<8,1>(r*8);
                                    f1+=es::convert<float>(z1)*(dw1*float(da[r]));
                                    acc1.template select<8,1>(r*8)=f1;
                                }
                            }
                        }
                    }
                }
                es::simd<uint32_t,8> oo(0,4);
                #pragma unroll
                for(int r=0;r<R;++r) {
                    if(original[r]>=0) {
                        const size_t oi=size_t(original[r])*p.shape.m+rt*8;
                        es::simd<float,8> z0=acc0.template select<8,1>(r*8);
                        es::scatter<float,8>(p.y0+oi,oo,z0);
                        if constexpr(Pair) {
                            es::simd<float,8> z1=acc1.template select<8,1>(r*8);
                            es::scatter<float,8>(p.y1+oi,oo,z1);
                        }
                    }
                }
            });
    });
}
A8Run enqueue_a8_gemm(sycl::queue&q,const DeviceProblem&p,A8Options o,A8Workspace w,
    const ExpertTilesWorkspace&tiles,uint32_t repeat,const std::vector<sycl::event>&deps) {
    validate_dg2_device(q.get_device());(void)nbytes(p.shape);validate_token_tile(repeat);
    if(!p.tokens||p.tokens>2048||!p.topk||p.topk>256||p.topk>p.shape.experts||p.shape.experts>256||
       !p.w0||!p.x||!p.ids||!p.y0||!p.status||(p.w1&&!p.y1)||(!p.w1&&p.y1)||
       !w.q||!w.scales||!w.invalid||!o.job_order||o.group!=32||p.layout!=Layout::s2tile8||
       o.kernel.split_k!=1||!o.kernel.local_size||o.kernel.local_size>32||
       (o.mode!=A8Mode::staged&&o.mode!=A8Mode::prequantized)||
       !tiles.expert||!tiles.begin||!tiles.rows||!tiles.count||!tiles.invalid||
       tiles.capacity<expert_tile_capacity(size_t(p.tokens)*p.topk,p.shape.experts,repeat)||tiles.expert_capacity<p.shape.experts)
        throw std::invalid_argument("multi-token GEMM requires G32/s2tile8/grouped/split1 and valid persistent scratch");
    if(p.activation_type!=ActivationType::f16&&p.activation_type!=ActivationType::f32)
        throw std::invalid_argument("invalid activation type");
    A8Run r;std::vector<sycl::event> d=deps;
    if(o.mode==A8Mode::staged){r.quant=enqueue_quantize_a8_g32(q,p,w,deps);r.has_quant=true;d={r.quant};}
    if(repeat==4)r.main=p.w1?launch_gemm<4,true>(q,p,o,w,tiles,d):launch_gemm<4,false>(q,p,o,w,tiles,d);
    else r.main=p.w1?launch_gemm<8,true>(q,p,o,w,tiles,d):launch_gemm<8,false>(q,p,o,w,tiles,d);
    r.done=r.main;return r;
}

template<int R>static void probe_repeat(sycl::queue&q) {
    constexpr int cases=37;
    std::vector<uint32_t> wb(cases*16,0);
    std::vector<int8_t> ab(cases*R*32);
    std::vector<int32_t> cb(cases*R*8),ref(cb.size()),got(cb.size());
    for(int c=0;c<cases;++c) {
        for(int k=0;k<32;++k)for(int n=0;n<8;++n) {
            int b=(c==1?-1:((k*13+n*7+c)%4-2));
            wb[c*16+(k/16)*8+n]|=(uint32_t(b)&3u)<<(2*(k%16));
        }
        for(int r=0;r<R;++r) {
            for(int k=0;k<32;++k)ab[(c*R+r)*32+k]=int8_t(c==1?-128:(c>=2&&c<34?(k==(c-2+r)%32?1:0):((k*37+r*53+c*19)%256-128)));
            for(int n=0;n<8;++n) {
                const size_t i=(c*R+r)*8+n;cb[i]=r*101-n*23+c;int z=cb[i];
                for(int k=0;k<32;++k){int b=c==1?-1:((k*13+n*7+c)%4-2);z+=int(ab[(c*R+r)*32+k])*b;}
                ref[i]=z;
            }
        }
    }
    uint32_t* dw=sycl::malloc_device<uint32_t>(wb.size(),q);
    int8_t* da=sycl::malloc_device<int8_t>(ab.size(),q);
    int32_t* dc=sycl::malloc_device<int32_t>(cb.size(),q);int32_t* dy=sycl::malloc_device<int32_t>(got.size(),q);
    try {
        if(!dw||!da||!dc||!dy)throw std::bad_alloc();
        auto ew=q.memcpy(dw,wb.data(),wb.size()*4);auto ea=q.memcpy(da,ab.data(),ab.size());auto ec=q.memcpy(dc,cb.data(),cb.size()*4);
        auto e=q.submit([&](sycl::handler&h){h.depends_on(std::vector<sycl::event>{ew,ea,ec});
            h.parallel_for<MultiTokenS2S8Probe<R>>(sycl::range<1>(cases),[=](sycl::id<1>ii) SYCL_ESIMD_KERNEL {
                const size_t c=ii[0];es::simd<uint32_t,16> bo(0,4);es::simd<uint32_t,32> ao(0,1);es::simd<uint32_t,8> co(0,4);
                es::simd<uint32_t,16>b=es::gather<uint32_t,16>(dw+c*16,bo);es::simd<int8_t,R*32>a;es::simd<int32_t,R*8>z;
                #pragma unroll
                for(int r=0;r<R;++r){a.template select<32,1>(r*32)=es::gather<int8_t,32>(da+(c*R+r)*32,ao);
                    z.template select<8,1>(r*8)=es::gather<int32_t,8>(dc+(c*R+r)*8,co);}
                z=xmx::dpas<8,R,int32_t,int32_t,uint32_t,int8_t,xmx::dpas_argument_type::s2,xmx::dpas_argument_type::s8>(z,b,a);
                #pragma unroll
                for(int r=0;r<R;++r){es::simd<int32_t,8> row=z.template select<8,1>(r*8);es::scatter<int32_t,8>(dy+(c*R+r)*8,co,row);}
            });});
        e.wait_and_throw();q.memcpy(got.data(),dy,got.size()*4).wait_and_throw();
        if(got!=ref)throw std::runtime_error("multi-token s2/s8 RC"+std::to_string(R)+" probe failed: no RC1 fallback claimed");
        std::cout<<"MULTITOKEN_DPAS_PROBE PASS repeat="<<R<<" cases="<<cases<<" outputs="<<got.size()<<" (functional only; inspect ISA separately)\n";
    }catch(...){q.wait();if(dw)sycl::free(dw,q);if(da)sycl::free(da,q);if(dc)sycl::free(dc,q);if(dy)sycl::free(dy,q);throw;}
    sycl::free(dw,q);sycl::free(da,q);sycl::free(dc,q);sycl::free(dy,q);
}
void run_multitoken_s2s8_probe(sycl::queue&q){validate_dg2_device(q.get_device());probe_repeat<4>(q);probe_repeat<8>(q);}
} // namespace maple_w2
