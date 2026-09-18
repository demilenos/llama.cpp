// SPDX-License-Identifier: MIT
#include "expert_tiles_reference.hpp"
#include "w2a8_reference.hpp"
#include <array>
#include <iostream>
#include <random>
#include <set>
using namespace maple_w2;
static size_t checks=0;
static void need(bool x,const char*m){++checks;if(!x)throw std::runtime_error(m);}
static void plan_case(const std::vector<int32_t>&ids,uint32_t e,uint32_t r) {
    auto g=group_jobs_reference(ids,e);auto t=make_expert_tiles_reference(g,r);
    std::vector<int>seen(ids.size());
    size_t ti=0;
    // Emulate GPU's per-expert prefix/strided descriptor construction, independently
    // of the CPU reference's push-back traversal.
    std::vector<int32_t> ee(t.expert.size(),-1),bb(ee.size(),-1),rr(ee.size(),-1);
    for(uint32_t b=0;b<=e;++b) {
        size_t base=0;for(uint32_t j=0;j<b;++j)base+=(size_t(g.counts[j])+r-1)/r;
        const size_t tiles=(size_t(g.counts[b])+r-1)/r;
        for(size_t lane=0;lane<32;++lane)for(size_t j=lane;j<tiles;j+=32){size_t i=base+j;
            ee[i]=int32_t(b);bb[i]=g.offsets[b]+int32_t(j*r);rr[i]=std::min(int32_t(r),g.counts[b]-int32_t(j*r));}
    }
    need(ee==t.expert&&bb==t.begin&&rr==t.rows,"GPU tile-plan emulation mismatch");
    for(;ti<t.expert.size();++ti){need(t.rows[ti]>0&&t.rows[ti]<=int32_t(r),"bad rows");
        need(t.begin[ti]>=g.offsets[t.expert[ti]]&&t.begin[ti]+t.rows[ti]<=g.offsets[t.expert[ti]+1],"cross-expert tile");
        for(int32_t m=0;m<t.rows[ti];++m){int32_t j=g.sorted_to_job[t.begin[ti]+m];
            need(j>=0&&size_t(j)<ids.size(),"unsafe gather");need(grouping_key(ids[j],e)==uint32_t(t.expert[ti]),"wrong expert");++seen[j];}}
    for(auto n:seen)need(n==1,"dropped/duplicate assignment");
    need(ti<=expert_tile_capacity(ids.size(),e,r),"launch capacity insufficient");
}
template<int R>static void integer_layout_probe() {
    // Independent packed-B / row-major-A / row-major-C multiply. Rows and columns
    // have distinct patterns; broadcasting row 0 cannot accidentally pass.
    for(int test=0;test<80;++test){std::array<uint32_t,16>b{};std::array<int8_t,R*32>a{};std::array<int32_t,R*8>c{};
        for(int k=0;k<32;++k)for(int n=0;n<8;++n){int v=(test==1?-1:((k*17+n*7+test)%4-2));b[(k/16)*8+n]|=(uint32_t(v)&3u)<<(2*(k%16));}
        for(int m=0;m<R;++m)for(int k=0;k<32;++k)a[m*32+k]=int8_t(test==1?-128:((m*29+k*37+test*3)%256-128));
        for(int m=0;m<R;++m)for(int n=0;n<8;++n){c[m*8+n]=m*57-n*39+test;int z=c[m*8+n],ref=z;
            for(int k=0;k<32;++k){unsigned bits=(b[(k/16)*8+n]>>(2*(k%16)))&3u;int v=int(bits);if(v&2)v-=4;
                z+=v*int(a[m*32+k]);ref+=(test==1?-1:((k*17+n*7+test)%4-2))*int(a[m*32+k]);}
            need(z==ref,"repeat-row signed INT2 mapping mismatch");}}
}
static std::vector<uint8_t> weights(Shape s,int seed) {
    std::vector<uint8_t>w(nbytes(s));
    for(uint32_t e=0;e<s.experts;++e)for(uint32_t m=0;m<s.m;++m)for(uint32_t kb=0;kb<s.k/256;++kb){auto*b=w.data()+native_offset(s,e,m,kb);
        put16(b+64,float_to_half(float(1+(e*3+m+kb*11+seed)%31)/512));
        for(unsigned k=0;k<256;++k)set_native_code(b,k,(k*7+m*5+e*2+seed)%3);}
    return w;
}
static void arithmetic(uint32_t r,bool per) {
    const Shape s{512,16,3};const uint32_t q=13,top=2;const size_t jobs=q*top;
    std::vector<int32_t>ids(jobs);for(size_t j=0;j<jobs;++j)ids[j]=int32_t((j*7+j/3)%3);
    // Dense reference never sorts jobs; tiled emulator reads packed storage and restores slots.
    auto group=group_jobs_reference(ids,3);auto tiles=make_expert_tiles_reference(group,r);
    const size_t rows=per?jobs:q;std::vector<float>x(rows*s.k);
    for(size_t i=0;i<x.size();++i)x[i]=float(int(i%67)-33)*(1.f+float((i/32)%11))/113.f;
    for(size_t i=0;i<32;++i)x[i]=0; // zero group alongside large nonzero per-row scales
    auto a=quantize_a8(x,uint32_t(rows),s.k,32);
    for(int projection:{0,1}) {
        auto w=weights(s,projection);auto packed=repack_s2tile8(w,s);
        std::vector<float>tiled(jobs*s.m,NAN),scalar(jobs*s.m,0);
        for(size_t j=0;j<jobs;++j)for(uint32_t n=0;n<s.m;++n){size_t xr=per?j:j/top;float acc=0;
            for(uint32_t k=0;k<s.k;k+=32){auto*b=w.data()+native_offset(s,ids[j],n,k/256);int z=0;
                for(unsigned zc=0;zc<32;++zc)z+=(int(native_code(b,k%256+zc))-1)*int(a.q[xr*s.k+k+zc]);
                float dw=half_to_float(u16(b+64)),da=a.scale[(xr*s.k+k)/32];acc+=float(z)*(dw*da);}
            scalar[j*s.m+n]=acc;}
        for(size_t t=0;t<tiles.expert.size();++t)for(uint32_t rt=0;rt<s.m/8;++rt) {
            std::vector<float>acc(r*8,0.f);
            for(uint32_t kb=0;kb<s.k/256;++kb){auto*b=packed.data()+tile_offset(s,tiles.expert[t],rt,kb);
                for(uint32_t kg=0;kg<256;kg+=32){std::vector<int8_t>av(r*32,0);std::vector<float>da(r,0);
                    for(int32_t row=0;row<tiles.rows[t];++row){size_t j=group.sorted_to_job[tiles.begin[t]+row],xr=per?j:j/top,ax=xr*s.k+kb*256+kg;
                        std::copy_n(a.q.begin()+ax,32,av.begin()+row*32);da[row]=a.scale[ax/32];}
                    for(uint32_t row=0;row<r;++row)for(unsigned n=0;n<8;++n){int z=0;
                        for(unsigned k=0;k<32;++k){unsigned kk=kg+k;uint32_t word=u32(b+(kk/16)*32+n*4);unsigned v=(word>>(2*(kk%16)))&3;
                            int sv=int(v)-(v&2?4:0);z+=sv*int(av[row*32+k]);}
                        float dw=half_to_float(u16(b+512+2*n));acc[row*8+n]+=float(z)*(dw*da[row]);}}
            }
            for(uint32_t row=0;row<r;++row)for(unsigned n=0;n<8;++n) {
                if(row<uint32_t(tiles.rows[t]))tiled[size_t(group.sorted_to_job[tiles.begin[t]+row])*s.m+rt*8+n]=acc[row*8+n];
                else need(acc[row*8+n]==0,"padded row acquired nonzero activation");
            }
        }
        for(size_t i=0;i<scalar.size();++i)need(std::isfinite(tiled[i])&&tiled[i]==scalar[i],"G32/per-row dw/da/output restoration changed");
    }
}
int main()try {
    std::mt19937 rng(504);
    for(uint32_t r:{4u,8u})for(uint32_t e:{1u,3u,8u,256u})for(size_t n:std::array<size_t,18>{1,3,4,5,7,8,9,13,31,32,33,104,127,128,129,257,2048,16384}) {
        std::vector<int32_t>ids(n);for(auto&x:ids)x=int32_t(rng()%e);plan_case(ids,e,r);
        std::fill(ids.begin(),ids.end(),int32_t(e-1));plan_case(ids,e,r);
        for(size_t j=0;j<n;++j)ids[j]=j%5==0?-1:j%7==0?int32_t(e):int32_t(j%e);
        plan_case(ids,e,r);
    }
    for(uint32_t bad:{0u,1u,2u,3u,5u,16u}){bool threw=false;try{expert_tile_capacity(8,3,bad);}catch(const std::invalid_argument&){threw=true;}need(threw,"bad repeat accepted");}
    auto g=group_jobs_reference({0,1,2,0},3);g.offsets[2]++;bool threw=false;try{make_expert_tiles_reference(g,4);}catch(const std::invalid_argument&){threw=true;}need(threw,"corrupt prefix accepted");
    integer_layout_probe<4>();integer_layout_probe<8>();
    for(uint32_t r:{4u,8u})for(bool per:{false,true})arithmetic(r,per);
    std::cout<<"PASS "<<checks<<" CPU checks: compact tile plan, tails, invalid bucket, RC4/8 integer layout, G32 per-row scales, paired projections and original slots. NOT GPU execution.\n";
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
