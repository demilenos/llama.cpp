// SPDX-License-Identifier: MIT
#include "maple_w2a8_gemm.hpp"
namespace maple_w2 {
class ExpertTilePlan;
sycl::event enqueue_expert_tiles(sycl::queue&q,size_t jobs,uint32_t experts,uint32_t repeat,
    ExpertGroupingWorkspace g,ExpertTilesWorkspace w,const std::vector<sycl::event>&deps) {
    const size_t bound=expert_tile_capacity(jobs,experts,repeat);
    if(!g.counts||!g.offsets||!g.sorted_to_job||g.job_capacity<jobs||g.expert_capacity<experts||
       !w.expert||!w.begin||!w.rows||!w.count||!w.invalid||w.capacity<bound||w.expert_capacity<experts)
        throw std::invalid_argument("missing/undersized multi-token tile workspace");
    // One group/expert, uniform prefix computed once/lane0 then broadcast.
    // Planner is charged to GEMM arm only; v0.4 count/prefix/scatter are unchanged.
    constexpr size_t lanes=32;
    return q.submit([&](sycl::handler&h) {
        h.depends_on(deps);
        h.parallel_for<ExpertTilePlan>(sycl::nd_range<1>{sycl::range<1>((experts+1)*lanes),sycl::range<1>(lanes)},
            [=](sycl::nd_item<1> it) {
                const uint32_t e=uint32_t(it.get_group_linear_id());const size_t lane=it.get_local_linear_id();
                int32_t base=0;int bad=0;
                if(lane==0) {
                    // Check the complete prefix, not just the current count. This also
                    // prevents malformed external counts from becoming unsafe descriptors.
                    int64_t total=0,tilebase=0;
                    for(uint32_t b=0;b<=experts;++b) {
                        const int32_t n=g.counts[b];
                        if(n<0||int64_t(g.offsets[b])!=total)bad=1;
                        total+=int64_t(n);
                        if(b<e && n>=0)tilebase+=(int64_t(n)+repeat-1)/repeat;
                    }
                    if(total!=int64_t(jobs)||int64_t(g.offsets[experts+1])!=total||tilebase>int64_t(w.capacity))bad=1;
                    base=bad?0:int32_t(tilebase);w.invalid[e]=bad;
                    if(e==experts) {
                        const int64_t end=tilebase+(int64_t(g.counts[e])+repeat-1)/repeat;
                        *w.count=bad||end<0||end>int64_t(w.capacity)?0:int32_t(end);
                    }
                }
                base=sycl::group_broadcast(it.get_group(),base,0);
                bad=sycl::group_broadcast(it.get_group(),bad,0);
                if(bad)return;
                const int32_t n=g.counts[e];const uint32_t nt=(uint32_t(n)+repeat-1)/repeat;
                for(uint32_t t=uint32_t(lane);t<nt;t+=lanes) {
                    const size_t i=size_t(base)+t;
                    if(i>=w.capacity){if(lane==0)w.invalid[e]=1;return;}
                    w.expert[i]=int32_t(e);w.begin[i]=g.offsets[e]+int32_t(t*repeat);
                    w.rows[i]=sycl::min(int32_t(repeat),n-int32_t(t*repeat));
                }
            });
    });
}
} // namespace maple_w2
