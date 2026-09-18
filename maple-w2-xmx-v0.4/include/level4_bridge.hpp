// SPDX-License-Identifier: MIT
#pragma once
#include "maple_level4.hpp"
#include <functional>
#include <memory>
namespace maple_w2::level4 {
struct BridgeCounters {uint64_t islands=0,inbound=0,outbound=0,bytes_h2d=0,bytes_d2h=0;};
// Callbacks integrate the user's EXISTING Vulkan shared allocator / submission
// implementation. They must be real queue ownership barriers + completion waits,
// not just a pointer cast or a SYCL event made from a Vulkan semaphore.
struct VulkanHandoff {
    std::function<void()> release_and_wait;  // finish incoming Vulkan writes; release EXTERNAL ownership as required
    std::function<void()> acquire_after_sycl; // record external -> Vulkan acquire before any consumer
    std::function<void()> poison; // inhibit fallback/consumer on a partially submitted island
    // Caller also retains this lease through DOWNSTREAM Vulkan completion;
    // an acquire callback may only record a barrier, not submit/complete it.
    std::shared_ptr<void> allocation_lease; // keeps views alive through this SYCL completion
    GraphSemantics semantics;
};
// Initial safe HOST-FENCE mode: exactly ONE wait at each outer boundary,
// zero inter-stage host waits. Not claimed to be GPU semaphore interop.
// Import/cache allocations ONCE outside this function. No activation CPU copies.
inline Run execute_host_fenced(sycl::queue&q,const Weights&wt,const Bindings&b,const Options&o,
                               const Workspace&w,const VulkanHandoff&h,BridgeCounters&stats) {
    validate(wt,b,o,w);
    if(const char*why=reject_graph(w.plan.config,h.semantics))throw std::invalid_argument(why);
    if(!h.release_and_wait||!h.acquire_after_sycl||!h.poison||!h.allocation_lease)
        throw std::invalid_argument("Level4 needs verified memory lease and all ownership handoff callbacks");
    if(o.diagnostic_stage_waits||o.moe.diagnostic_host_waits)throw std::invalid_argument("diagnostic waits forbidden in Level4 bridge");
    // All predictable validation has completed. No output may be consumed before
    // successful acquire; a submission/runtime exception MUST poison the region.
    auto keepalive=h.allocation_lease;bool released=false;
    try {
        released=true;h.release_and_wait();++stats.inbound;
        auto run=enqueue(q,wt,b,o,w);run.done.wait_and_throw();
        h.acquire_after_sycl();++stats.outbound;++stats.islands;return run;
    } catch(...) {
        const auto original=std::current_exception();
        if(released){try{q.wait_and_throw();}catch(...){}try{h.poison();}catch(...){} }
        std::rethrow_exception(original);
    }
}
// Descriptor for an already-imported physical allocation. gpu_ptr MUST be the
// result of a Level Zero import in the SYCL queue's context, NOT VkDeviceAddress.
struct ImportedAllocationView {
    void* gpu_ptr=nullptr;size_t bytes=0;uint64_t generation=0;
    std::shared_ptr<void> lease;
    template<class T>T*at(size_t offset_bytes,size_t count,uint64_t expected_generation)const {
        const size_t n=checked_mul(count,sizeof(T));
        if(!gpu_ptr||!lease||generation!=expected_generation||offset_bytes%alignof(T)||
           offset_bytes>bytes||n>bytes-offset_bytes)
            throw std::invalid_argument("invalid/stale/out-of-bounds imported buffer view");
        return reinterpret_cast<T*>(static_cast<uint8_t*>(gpu_ptr)+offset_bytes);
    }
};
} // namespace
