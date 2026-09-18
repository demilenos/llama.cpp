// SPDX-License-Identifier: MIT
#pragma once
#include "maple_w2a8.hpp"
#include "expert_grouping.hpp"
#include "expert_tiles_reference.hpp"
namespace maple_w2 {
// Persistent, context-compatible USM. Descriptor entries [0,*count) are valid.
// At most capacity tiles are launched, with a device-side count guard. No host readback.
struct ExpertTilesWorkspace {
    int32_t *expert=nullptr,*begin=nullptr,*rows=nullptr,*count=nullptr;
    int32_t *invalid=nullptr; // E+1 entries, overwritten by planner each call
    size_t capacity=0;uint32_t expert_capacity=0;
};
sycl::event enqueue_expert_tiles(sycl::queue&,size_t jobs,uint32_t experts,uint32_t repeat,
    ExpertGroupingWorkspace,ExpertTilesWorkspace,const std::vector<sycl::event>& dependencies={});
// G32, s2tile8, split=1 only. Quantizer is the unchanged v0.4 quantizer.
// staged quantizes this call's x; prequantized consumes this call's dependency producer.
A8Run enqueue_a8_gemm(sycl::queue&,const DeviceProblem&,A8Options,A8Workspace,
    const ExpertTilesWorkspace&,uint32_t repeat,const std::vector<sycl::event>& dependencies={});
void run_multitoken_s2s8_probe(sycl::queue&); // RC4 and RC8, exact signed integer outputs
} // namespace maple_w2
