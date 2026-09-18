// SPDX-License-Identifier: MIT
// Host-only link stubs for the existing submission DAG mock. Production SYCL uses src/expert_tiles.cpp and src/maple_w2a8_gemm.cpp.
#include "maple_w2a8_gemm.hpp"
namespace maple_w2 {
sycl::event enqueue_expert_tiles(sycl::queue&,size_t,uint32_t,uint32_t,ExpertGroupingWorkspace,ExpertTilesWorkspace,const std::vector<sycl::event>&){ return {}; }
A8Run enqueue_a8_gemm(sycl::queue&,const DeviceProblem&,A8Options,A8Workspace,const ExpertTilesWorkspace&,uint32_t,const std::vector<sycl::event>&){ return {}; }
} // namespace maple_w2
