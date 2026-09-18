// SPDX-License-Identifier: MIT
#pragma once
// Windows x64: Vulkan 1.1+ with VK_KHR_external_memory_win32 and a matching
// Level Zero device. Shared memory is NOT vkGetBufferDeviceAddress cast to USM.
#if !defined(_WIN32) && !defined(MAPLE_ZC_NATIVE_MOCK)
#error "Level4 zero-copy native transport currently targets Windows x64"
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <windows.h>
#include <vulkan/vulkan.h>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include "maple_level4.hpp"
#include "level4_zerocopy_contract.hpp"
#include <memory>
#include <mutex>
#include <vector>
namespace maple_w2::level4::zc {
struct Counters {
    uint64_t allocations=0,exports=0,imports=0,cache_hits=0,islands=0;
    uint64_t inbound_submits=0,outbound_submits=0;
    uint64_t activation_h2d_bytes=0,activation_d2h_bytes=0,boundary_d2d_bytes=0;
};
struct Timing {double inbound_host_us=0,sycl_host_us=0,acquire_submit_us=0,total_host_us=0;};
struct Result {Run run;Timing timing;};
struct State;struct ImportedMemory;class Context;
class Buffer {
    friend class Context;
    std::shared_ptr<ImportedMemory> memory_;
    std::shared_ptr<void> buffer_owner_;
    VkBuffer buffer_=VK_NULL_HANDLE;BufferExtent extent_;
    Buffer(std::shared_ptr<ImportedMemory>,std::shared_ptr<void>,VkBuffer,BufferExtent);
public:
    VkBuffer vk_buffer()const{return buffer_;}
    size_t bytes()const{return extent_.buffer_bytes;}
    uint64_t generation()const{return extent_.generation;}
    void* data(size_t tensor_offset,size_t bytes,size_t alignment=1)const;
    template<class T>T*at(size_t tensor_offset,size_t count)const {
        if(count>SIZE_MAX/sizeof(T))throw std::overflow_error("tensor count overflow");
        return static_cast<T*>(data(tensor_offset,count*sizeof(T),alignof(T)));
    }
};
// Adoption is for an allocation that was created exportable originally. Adding
// export flags after allocation is impossible. source_owner must retain BOTH the
// VkBuffer and the VkDeviceMemory until this view/import is retired.
struct ExportedBuffer {
    VkBuffer buffer=VK_NULL_HANDLE;VkDeviceMemory memory=VK_NULL_HANDLE;
    BufferExtent extent;
    std::shared_ptr<void> source_owner;
};
// Native SYCL-only memory. A lifetime owner is mandatory. Imported regions may
// not be disguised as private allocations: prepare() checks the import registry.
struct PrivateRegion {void* base=nullptr;size_t bytes=0;std::shared_ptr<void> owner;};
class Prepared {
    friend class Context;
    std::shared_ptr<State> state_;Weights weights_;Bindings bindings_;Options options_;Workspace workspace_;
    std::vector<std::shared_ptr<Buffer>> shared_;std::vector<PrivateRegion> private_;
    GraphSemantics semantics_;
};
class Context : public std::enable_shared_from_this<Context> {
    std::shared_ptr<State> state_;
    struct Transport;std::unique_ptr<Transport> transport_;
    explicit Context(std::shared_ptr<State>);
    std::shared_ptr<Buffer> adopt_unlocked(const ExportedBuffer&);
public:
    ~Context();Context(const Context&)=delete;Context&operator=(const Context&)=delete;
    // Every submission touching these resources must use this exact VkQueue and
    // the same queue_mutex. Producer command buffers MUST be submitted first.
    // device_owner is retained by all buffers, imports and prepared calls.
    static std::shared_ptr<Context> create(sycl::queue queue,VkPhysicalDevice physical,VkDevice device,
        VkQueue vk_queue,uint32_t family,std::shared_ptr<std::recursive_mutex> queue_mutex,
        std::shared_ptr<void> device_owner,uint64_t fence_timeout_ns=60'000'000'000ull);
    std::shared_ptr<Buffer> create_buffer(size_t bytes,
        VkBufferUsageFlags usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::shared_ptr<Buffer> adopt(const ExportedBuffer&);
    Prepared prepare(const Weights&,const Bindings&,Options,const Workspace&,GraphSemantics,
        std::vector<std::shared_ptr<Buffer>> shared,std::vector<PrivateRegion> private_regions);
    // No imports/allocations/H2D/D2H. Strict contiguous boundary views write
    // directly into exported Vulkan storage. One inbound fence + one SYCL tail
    // wait; acquire is queued (not host waited) on the SAME Vulkan queue.
    // The returned gpu event is SYCL completion, NOT downstream Vulkan completion.
    Result execute(const Prepared&);
    // Completes queued consumers before resize/teardown. Never called per op.
    void drain();Counters counters()const;bool poisoned()const;
};
} // namespace
