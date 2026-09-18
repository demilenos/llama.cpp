// SPDX-License-Identifier: MIT
#pragma once
// No Vulkan/SYCL dependency: offset, identity and ownership contracts are tested
// independently of a GPU. Native transport lives in integration/*win32*.
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
namespace maple_w2::level4::zc {
inline size_t checked_add(size_t a,size_t b) {
    if(b>std::numeric_limits<size_t>::max()-a)throw std::overflow_error("zero-copy byte offset overflow");
    return a+b;
}
struct BufferExtent {
    size_t allocation_bytes=0,bind_offset=0,buffer_bytes=0;
    uint64_t generation=0;
    void validate()const {
        if(!allocation_bytes||!buffer_bytes||!generation||bind_offset>allocation_bytes||
           buffer_bytes>allocation_bytes-bind_offset)throw std::invalid_argument("invalid exported buffer extent");
    }
    size_t resolve(size_t tensor_offset,size_t bytes,size_t alignment,uint64_t expected)const {
        validate();
        if(!bytes||!alignment||(alignment&(alignment-1))||expected!=generation||
           tensor_offset>buffer_bytes||bytes>buffer_bytes-tensor_offset)
            throw std::invalid_argument("stale or out-of-range zero-copy tensor");
        auto offset=checked_add(bind_offset,tensor_offset);
        if(offset%alignment)throw std::invalid_argument("misaligned zero-copy tensor");
        return offset;
    }
};
inline bool covers(const void*base,size_t capacity,const void*p,size_t n) {
    const auto a=reinterpret_cast<uintptr_t>(base),b=reinterpret_cast<uintptr_t>(p);
    return base&&p&&n&&capacity<=UINTPTR_MAX-a&&b>=a&&b-a<=capacity&&n<=capacity-(b-a);
}
struct DeviceIdentity {
    uint32_t vendor=0,device=0,node_mask=0;
    std::array<uint8_t,8> luid{};bool luid_valid=false;
};
inline void require_same_device(const DeviceIdentity&a,const DeviceIdentity&b) {
    bool nonzero=false;for(auto x:a.luid)nonzero|=x!=0;
    if(a.vendor!=0x8086||b.vendor!=a.vendor||a.device!=b.device||!a.luid_valid||!b.luid_valid||!nonzero||
       a.luid!=b.luid||a.node_mask!=b.node_mask||!a.node_mask||(a.node_mask&(a.node_mask-1)))
        throw std::invalid_argument("Vulkan/LevelZero LUID, node mask or PCI identity mismatch");
}
enum class Ownership { vulkan, release_submitted, external, acquire_submitted, poisoned };
class OwnershipState {
    Ownership value_=Ownership::vulkan;
public:
    Ownership value()const{return value_;}
    void release(){if(value_!=Ownership::vulkan&&value_!=Ownership::acquire_submitted)throw std::logic_error("ownership release out of order");value_=Ownership::release_submitted;}
    void released(){if(value_!=Ownership::release_submitted)throw std::logic_error("missing Vulkan completion fence");value_=Ownership::external;}
    void acquire(){if(value_!=Ownership::external)throw std::logic_error("missing SYCL completion");value_=Ownership::acquire_submitted;}
    void poison()noexcept{value_=Ownership::poisoned;}
    void check()const{if(value_==Ownership::poisoned)throw std::runtime_error("zero-copy context poisoned; do not consume output or fallback in-place");}
};
} // namespace
