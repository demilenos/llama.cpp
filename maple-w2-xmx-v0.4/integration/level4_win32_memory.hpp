// SPDX-License-Identifier: MIT
#pragma once
// Optional loader-time Windows external-memory import. Not included by default.
// Link ze_loader.lib; initialize the caller's Vulkan allocation with
// VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT export support at creation.
#if !defined(_WIN32)
#error "level4_win32_memory.hpp requires Windows. Use existing platform importer elsewhere."
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include "level4_bridge.hpp"
namespace maple_w2::level4 {
// Queue copy retains the context. Destructor drains only at ALLOCATION TEARDOWN,
// never per matmul or island. Model/workspace caches must retain this owner.
class Win32ImportedAllocation {
    sycl::queue queue_;ze_context_handle_t context_=nullptr;void* ptr_=nullptr;size_t bytes_=0;
    static void check(ze_result_t result,const char*op) {
        if(result!=ZE_RESULT_SUCCESS)throw std::runtime_error(std::string(op)+" failed: "+std::to_string(uint32_t(result)));
    }
    Win32ImportedAllocation(sycl::queue q):queue_(q){}
public:
    Win32ImportedAllocation(const Win32ImportedAllocation&)=delete;
    ~Win32ImportedAllocation(){if(ptr_){try{queue_.wait_and_throw();}catch(...){return;} (void)zeMemFree(context_,ptr_);}}
    static std::shared_ptr<Win32ImportedAllocation> import(sycl::queue q,HANDLE borrowed,size_t bytes,
                                                          bool physical_device_match_confirmed) {
        if(!physical_device_match_confirmed||!bytes||!borrowed||borrowed==INVALID_HANDLE_VALUE)
            throw std::invalid_argument("verify Vulkan/LevelZero physical device and exportable allocation before import");
        if(q.get_device().get_backend()!=sycl::backend::ext_oneapi_level_zero)
            throw std::invalid_argument("Level4 import needs a LevelZero queue");
        auto owner=std::shared_ptr<Win32ImportedAllocation>(new Win32ImportedAllocation(q));
        owner->context_=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        const auto dev=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
        ze_device_external_memory_properties_t props{};props.stype=ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES;
        check(zeDeviceGetExternalMemoryProperties(dev,&props),"zeDeviceGetExternalMemoryProperties");
        if(!(props.memoryAllocationImportTypes&ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32))
            throw std::runtime_error("driver does not expose OPAQUE_WIN32 memory import");
        HANDLE duplicate=nullptr;
        if(!DuplicateHandle(GetCurrentProcess(),borrowed,GetCurrentProcess(),&duplicate,0,FALSE,DUPLICATE_SAME_ACCESS))
            throw std::runtime_error("DuplicateHandle failed");
        ze_external_memory_import_win32_handle_t ext{};ext.stype=ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32;
        ext.flags=ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32;ext.handle=duplicate;
        ze_device_mem_alloc_desc_t desc{};desc.stype=ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;desc.pNext=&ext;
        const auto result=zeMemAllocDevice(owner->context_,&desc,bytes,64,dev,&owner->ptr_);
        CloseHandle(duplicate);check(result,"zeMemAllocDevice(external import)");owner->bytes_=bytes;return owner;
    }
    ImportedAllocationView view(uint64_t generation,const std::shared_ptr<Win32ImportedAllocation>&self) {
        if(self.get()!=this)throw std::invalid_argument("wrong import owner");return {ptr_,bytes_,generation,self};
    }
};
} // namespace
