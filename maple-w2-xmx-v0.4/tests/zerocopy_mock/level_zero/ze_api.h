#pragma once
// API-surface test double; numeric enums are intentionally NOT a SDK ABI.
#include <cstddef>
#include <cstdint>
using ze_context_handle_t=void*;using ze_device_handle_t=void*;
enum ze_result_t{ZE_RESULT_SUCCESS=0,ZE_RESULT_ERROR_UNSUPPORTED_FEATURE=1};
enum ze_structure_type_t {ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES=1,ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES,ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES,ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES,ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32,ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
enum ze_memory_type_t{ZE_MEMORY_TYPE_UNKNOWN,ZE_MEMORY_TYPE_DEVICE};
constexpr uint32_t ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32=4;constexpr size_t ZE_MAX_DEVICE_LUID_SIZE_EXT=8;
struct ze_memory_allocation_properties_t{ze_structure_type_t stype{};void*pNext=nullptr;ze_memory_type_t type{};};
struct ze_device_luid_ext_t{uint8_t id[8]{};};
struct ze_device_luid_ext_properties_t{ze_structure_type_t stype{};void*pNext=nullptr;ze_device_luid_ext_t luid;uint32_t nodeMask=0;};
struct ze_device_properties_t{ze_structure_type_t stype{};void*pNext=nullptr;uint32_t vendorId=0,deviceId=0;};
struct ze_device_external_memory_properties_t{ze_structure_type_t stype{};void*pNext=nullptr;uint32_t memoryAllocationImportTypes=0;};
struct ze_external_memory_import_win32_handle_t{ze_structure_type_t stype{};const void*pNext=nullptr;uint32_t flags=0;void*handle=nullptr;const void*name=nullptr;};
struct ze_device_mem_alloc_desc_t{ze_structure_type_t stype{};const void*pNext=nullptr;uint32_t flags=0,ordinal=0;};
extern "C" {
ze_result_t zeMemGetAllocProperties(ze_context_handle_t,const void*,ze_memory_allocation_properties_t*,ze_device_handle_t*);
ze_result_t zeMemGetAddressRange(ze_context_handle_t,const void*,void**,size_t*);
ze_result_t zeDeviceGetProperties(ze_device_handle_t,ze_device_properties_t*);
ze_result_t zeDeviceGetExternalMemoryProperties(ze_device_handle_t,ze_device_external_memory_properties_t*);
ze_result_t zeMemAllocDevice(ze_context_handle_t,const ze_device_mem_alloc_desc_t*,size_t,size_t,ze_device_handle_t,void**);
ze_result_t zeMemFree(ze_context_handle_t,void*);
}
