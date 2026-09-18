#pragma once
// Host API-contract double, NOT real Vulkan declarations or Vulkan validation.
#include <cstddef>
#include <cstdint>
#include <windows.h>
using VkDeviceSize=uint64_t;using VkFlags=uint32_t;using VkBool32=uint32_t;
using VkBufferUsageFlags=uint32_t;using VkPipelineStageFlags=uint32_t;using VkAccessFlags=uint32_t;
using VkPhysicalDevice=void*;using VkDevice=void*;using VkQueue=void*;using VkDeviceMemory=void*;using VkBuffer=void*;using VkCommandPool=void*;using VkCommandBuffer=void*;using VkFence=void*;
using VkSemaphore=void*;using VkInstance=void*;
#define VK_NULL_HANDLE nullptr
constexpr VkBool32 VK_TRUE=1;constexpr size_t VK_LUID_SIZE=8;
enum VkResult{VK_SUCCESS=0,VK_NOT_READY=1,VK_TIMEOUT=2,VK_ERROR_UNKNOWN=-1};
enum VkStructureType {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES=1,VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES,VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,VK_STRUCTURE_TYPE_SUBMIT_INFO,VK_STRUCTURE_TYPE_APPLICATION_INFO,VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
constexpr uint32_t VK_BUFFER_USAGE_STORAGE_BUFFER_BIT=1,VK_BUFFER_USAGE_TRANSFER_SRC_BIT=2,VK_BUFFER_USAGE_TRANSFER_DST_BIT=4;
constexpr uint32_t VK_QUEUE_COMPUTE_BIT=1,VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT=1,VK_COMMAND_BUFFER_LEVEL_PRIMARY=0,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT=1;
constexpr uint32_t VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT=1,VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT=1,VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT=2;
constexpr uint32_t VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT=1,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT=2,VK_MEMORY_PROPERTY_HOST_COHERENT_BIT=4,VK_SHARING_MODE_EXCLUSIVE=0;
constexpr uint32_t VK_PIPELINE_STAGE_ALL_COMMANDS_BIT=1,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT=2,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT=4,VK_PIPELINE_STAGE_TRANSFER_BIT=8,VK_PIPELINE_STAGE_HOST_BIT=16;
constexpr uint32_t VK_ACCESS_MEMORY_READ_BIT=1,VK_ACCESS_MEMORY_WRITE_BIT=2,VK_ACCESS_TRANSFER_READ_BIT=4,VK_ACCESS_TRANSFER_WRITE_BIT=8,VK_ACCESS_HOST_READ_BIT=16;
constexpr uint32_t VK_QUEUE_FAMILY_EXTERNAL=~1u,VK_QUEUE_FAMILY_IGNORED=~0u;
constexpr uint32_t VK_API_VERSION_1_1=(1u<<22)|(1u<<12);
constexpr uint64_t VK_WHOLE_SIZE=~uint64_t(0);
#define VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME "VK_KHR_external_memory_win32"
struct VkPhysicalDeviceIDProperties{VkStructureType sType;void*pNext=nullptr;uint8_t deviceUUID[16]{},driverUUID[16]{},deviceLUID[8]{};uint32_t deviceNodeMask=0;VkBool32 deviceLUIDValid=0;};
struct VkPhysicalDeviceProperties{uint32_t apiVersion=0,driverVersion=0,vendorID=0,deviceID=0;char deviceName[256]{};};
struct VkPhysicalDeviceProperties2{VkStructureType sType;void*pNext=nullptr;VkPhysicalDeviceProperties properties;};
struct VkQueueFamilyProperties{uint32_t queueFlags=0,queueCount=1;};
struct VkCommandPoolCreateInfo{VkStructureType sType;const void*pNext=nullptr;uint32_t flags=0,queueFamilyIndex=0;};
struct VkCommandBufferAllocateInfo{VkStructureType sType;const void*pNext=nullptr;VkCommandPool commandPool=nullptr;uint32_t level=0,commandBufferCount=0;};
struct VkFenceCreateInfo{VkStructureType sType;const void*pNext=nullptr;uint32_t flags=0;};
struct VkMemoryGetWin32HandleInfoKHR{VkStructureType sType;const void*pNext=nullptr;VkDeviceMemory memory=nullptr;uint32_t handleType=0;};
struct VkPhysicalDeviceExternalBufferInfo{VkStructureType sType;const void*pNext=nullptr;uint32_t flags=0;VkBufferUsageFlags usage=0;uint32_t handleType=0;};
struct VkExternalMemoryProperties{uint32_t externalMemoryFeatures=0,exportFromImportedHandleTypes=0,compatibleHandleTypes=0;};
struct VkExternalBufferProperties{VkStructureType sType;void*pNext=nullptr;VkExternalMemoryProperties externalMemoryProperties;};
struct VkExternalMemoryBufferCreateInfo{VkStructureType sType;const void*pNext=nullptr;uint32_t handleTypes=0;};
struct VkBufferCreateInfo{VkStructureType sType;const void*pNext=nullptr;uint32_t flags=0;VkDeviceSize size=0;VkBufferUsageFlags usage=0;uint32_t sharingMode=0,queueFamilyIndexCount=0;const uint32_t*pQueueFamilyIndices=nullptr;};
struct VkMemoryRequirements{VkDeviceSize size=0,alignment=64;uint32_t memoryTypeBits=1;};
struct VkMemoryType{uint32_t propertyFlags=0,heapIndex=0;};
struct VkPhysicalDeviceMemoryProperties{uint32_t memoryTypeCount=1;VkMemoryType memoryTypes[32];};
struct VkMemoryDedicatedAllocateInfo{VkStructureType sType;const void*pNext=nullptr;void*image=nullptr;VkBuffer buffer=nullptr;};
struct VkExportMemoryAllocateInfo{VkStructureType sType;const void*pNext=nullptr;uint32_t handleTypes=0;};
struct VkMemoryAllocateInfo{VkStructureType sType;const void*pNext=nullptr;VkDeviceSize allocationSize=0;uint32_t memoryTypeIndex=0;};
struct VkCommandBufferBeginInfo{VkStructureType sType;const void*pNext=nullptr;uint32_t flags=0;const void*pInheritanceInfo=nullptr;};
struct VkBufferMemoryBarrier{VkStructureType sType;const void*pNext=nullptr;VkAccessFlags srcAccessMask=0,dstAccessMask=0;uint32_t srcQueueFamilyIndex=0,dstQueueFamilyIndex=0;VkBuffer buffer=nullptr;VkDeviceSize offset=0,size=0;};
struct VkSubmitInfo{VkStructureType sType;const void*pNext=nullptr;uint32_t waitSemaphoreCount=0;const VkSemaphore*pWaitSemaphores=nullptr;const VkPipelineStageFlags*pWaitDstStageMask=nullptr;uint32_t commandBufferCount=0;const VkCommandBuffer*pCommandBuffers=nullptr;uint32_t signalSemaphoreCount=0;const VkSemaphore*pSignalSemaphores=nullptr;};
struct VkApplicationInfo{VkStructureType sType;const void*pNext=nullptr;const char*pApplicationName=nullptr;uint32_t applicationVersion=0;const char*pEngineName=nullptr;uint32_t engineVersion=0,apiVersion=0;};
struct VkInstanceCreateInfo{VkStructureType sType;const void*pNext=nullptr;uint32_t flags=0;const VkApplicationInfo*pApplicationInfo=nullptr;uint32_t enabledLayerCount=0;const char*const*ppEnabledLayerNames=nullptr;uint32_t enabledExtensionCount=0;const char*const*ppEnabledExtensionNames=nullptr;};
struct VkDeviceQueueCreateInfo{VkStructureType sType;const void*pNext=nullptr;uint32_t flags=0,queueFamilyIndex=0,queueCount=0;const float*pQueuePriorities=nullptr;};
struct VkDeviceCreateInfo{VkStructureType sType;const void*pNext=nullptr;uint32_t flags=0,queueCreateInfoCount=0;const VkDeviceQueueCreateInfo*pQueueCreateInfos=nullptr;uint32_t enabledLayerCount=0;const char*const*ppEnabledLayerNames=nullptr;uint32_t enabledExtensionCount=0;const char*const*ppEnabledExtensionNames=nullptr;const void*pEnabledFeatures=nullptr;};
struct VkBufferCopy{VkDeviceSize srcOffset=0,dstOffset=0,size=0;};
struct VkMappedMemoryRange{VkStructureType sType;const void*pNext=nullptr;VkDeviceMemory memory=nullptr;VkDeviceSize offset=0,size=0;};
using PFN_vkVoidFunction=void(*)();using PFN_vkGetMemoryWin32HandleKHR=VkResult(*)(VkDevice,const VkMemoryGetWin32HandleInfoKHR*,HANDLE*);
extern "C" {
void vkGetPhysicalDeviceProperties2(VkPhysicalDevice,VkPhysicalDeviceProperties2*);
void vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice,uint32_t*,VkQueueFamilyProperties*);
PFN_vkVoidFunction vkGetDeviceProcAddr(VkDevice,const char*);
VkResult vkCreateCommandPool(VkDevice,const VkCommandPoolCreateInfo*,const void*,VkCommandPool*);
VkResult vkAllocateCommandBuffers(VkDevice,const VkCommandBufferAllocateInfo*,VkCommandBuffer*);
VkResult vkCreateFence(VkDevice,const VkFenceCreateInfo*,const void*,VkFence*);
void vkDestroyFence(VkDevice,VkFence,const void*);void vkDestroyCommandPool(VkDevice,VkCommandPool,const void*);
VkResult vkQueueWaitIdle(VkQueue);
void vkDestroyBuffer(VkDevice,VkBuffer,const void*);void vkFreeMemory(VkDevice,VkDeviceMemory,const void*);
void vkGetPhysicalDeviceExternalBufferProperties(VkPhysicalDevice,const VkPhysicalDeviceExternalBufferInfo*,VkExternalBufferProperties*);
VkResult vkCreateBuffer(VkDevice,const VkBufferCreateInfo*,const void*,VkBuffer*);
void vkGetBufferMemoryRequirements(VkDevice,VkBuffer,VkMemoryRequirements*);
void vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice,VkPhysicalDeviceMemoryProperties*);
VkResult vkAllocateMemory(VkDevice,const VkMemoryAllocateInfo*,const void*,VkDeviceMemory*);
VkResult vkBindBufferMemory(VkDevice,VkBuffer,VkDeviceMemory,VkDeviceSize);
VkResult vkResetCommandBuffer(VkCommandBuffer,uint32_t);
VkResult vkBeginCommandBuffer(VkCommandBuffer,const VkCommandBufferBeginInfo*);
void vkCmdPipelineBarrier(VkCommandBuffer,VkPipelineStageFlags,VkPipelineStageFlags,uint32_t,uint32_t,const void*,uint32_t,const VkBufferMemoryBarrier*,uint32_t,const void*);
VkResult vkEndCommandBuffer(VkCommandBuffer);VkResult vkResetFences(VkDevice,uint32_t,const VkFence*);
VkResult vkQueueSubmit(VkQueue,uint32_t,const VkSubmitInfo*,VkFence);
VkResult vkWaitForFences(VkDevice,uint32_t,const VkFence*,VkBool32,uint64_t);
VkResult vkCreateInstance(const VkInstanceCreateInfo*,const void*,VkInstance*);void vkDestroyInstance(VkInstance,const void*);
VkResult vkEnumeratePhysicalDevices(VkInstance,uint32_t*,VkPhysicalDevice*);
void vkGetPhysicalDeviceProperties(VkPhysicalDevice,VkPhysicalDeviceProperties*);
VkResult vkCreateDevice(VkPhysicalDevice,const VkDeviceCreateInfo*,const void*,VkDevice*);void vkDestroyDevice(VkDevice,const void*);VkResult vkDeviceWaitIdle(VkDevice);
void vkGetDeviceQueue(VkDevice,uint32_t,uint32_t,VkQueue*);
void vkCmdFillBuffer(VkCommandBuffer,VkBuffer,VkDeviceSize,VkDeviceSize,uint32_t);
void vkCmdCopyBuffer(VkCommandBuffer,VkBuffer,VkBuffer,uint32_t,const VkBufferCopy*);
VkResult vkMapMemory(VkDevice,VkDeviceMemory,VkDeviceSize,VkDeviceSize,uint32_t,void**);void vkUnmapMemory(VkDevice,VkDeviceMemory);
VkResult vkInvalidateMappedMemoryRanges(VkDevice,uint32_t,const VkMappedMemoryRange*);
}
