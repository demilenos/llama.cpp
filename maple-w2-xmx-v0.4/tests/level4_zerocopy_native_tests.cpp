// SPDX-License-Identifier: MIT
// Real transport and Level4 source compiled against host API doubles. These
// tests do NOT prove Win32 SDK compilation, residency, cache coherency or GPU sync.
#include "level4_zerocopy_win32.hpp"
#include "level4_cpu_stubs.hpp"
#include "ggml_level4_zerocopy.hpp"
#include <iostream>
#include <map>
#include <set>
#include <new>
using namespace maple_w2;namespace l4=maple_w2::level4;namespace zc=l4::zc;
namespace mock {
size_t checks=0,close_count=0,import_count=0,free_count=0;bool bad_luid=false,exportable=true,import_fail=false,submit_fail=false;
void need(bool x,const char*s){++checks;if(!x)throw std::runtime_error(s);}
struct Memory {void*p;size_t n;bool imported=false;explicit Memory(size_t size):p(::operator new(size,std::align_val_t(64))),n(size){std::memset(p,0,n);}~Memory(){::operator delete(p,std::align_val_t(64));}};
struct Buffer {size_t n=0;Memory*memory=nullptr;size_t offset=0;bool external=false;};
struct Command {std::vector<VkBufferMemoryBarrier>barriers;};
struct Pool {std::vector<Command*>commands;~Pool(){for(auto*p:commands)delete p;}};
std::map<void*,Memory*>ranges;std::set<void*>handles;
Memory* find(const void*p){for(const auto&v:ranges)if(zc::covers(v.first,v.second->n,p,1))return v.second;return nullptr;}
struct Private {Memory m;explicit Private(size_t n):m(n){ranges[m.p]=&m;}~Private(){ranges.erase(m.p);}};
VkResult export_nt(VkDevice,const VkMemoryGetWin32HandleInfoKHR*p,HANDLE*out){need(p->handleType==VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,"wrong handle type");auto h=new Memory*(static_cast<Memory*>(p->memory));handles.insert(h);*out=h;return VK_SUCCESS;}
}
extern "C" {
BOOL CloseHandle(HANDLE h){mock::need(mock::handles.erase(h)==1,"handle close mismatch");delete static_cast<mock::Memory**>(h);++mock::close_count;return 1;}
void vkGetPhysicalDeviceProperties2(VkPhysicalDevice,VkPhysicalDeviceProperties2*p){p->properties.apiVersion=VK_API_VERSION_1_1;p->properties.vendorID=0x8086;p->properties.deviceID=0x56a1;auto id=static_cast<VkPhysicalDeviceIDProperties*>(p->pNext);id->deviceLUIDValid=1;id->deviceNodeMask=1;id->deviceLUID[0]=7;}
void vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice,uint32_t*n,VkQueueFamilyProperties*p){*n=1;if(p)p->queueFlags=VK_QUEUE_COMPUTE_BIT;}
PFN_vkVoidFunction vkGetDeviceProcAddr(VkDevice,const char*){return reinterpret_cast<PFN_vkVoidFunction>(mock::export_nt);}
VkResult vkCreateCommandPool(VkDevice,const VkCommandPoolCreateInfo*,const void*,VkCommandPool*p){*p=new mock::Pool;return VK_SUCCESS;}
VkResult vkAllocateCommandBuffers(VkDevice,const VkCommandBufferAllocateInfo*p,VkCommandBuffer*out){auto pool=static_cast<mock::Pool*>(p->commandPool);for(uint32_t i=0;i<p->commandBufferCount;++i){auto cmd=new mock::Command;pool->commands.push_back(cmd);out[i]=cmd;}return VK_SUCCESS;}
VkResult vkCreateFence(VkDevice,const VkFenceCreateInfo*,const void*,VkFence*p){*p=new bool(false);return VK_SUCCESS;}
void vkDestroyFence(VkDevice,VkFence f,const void*){delete static_cast<bool*>(f);}void vkDestroyCommandPool(VkDevice,VkCommandPool p,const void*){delete static_cast<mock::Pool*>(p);}
VkResult vkQueueWaitIdle(VkQueue){return VK_SUCCESS;}
void vkDestroyBuffer(VkDevice,VkBuffer b,const void*){delete static_cast<mock::Buffer*>(b);}
void vkFreeMemory(VkDevice,VkDeviceMemory m,const void*){auto p=static_cast<mock::Memory*>(m);mock::need(!p->imported,"freed Vulkan memory before zeMemFree");mock::ranges.erase(p->p);delete p;}
void vkGetPhysicalDeviceExternalBufferProperties(VkPhysicalDevice,const VkPhysicalDeviceExternalBufferInfo*,VkExternalBufferProperties*p){p->externalMemoryProperties.compatibleHandleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;p->externalMemoryProperties.externalMemoryFeatures=mock::exportable?VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT:0;}
VkResult vkCreateBuffer(VkDevice,const VkBufferCreateInfo*p,const void*,VkBuffer*b){mock::need(p->pNext!=nullptr,"missing exportable buffer pNext");*b=new mock::Buffer{size_t(p->size)};return VK_SUCCESS;}
void vkGetBufferMemoryRequirements(VkDevice,VkBuffer b,VkMemoryRequirements*r){r->size=(static_cast<mock::Buffer*>(b)->n+63)&~size_t(63);r->alignment=64;r->memoryTypeBits=1;}
void vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice,VkPhysicalDeviceMemoryProperties*p){p->memoryTypeCount=1;p->memoryTypes[0].propertyFlags=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;}
VkResult vkAllocateMemory(VkDevice,const VkMemoryAllocateInfo*p,const void*,VkDeviceMemory*m){auto ex=static_cast<const VkExportMemoryAllocateInfo*>(p->pNext);mock::need(ex&&ex->pNext,"missing export + dedicated memory chain");auto obj=new mock::Memory(size_t(p->allocationSize));mock::ranges[obj->p]=obj;*m=obj;return VK_SUCCESS;}
VkResult vkBindBufferMemory(VkDevice,VkBuffer b,VkDeviceMemory m,VkDeviceSize off){auto v=static_cast<mock::Buffer*>(b);v->memory=static_cast<mock::Memory*>(m);v->offset=size_t(off);return VK_SUCCESS;}
VkResult vkResetCommandBuffer(VkCommandBuffer p,uint32_t){static_cast<mock::Command*>(p)->barriers.clear();return VK_SUCCESS;}
VkResult vkBeginCommandBuffer(VkCommandBuffer,const VkCommandBufferBeginInfo*){return VK_SUCCESS;}
void vkCmdPipelineBarrier(VkCommandBuffer c,VkPipelineStageFlags src,VkPipelineStageFlags dst,uint32_t,uint32_t,const void*,uint32_t n,const VkBufferMemoryBarrier*b,uint32_t,const void*){mock::need((src==VK_PIPELINE_STAGE_ALL_COMMANDS_BIT&&dst==VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)||(src==VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT&&dst==VK_PIPELINE_STAGE_ALL_COMMANDS_BIT),"invalid stage masks");auto p=static_cast<mock::Command*>(c);p->barriers.assign(b,b+n);}
VkResult vkEndCommandBuffer(VkCommandBuffer){return VK_SUCCESS;}
VkResult vkResetFences(VkDevice,uint32_t n,const VkFence*f){for(uint32_t i=0;i<n;++i)*static_cast<bool*>(f[i])=false;return VK_SUCCESS;}
VkResult vkQueueSubmit(VkQueue,uint32_t n,const VkSubmitInfo*s,VkFence f){if(mock::submit_fail)return VK_ERROR_UNKNOWN;for(uint32_t i=0;i<n;++i)for(uint32_t j=0;j<s[i].commandBufferCount;++j)for(const auto&m:static_cast<mock::Command*>(s[i].pCommandBuffers[j])->barriers){auto b=static_cast<mock::Buffer*>(m.buffer);if(m.dstQueueFamilyIndex==VK_QUEUE_FAMILY_EXTERNAL){mock::need(!b->external,"double external release");mock::need(!m.dstAccessMask&&m.srcAccessMask,"bad release access masks");b->external=true;}else{mock::need(b->external,"acquire without release");mock::need(!m.srcAccessMask&&m.dstAccessMask,"bad acquire access masks");b->external=false;}}*static_cast<bool*>(f)=true;return VK_SUCCESS;}
VkResult vkWaitForFences(VkDevice,uint32_t n,const VkFence*f,VkBool32,uint64_t){for(uint32_t i=0;i<n;++i)if(!*static_cast<bool*>(f[i]))return VK_TIMEOUT;return VK_SUCCESS;}
ze_result_t zeMemGetAllocProperties(ze_context_handle_t c,const void*p,ze_memory_allocation_properties_t*r,ze_device_handle_t*d){if(c!=reinterpret_cast<void*>(0x101)||!mock::find(p))return ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;r->type=ZE_MEMORY_TYPE_DEVICE;if(d)*d=reinterpret_cast<void*>(0x102);return ZE_RESULT_SUCCESS;}
ze_result_t zeMemGetAddressRange(ze_context_handle_t,const void*p,void**b,size_t*n){auto m=mock::find(p);if(!m)return ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;*b=m->p;*n=m->n;return ZE_RESULT_SUCCESS;}
ze_result_t zeDeviceGetProperties(ze_device_handle_t,ze_device_properties_t*p){p->vendorId=0x8086;p->deviceId=0x56a1;auto l=static_cast<ze_device_luid_ext_properties_t*>(p->pNext);l->luid.id[0]=mock::bad_luid?9:7;l->nodeMask=1;return ZE_RESULT_SUCCESS;}
ze_result_t zeDeviceGetExternalMemoryProperties(ze_device_handle_t,ze_device_external_memory_properties_t*p){p->memoryAllocationImportTypes=ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32;return ZE_RESULT_SUCCESS;}
ze_result_t zeMemAllocDevice(ze_context_handle_t,const ze_device_mem_alloc_desc_t*p,size_t n,size_t,ze_device_handle_t,void**out){if(mock::import_fail)return ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;auto ex=static_cast<const ze_external_memory_import_win32_handle_t*>(p->pNext);mock::need(ex&&mock::handles.count(ex->handle),"unowned external NT handle");auto m=*static_cast<mock::Memory**>(ex->handle);mock::need(n==m->n,"import wrong allocation size");m->imported=true;*out=m->p;++mock::import_count;return ZE_RESULT_SUCCESS;}
ze_result_t zeMemFree(ze_context_handle_t,void*p){auto m=mock::find(p);mock::need(m&&m->imported,"invalid import free");m->imported=false;++mock::free_count;return ZE_RESULT_SUCCESS;}
}
int main()try {
 using mock::need;auto rejection=[](auto fn){bool rejected=false;try{fn();}catch(const std::exception&){rejected=true;}mock::need(rejected,"expected rejection");};
 sycl::queue q;sycl::mock_execute=true;
 auto create=[&]{return zc::Context::create(q,(void*)1,(void*)2,(void*)3,0,std::make_shared<std::recursive_mutex>(),std::make_shared<int>(0));};
 mock::bad_luid=true;rejection(create);mock::bad_luid=false;
 {
 auto ctx=create();rejection([&]{ctx->create_buffer(4096,0x80000000u);});mock::exportable=false;rejection([&]{ctx->create_buffer(4096);});mock::exportable=true;
 mock::import_fail=true;rejection([&]{ctx->create_buffer(4096);});mock::import_fail=false;need(mock::handles.empty(),"failed import leaked NT handle");
 auto io=ctx->create_buffer(32768);auto native=static_cast<mock::Buffer*>(io->vk_buffer());
 auto alias=ctx->adopt({io->vk_buffer(),native->memory,{native->memory->n,0,io->bytes(),io->generation()},io});
 need(ctx->counters().imports==1&&ctx->counters().cache_hits==1,"not cached once");
 rejection([&]{ctx->adopt({io->vk_buffer(),native->memory,{native->memory->n,0,io->bytes(),io->generation()+1},io});});
 need(io->at<float>(64,4)==reinterpret_cast<float*>(static_cast<uint8_t*>(native->memory->p)+64),"offset not relative to allocation");
 // The exact native resolver, not just its portable arithmetic, must add both offsets.
 auto sub_native=new mock::Buffer{1024,native->memory,4096};
 struct SubLease {std::shared_ptr<zc::Buffer> arena;mock::Buffer*buffer;~SubLease(){delete buffer;}};
 auto sub_owner=std::make_shared<SubLease>();sub_owner->arena=io;sub_owner->buffer=sub_native;
 auto sub=ctx->adopt({sub_native,native->memory,{native->memory->n,4096,1024,io->generation()},sub_owner});
 need(sub->at<float>(64,4)==reinterpret_cast<float*>(static_cast<uint8_t*>(native->memory->p)+4096+64),"binding + tensor offset not composed");
 rejection([&]{sub->at<float>(1020,2);});
 ggml_tensor tensor;tensor.ne[0]=16;tensor.ne[1]=2;tensor.ne[2]=1;tensor.nb[0]=4;tensor.nb[1]=64;tensor.view_offs=9999; // already resolved: MUST NOT add again
 zc::TensorBinding meta{&tensor,sub,64};auto mv=meta.read(2,16);
 need(mv.data==sub->at<float>(64,32),"metadata adapter added view_offs twice");
 tensor.nb[1]=80;rejection([&]{meta.read(2,16);});tensor.nb[1]=64;
 tensor.type=GGML_TYPE_F16;rejection([&]{meta.read(2,16);});tensor.type=GGML_TYPE_F32;


 rejection([&]{io->at<float>(io->bytes()-2,1);});rejection([&]{io->at<float>(4,SIZE_MAX);});
 l4::Config c{3,256,256,256,128,256,8,2,l4::Entry::advance};auto plan=l4::make_plan(c,1,1,1,1,1,1,true);
 auto scratch=std::make_shared<mock::Private>(plan.bytes),weights=std::make_shared<mock::Private>(1024*1024);auto ws=l4::bind_workspace(scratch->m.p,scratch->m.n,plan);
 size_t cursor=0;auto copy=[&](const void*p,size_t n){cursor=l4::aligned(cursor);auto dst=static_cast<uint8_t*>(weights->m.p)+cursor;std::memcpy(dst,p,n);cursor+=n;need(cursor<=weights->m.n,"weights exceed arena");return dst;};
 auto weight=[&](Shape s,unsigned seed){auto v=repack_s2tile8(l4::synthetic_weight(s,seed),s);return copy(v.data(),v.size());};
 l4::Weights wt;wt.o=weight({256,256,1},1);wt.q=weight({256,256,1},2);wt.k=weight({256,128,1},3);wt.v=weight({256,128,1},4);
 wt.gate=weight({256,256,8},5);wt.up=weight({256,256,8},6);wt.down=weight({256,256,8},7);
 std::vector<float>g(256,1.f),r(8*256);for(size_t i=0;i<r.size();++i)r[i]=float(int(i%17)-8)/1024;
 wt.ffn_norm=reinterpret_cast<float*>(copy(g.data(),g.size()*4));wt.next_attn_norm=reinterpret_cast<float*>(copy(g.data(),g.size()*4));wt.router=reinterpret_cast<float*>(copy(r.data(),r.size()*4));wt.current_layer=0;wt.next_layer=1;
 size_t pos=64;auto view=[&](uint32_t width){auto p=io->at<float>(pos,size_t(c.tokens)*width);pos=l4::aligned(pos+size_t(c.tokens)*width*4);return p;};
 auto x=view(256),a=view(256),h=view(256),qv=view(256),k=view(128),v=view(128);
 for(size_t i=0;i<3*256;++i){x[i]=float(int(i%31)-15)/64;a[i]=float(int(i%29)-14)/64;}
 l4::Bindings b{l4::contiguous_read(x,3,256),l4::contiguous_read(a,3,256),l4::contiguous_write(h,3,256),l4::contiguous_write(qv,3,256),l4::contiguous_write(k,3,128),l4::contiguous_write(v,3,128)};
 l4::GraphSemantics sem;sem.raw_fa_before_o=sem.raw_qkv_before_qk_norm=sem.sequential_residual=sem.maple_clamped_swiglu=sem.weights_validated=true;
 l4::Options opt;std::vector<zc::PrivateRegion>priv{{scratch->m.p,scratch->m.n,scratch},{weights->m.p,weights->m.n,weights}};
 rejection([&]{ctx->prepare(wt,b,opt,ws,sem,{io,alias},priv);});
 auto bad=b;bad.hidden.data=x;rejection([&]{ctx->prepare(wt,bad,opt,ws,sem,{io},priv);});
 rejection([&]{ctx->prepare(wt,b,opt,ws,sem,{io},{});});
 auto prepared=ctx->prepare(wt,b,opt,ws,sem,{io},priv);const auto before=ctx->counters();
 sycl::mock_waits=0;auto run=ctx->execute(prepared);need(sycl::mock_waits==1,"more than SYCL outer wait");
 need(!native->external,"missing Vulkan acquire");need(run.run.boundary_traffic.bytes()==0,"boundary copy present");
 for(size_t i=0;i<3*256;++i)need(std::isfinite(h[i])&&std::isfinite(qv[i]),"nonfinite output");
 auto expected=std::vector<float>(h,h+3*256);auto run2=ctx->execute(prepared);need(std::equal(expected.begin(),expected.end(),h),"reuse changed result");
 const auto after=ctx->counters();need(after.imports==before.imports&&after.allocations==before.allocations&&after.exports==before.exports,"hotpath alloc/import");
 need(after.islands==2&&after.inbound_submits==2&&after.outbound_submits==2,"wrong boundary submit counts");
 need(after.activation_h2d_bytes==0&&after.activation_d2h_bytes==0&&after.boundary_d2d_bytes==0,"copy counters");
 mock::submit_fail=true;rejection([&]{ctx->execute(prepared);});mock::submit_fail=false;need(ctx->poisoned(),"submission failure not poisoned");rejection([&]{ctx->execute(prepared);});ctx->drain();
 }
 need(mock::handles.empty()&&mock::ranges.empty(),"leaked normal-lifetime resource");need(mock::import_count==mock::free_count,"import not released once");
 std::cout<<"Level4 zero-copy NATIVE API-DOUBLE tests PASS checks="<<mock::checks<<"; NOT real Vulkan/L0/GPU validation\n";return 0;
}catch(const std::exception&e){std::cerr<<"zero-copy mock FAIL: "<<e.what()<<'\n';return 1;}
