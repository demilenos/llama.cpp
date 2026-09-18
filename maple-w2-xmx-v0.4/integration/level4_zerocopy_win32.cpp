#include "level4_device_identity_win32.hpp"
// SPDX-License-Identifier: MIT
#include "level4_zerocopy_win32.hpp"
#include "dg2_validation.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <map>
#include <utility>
namespace maple_w2::level4::zc {
namespace {
using Clock=std::chrono::steady_clock;
void vk_check(VkResult r,const char*n){if(r!=VK_SUCCESS)throw std::runtime_error(std::string(n)+" VkResult="+std::to_string(int(r)));}
void ze_check(ze_result_t r,const char*n){if(r!=ZE_RESULT_SUCCESS)throw std::runtime_error(std::string(n)+" ze_result="+std::to_string(uint32_t(r)));}
template<class T>uint64_t handle_key(T h){static_assert(sizeof(h)<=8);uint64_t v=0;std::memcpy(&v,&h,sizeof(h));return v;}
double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
struct UniqueHandle {HANDLE value=nullptr;~UniqueHandle(){if(value&&value!=INVALID_HANDLE_VALUE)CloseHandle(value);}};
}
struct State {
    sycl::queue queue;VkPhysicalDevice physical;VkDevice device;VkQueue vk_queue;uint32_t family;
    ze_context_handle_t ze_context=nullptr;ze_device_handle_t ze_device=nullptr;
    std::shared_ptr<std::recursive_mutex> mutex;std::shared_ptr<void> device_owner;
    PFN_vkGetMemoryWin32HandleKHR export_handle=nullptr;
    std::map<std::pair<uint64_t,uint64_t>,std::weak_ptr<ImportedMemory>> imports;
    Counters stats;OwnershipState ownership;uint64_t next_generation=1,timeout_ns;
    State(sycl::queue q,VkPhysicalDevice p,VkDevice d,VkQueue v,uint32_t f,std::shared_ptr<std::recursive_mutex> m,
          std::shared_ptr<void> own,uint64_t timeout):queue(std::move(q)),physical(p),device(d),vk_queue(v),family(f),mutex(std::move(m)),device_owner(std::move(own)),timeout_ns(timeout){}
    void drain_locked() {
        // Always drain BOTH APIs, even if the first reports an asynchronous error.
        std::exception_ptr error;try{queue.wait_and_throw();}catch(...){error=std::current_exception();}
        auto r=vkQueueWaitIdle(vk_queue);if(r!=VK_SUCCESS&&!error)try{vk_check(r,"vkQueueWaitIdle");}catch(...){error=std::current_exception();}
        if(error)std::rethrow_exception(error);
    }
};
struct ImportedMemory {
    std::shared_ptr<State> state;std::shared_ptr<void> source_owner;
    VkDeviceMemory memory=VK_NULL_HANDLE;void*pointer=nullptr;size_t bytes=0;uint64_t generation=0;
    ~ImportedMemory(){if(!pointer)return;
        try {
            std::unique_lock<std::recursive_mutex> lock(*state->mutex);state->drain_locked();
            ze_check(zeMemFree(state->ze_context,pointer),"zeMemFree(external)");pointer=nullptr;
        } catch(...) {
            // Fail closed. Do not release Vulkan memory under potentially pending
            // external work. Preserve context + source owner until process exit.
            std::fprintf(stderr,"LEVEL4_ZC: unsafe teardown; quarantining external allocation\n");
            (void)new std::pair<std::shared_ptr<State>,std::shared_ptr<void>>(state,source_owner);
        }
    }
};
namespace {
struct OwnedVk {
    std::shared_ptr<State> state;VkBuffer buffer=VK_NULL_HANDLE;VkDeviceMemory memory=VK_NULL_HANDLE;
    ~OwnedVk(){if(buffer)vkDestroyBuffer(state->device,buffer,nullptr);if(memory)vkFreeMemory(state->device,memory,nullptr);}
};
void check_native_range(const State&s,const void*p,size_t n) {
    if(!p||!n)throw std::invalid_argument("empty device region");
    ze_memory_allocation_properties_t prop{};prop.stype=ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    ze_device_handle_t dev=nullptr;ze_check(zeMemGetAllocProperties(s.ze_context,p,&prop,&dev),"zeMemGetAllocProperties");
    if(prop.type!=ZE_MEMORY_TYPE_DEVICE||dev!=s.ze_device)throw std::invalid_argument("Level4 zero-copy region is not this device/context's device memory");
    void*base=nullptr;size_t size=0;ze_check(zeMemGetAddressRange(s.ze_context,p,&base,&size),"zeMemGetAddressRange");
    if(!covers(base,size,p,n))throw std::invalid_argument("native region exceeds imported/USM allocation");
}
}
Buffer::Buffer(std::shared_ptr<ImportedMemory> m,std::shared_ptr<void> o,VkBuffer b,BufferExtent e)
 :memory_(std::move(m)),buffer_owner_(std::move(o)),buffer_(b),extent_(e){}
void*Buffer::data(size_t offset,size_t n,size_t alignment)const {
    const auto absolute=extent_.resolve(offset,n,alignment,memory_->generation);
    if(memory_->bytes!=extent_.allocation_bytes)throw std::logic_error("import/allocation extent mismatch");
    const auto p=reinterpret_cast<uintptr_t>(memory_->pointer);
    if(absolute>UINTPTR_MAX-p||n>UINTPTR_MAX-p-absolute||(p+absolute)%alignment)
        throw std::invalid_argument("native pointer arithmetic overflow or alignment");
    return reinterpret_cast<void*>(p+absolute);
}
struct Context::Transport {
    std::shared_ptr<State> state;VkCommandPool pool=VK_NULL_HANDLE;
    VkCommandBuffer release=VK_NULL_HANDLE,acquire=VK_NULL_HANDLE;
    VkFence release_fence=VK_NULL_HANDLE,acquire_fence=VK_NULL_HANDLE;
    bool acquire_pending=false;
    std::vector<std::shared_ptr<Buffer>> previous_shared;
    std::vector<PrivateRegion> previous_private;
    explicit Transport(std::shared_ptr<State>s):state(std::move(s)){}
    ~Transport(){
        if(release_fence)vkDestroyFence(state->device,release_fence,nullptr);
        if(acquire_fence)vkDestroyFence(state->device,acquire_fence,nullptr);
        if(pool)vkDestroyCommandPool(state->device,pool,nullptr);
    }
};
Context::Context(std::shared_ptr<State>s):state_(std::move(s)),transport_(new Transport(state_)){}
std::shared_ptr<Context> Context::create(sycl::queue q,VkPhysicalDevice p,VkDevice d,VkQueue v,uint32_t family,
    std::shared_ptr<std::recursive_mutex>mutex,std::shared_ptr<void>owner,uint64_t timeout) {
    if(!p||!d||!v||!mutex||!owner||!timeout)throw std::invalid_argument("missing Vulkan queue/device owner or timeout");
    if(q.get_device().get_backend()!=sycl::backend::ext_oneapi_level_zero)
        throw std::invalid_argument("Level4 zero-copy requires LevelZero, not SYCL OpenCL");
    const auto devices=q.get_context().get_devices();
    if(devices.size()!=1||devices.front()!=q.get_device())throw std::invalid_argument("use an explicit single-device SYCL context");
    auto s=std::make_shared<State>(q,p,d,v,family,std::move(mutex),std::move(owner),timeout);
    s->ze_context=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
    s->ze_device=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 vp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};vp.pNext=&id;vkGetPhysicalDeviceProperties2(p,&vp);
    if(vp.properties.apiVersion<VK_API_VERSION_1_1)throw std::runtime_error("Vulkan 1.1+ required for external-memory ownership transport");
    ze_device_luid_ext_properties_t zl{};zl.stype=ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES;
    ze_device_properties_t zp{};zp.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;zp.pNext=&zl;
    ze_check(zeDeviceGetProperties(s->ze_device,&zp),"zeDeviceGetProperties(LUID)");
    DeviceIdentity a{vp.properties.vendorID,vp.properties.deviceID,id.deviceNodeMask,{},bool(id.deviceLUIDValid)},b{zp.vendorId,zp.deviceId,zl.nodeMask,{},true};
    static_assert(VK_LUID_SIZE==8&&ZE_MAX_DEVICE_LUID_SIZE_EXT==8);
    std::memcpy(a.luid.data(),id.deviceLUID,8);std::memcpy(b.luid.data(),zl.luid.id,8);
    std::fprintf(stderr,"LEVEL4_ZC_IDENTITY vk_vendor=%x vk_device=%x vk_mask=%x vk_valid=%d ze_vendor=%x ze_device=%x ze_mask=%x vk_luid=",a.vendor,a.device,a.node_mask,int(a.luid_valid),b.vendor,b.device,b.node_mask);
    for(auto byte:a.luid)std::fprintf(stderr,"%02x",unsigned(byte));
    std::fprintf(stderr," ze_luid=");for(auto byte:b.luid)std::fprintf(stderr,"%02x",unsigned(byte));std::fprintf(stderr,"\n");
#if defined(_WIN32) && !defined(MAPLE_ZC_NATIVE_MOCK)
    require_same_windows_device(a,b,id.deviceUUID,zp.uuid.id,(zp.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE)!=0);
#else
    require_same_device(a,b);
#endif
    std::fprintf(stderr,"LEVEL4_ZC_DEVICE vendor=0x%x device=0x%x vk_api=%u vk_driver=%u node_mask=0x%x luid=",a.vendor,a.device,vp.properties.apiVersion,vp.properties.driverVersion,a.node_mask);
    for(auto byte:a.luid)std::fprintf(stderr,"%02x",unsigned(byte));
    std::fprintf(stderr," native_identity_match=1 host_fenced=1\n");
    uint32_t nf=0;vkGetPhysicalDeviceQueueFamilyProperties(p,&nf,nullptr);std::vector<VkQueueFamilyProperties>families(nf);
    vkGetPhysicalDeviceQueueFamilyProperties(p,&nf,families.data());
    if(family>=nf||!(families[family].queueFlags&VK_QUEUE_COMPUTE_BIT))throw std::invalid_argument("not a Vulkan compute queue family");
    s->export_handle=reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(d,"vkGetMemoryWin32HandleKHR"));
    if(!s->export_handle)throw std::runtime_error("enable VK_KHR_external_memory_win32 at VkDevice creation");
    ze_device_external_memory_properties_t ep{};ep.stype=ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES;
    ze_check(zeDeviceGetExternalMemoryProperties(s->ze_device,&ep),"zeDeviceGetExternalMemoryProperties");
    if(!(ep.memoryAllocationImportTypes&ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32))
        throw std::runtime_error("LevelZero driver does not advertise OPAQUE_WIN32 memory import; no copy fallback");
    auto ctx=std::shared_ptr<Context>(new Context(s));
    VkCommandPoolCreateInfo pc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pc.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;pc.queueFamilyIndex=family;
    vk_check(vkCreateCommandPool(d,&pc,nullptr,&ctx->transport_->pool),"vkCreateCommandPool");
    VkCommandBufferAllocateInfo ac{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ac.commandPool=ctx->transport_->pool;ac.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ac.commandBufferCount=2;
    VkCommandBuffer cb[2]{};vk_check(vkAllocateCommandBuffers(d,&ac,cb),"vkAllocateCommandBuffers");ctx->transport_->release=cb[0];ctx->transport_->acquire=cb[1];
    VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vk_check(vkCreateFence(d,&fc,nullptr,&ctx->transport_->release_fence),"vkCreateFence(release)");
    vk_check(vkCreateFence(d,&fc,nullptr,&ctx->transport_->acquire_fence),"vkCreateFence(acquire)");return ctx;
}
Context::~Context(){if(!transport_)return;try{drain();}catch(...){
    std::fprintf(stderr,"LEVEL4_ZC: teardown drain failed; quarantining command pool and leases\n");(void)transport_.release();return;}
    transport_.reset();
}
void Context::drain(){
    std::vector<std::shared_ptr<Buffer>> retired_shared;
    std::vector<PrivateRegion> retired_private;
    {std::lock_guard<std::recursive_mutex>lock(*state_->mutex);state_->drain_locked();
     if(transport_){transport_->acquire_pending=false;retired_shared.swap(transport_->previous_shared);retired_private.swap(transport_->previous_private);}}
}
Counters Context::counters()const{std::lock_guard<std::recursive_mutex>lock(*state_->mutex);return state_->stats;}
bool Context::poisoned()const{std::lock_guard<std::recursive_mutex>lock(*state_->mutex);return state_->ownership.value()==Ownership::poisoned;}
std::shared_ptr<Buffer> Context::adopt_unlocked(const ExportedBuffer&b) {
    state_->ownership.check();b.extent.validate();if(!b.buffer||!b.memory||!b.source_owner)throw std::invalid_argument("exported buffer needs native handles and allocation lease");
    const auto key=std::make_pair(handle_key(b.memory),b.extent.generation);
    // A live handle must never be relabelled with a new generation. Never retire
    // a mapping behind an in-flight graph; keep old source_owner until drain.
    for(auto it=state_->imports.begin();it!=state_->imports.end();) {
        auto m=it->second.lock();if(!m){it=state_->imports.erase(it);continue;}
        if(it->first.first==key.first&&it->first.second!=key.second)throw std::invalid_argument("VkDeviceMemory generation changed while previous import still live");++it;
    }
    auto found=state_->imports.find(key);std::shared_ptr<ImportedMemory>m=found==state_->imports.end()?nullptr:found->second.lock();
    if(m){if(m->bytes!=b.extent.allocation_bytes)throw std::invalid_argument("cached import size mismatch");++state_->stats.cache_hits;}
    else {
        UniqueHandle h;VkMemoryGetWin32HandleInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};gi.memory=b.memory;gi.handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        vk_check(state_->export_handle(state_->device,&gi,&h.value),"vkGetMemoryWin32HandleKHR");++state_->stats.exports;
        if(!h.value||h.value==INVALID_HANDLE_VALUE)throw std::runtime_error("invalid exported NT handle");
        ze_external_memory_import_win32_handle_t ex{};ex.stype=ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32;ex.flags=ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32;ex.handle=h.value;
        ze_device_mem_alloc_desc_t md{};md.stype=ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;md.pNext=&ex;
        m=std::make_shared<ImportedMemory>();m->state=state_;m->source_owner=b.source_owner;m->memory=b.memory;m->bytes=b.extent.allocation_bytes;m->generation=b.extent.generation;
        void*ptr=nullptr;ze_check(zeMemAllocDevice(state_->ze_context,&md,b.extent.allocation_bytes,0,state_->ze_device,&ptr),"zeMemAllocDevice(OPAQUE_WIN32)");
        try{check_native_range(*state_,ptr,b.extent.allocation_bytes);}catch(...){(void)zeMemFree(state_->ze_context,ptr);throw;}
        m->pointer=ptr;
        state_->imports[key]=m;++state_->stats.imports;
        // The exported NT handle is closed once, after a successful import.
    }
    return std::shared_ptr<Buffer>(new Buffer(m,b.source_owner,b.buffer,b.extent));
}
std::shared_ptr<Buffer> Context::adopt(const ExportedBuffer&b){std::lock_guard<std::recursive_mutex>lock(*state_->mutex);return adopt_unlocked(b);}
std::shared_ptr<Buffer> Context::create_buffer(size_t bytes,VkBufferUsageFlags usage) {
    std::lock_guard<std::recursive_mutex>lock(*state_->mutex);state_->ownership.check();if(!bytes)throw std::invalid_argument("zero-sized shared buffer");
    constexpr VkBufferUsageFlags supported=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if(!usage||(usage&~supported))throw std::invalid_argument("shared allocator supports storage/transfer usage only; BDA and sparse allocation need a different contract");
    VkPhysicalDeviceExternalBufferInfo query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};query.usage=usage;query.handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkExternalBufferProperties props{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};vkGetPhysicalDeviceExternalBufferProperties(state_->physical,&query,&props);
    if(!(props.externalMemoryProperties.externalMemoryFeatures&VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT)||
       !(props.externalMemoryProperties.compatibleHandleTypes&VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT))throw std::runtime_error("Vulkan buffer usage cannot export OPAQUE_WIN32");
    auto owner=std::make_shared<OwnedVk>();owner->state=state_;
    VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};ext.handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.pNext=&ext;bi.size=bytes;bi.usage=usage;bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(state_->device,&bi,nullptr,&owner->buffer),"vkCreateBuffer(exportable)");
    VkMemoryRequirements req{};vkGetBufferMemoryRequirements(state_->device,owner->buffer,&req);
    VkPhysicalDeviceMemoryProperties mp{};vkGetPhysicalDeviceMemoryProperties(state_->physical,&mp);uint32_t type=UINT32_MAX;
    for(uint32_t i=0;i<mp.memoryTypeCount;++i)if((req.memoryTypeBits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)){type=i;break;}
    if(type==UINT32_MAX)throw std::runtime_error("no device-local external memory type");
    // Dedicated binding also satisfies DEDICATED_ONLY. One VkBuffer can still be
    // an arena holding many tensors. No per-tensor physical allocation required.
    VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};dedicated.buffer=owner->buffer;
    VkExportMemoryAllocateInfo export_info{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};export_info.handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;export_info.pNext=&dedicated;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.pNext=&export_info;ai.allocationSize=req.size;ai.memoryTypeIndex=type;
    vk_check(vkAllocateMemory(state_->device,&ai,nullptr,&owner->memory),"vkAllocateMemory(exportable)");
    vk_check(vkBindBufferMemory(state_->device,owner->buffer,owner->memory,0),"vkBindBufferMemory");
    if(req.size>SIZE_MAX)throw std::overflow_error("allocation exceeds size_t");
    ++state_->stats.allocations;
    const auto generation=state_->next_generation++;if(!generation)throw std::overflow_error("allocation generation overflow");
    return adopt_unlocked({owner->buffer,owner->memory,{size_t(req.size),0,bytes,generation},owner});
}
Prepared Context::prepare(const Weights&wt,const Bindings&b,Options o,const Workspace&w,GraphSemantics sem,
    std::vector<std::shared_ptr<Buffer>> shared,std::vector<PrivateRegion> private_regions) {
    // Always prove the no-copy contract BEFORE the ownership transfer.
    o.boundary=BoundaryPolicy::direct_required;
    if(o.diagnostic_stage_waits||o.moe.diagnostic_host_waits)throw std::invalid_argument("per-stage host waits forbidden in zero-copy executor");
    (void)bind_workspace(w.base,w.capacity,w.plan);
    validate(wt,b,o,w);validate_dg2_device(state_->queue.get_device());
    std::lock_guard<std::recursive_mutex>lock(*state_->mutex);state_->ownership.check();
    if(shared.empty())throw std::invalid_argument("no exported boundary buffers");
    for(size_t i=0;i<shared.size();++i){auto&x=shared[i];if(!x||x->memory_->state!=state_)throw std::invalid_argument("foreign shared-buffer context");
        for(size_t j=0;j<i;++j)if(x->buffer_==shared[j]->buffer_)throw std::invalid_argument("list each shared VkBuffer once");
        // Distinct VkBuffer aliases require an alias memory barrier strategy not
        // provided by this first transport. Non-overlapping suballocations work.
        for(size_t j=0;j<i;++j)if(x->memory_==shared[j]->memory_) {
            auto a=x->extent_.bind_offset,z=shared[j]->extent_.bind_offset;
            if(a<checked_add(z,shared[j]->bytes())&&z<checked_add(a,x->bytes()))throw std::invalid_argument("overlapping VkBuffer aliases unsupported");
        }
    }
    for(const auto&r:private_regions){if(!r.owner)throw std::invalid_argument("private allocation lifetime lease missing");check_native_range(*state_,r.base,r.bytes);
        for(const auto&kv:state_->imports)if(auto m=kv.second.lock())if(covers(m->pointer,m->bytes,r.base,1)||covers(r.base,r.bytes,m->pointer,1))throw std::invalid_argument("imported memory must be registered as a shared buffer, not private USM");}
    auto shared_covers=[&](const void*p,size_t n){for(const auto&x:shared)if(covers(x->data(0,x->bytes()),x->bytes(),p,n))return true;return false;};
    auto owned=[&](const void*p,size_t n){if(!n)return;if(shared_covers(p,n))return;for(const auto&r:private_regions)if(covers(r.base,r.bytes,p,n))return;throw std::invalid_argument("device pointer is not covered by a registered lifetime lease");};
    const auto c=w.plan.config;
    auto boundary=[&](const auto&v,uint32_t width){size_t n=checked_mul(required_elements(v,c.tokens,width),sizeof(float));if(!shared_covers(v.data,n))throw std::invalid_argument("boundary is not an imported Vulkan buffer");};
    boundary(b.residual,c.hidden);boundary(b.hidden,c.hidden);if(has_post(c.entry))boundary(b.attention,c.attention);
    if(has_qkv(c.entry)){boundary(b.q,c.q_width);boundary(b.k,c.kv_width);boundary(b.v,c.kv_width);}
    owned(w.base,w.plan.bytes);
    if(has_post(c.entry)){owned(wt.o,nbytes({c.attention,c.hidden,1}));for(auto p:{wt.gate,wt.up})owned(p,nbytes({c.hidden,c.ffn,c.experts}));owned(wt.down,nbytes({c.ffn,c.hidden,c.experts}));owned(wt.ffn_norm,size_t(c.hidden)*4);owned(wt.router,size_t(c.hidden)*c.experts*4);}
    if(has_qkv(c.entry)){owned(wt.q,nbytes({c.hidden,c.q_width,1}));owned(wt.k,nbytes({c.hidden,c.kv_width,1}));owned(wt.v,nbytes({c.hidden,c.kv_width,1}));owned(wt.next_attn_norm,size_t(c.hidden)*4);}
    sem.same_device_context=true; // only this property is established here
    if(const char*why=reject_graph(c,sem))throw std::invalid_argument(why);
    Prepared p;p.state_=state_;p.weights_=wt;p.bindings_=b;p.options_=o;p.workspace_=w;p.shared_=std::move(shared);p.private_=std::move(private_regions);p.semantics_=sem;return p;
}
Result Context::execute(const Prepared&p) {
    if(p.state_!=state_)throw std::invalid_argument("prepared island belongs to a different context");
    validate(p.weights_,p.bindings_,p.options_,p.workspace_);
    // Declare retired leases BEFORE the lock so destruction happens after unlock.
    std::vector<std::shared_ptr<Buffer>>retired_shared;std::vector<PrivateRegion>retired_private;
    std::unique_lock<std::recursive_mutex>lock(*state_->mutex);state_->ownership.check();
    auto&tr=*transport_;Result result;const auto t0=Clock::now();
    auto begin=[&](VkCommandBuffer cb){vk_check(vkResetCommandBuffer(cb,0),"vkResetCommandBuffer");VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;vk_check(vkBeginCommandBuffer(cb,&bi),"vkBeginCommandBuffer");};
    auto barriers=[&](VkCommandBuffer cb,bool release){std::vector<VkBufferMemoryBarrier>v;v.reserve(p.shared_.size());for(const auto&b:p.shared_){VkBufferMemoryBarrier m{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};m.buffer=b->buffer_;m.offset=0;m.size=b->bytes();m.srcQueueFamilyIndex=release?state_->family:VK_QUEUE_FAMILY_EXTERNAL;m.dstQueueFamilyIndex=release?VK_QUEUE_FAMILY_EXTERNAL:state_->family;m.srcAccessMask=release?(VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT):0;m.dstAccessMask=release?0:(VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT);v.push_back(m);}
        vkCmdPipelineBarrier(cb,release?VK_PIPELINE_STAGE_ALL_COMMANDS_BIT:VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,release?VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT:VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,uint32_t(v.size()),v.data(),0,nullptr);};
    auto submit=[&](VkCommandBuffer cb,VkFence f){vk_check(vkEndCommandBuffer(cb),"vkEndCommandBuffer");vk_check(vkResetFences(state_->device,1,&f),"vkResetFences");VkSubmitInfo s{VK_STRUCTURE_TYPE_SUBMIT_INFO};s.commandBufferCount=1;s.pCommandBuffers=&cb;vk_check(vkQueueSubmit(state_->vk_queue,1,&s,f),"vkQueueSubmit");};
    try {
        if(tr.acquire_pending){vk_check(vkWaitForFences(state_->device,1,&tr.acquire_fence,VK_TRUE,state_->timeout_ns),"acquire command retirement");tr.acquire_pending=false;}
        // Keep both previous downstream and current resources alive until the
        // new release fence covers all prior submissions on this Vulkan queue.
        auto current_shared=p.shared_;auto current_private=p.private_;
        begin(tr.release);barriers(tr.release,true);state_->ownership.release();
        submit(tr.release,tr.release_fence);++state_->stats.inbound_submits;
        vk_check(vkWaitForFences(state_->device,1,&tr.release_fence,VK_TRUE,state_->timeout_ns),"Vulkan release fence");state_->ownership.released();
        retired_shared.swap(tr.previous_shared);retired_private.swap(tr.previous_private);
        tr.previous_shared=std::move(current_shared);tr.previous_private=std::move(current_private);
        const auto t1=Clock::now();result.run=level4::enqueue(state_->queue,p.weights_,p.bindings_,p.options_,p.workspace_);
        result.run.done.wait_and_throw();const auto t2=Clock::now();
        if(result.run.boundary_traffic.bytes())throw std::logic_error("strict zero-copy path submitted a boundary copy");
        begin(tr.acquire);barriers(tr.acquire,false);submit(tr.acquire,tr.acquire_fence);tr.acquire_pending=true;state_->ownership.acquire();
        ++state_->stats.outbound_submits;++state_->stats.islands;
        const auto t3=Clock::now();result.timing={us(t0,t1),us(t1,t2),us(t2,t3),us(t0,t3)};return result;
    } catch(...) {
        state_->ownership.poison();
        // If a release was queued then even a failed SYCL submission forbids
        // falling back on the same output. Preserve ALL leases until drain.
        tr.previous_shared.insert(tr.previous_shared.end(),p.shared_.begin(),p.shared_.end());
        tr.previous_private.insert(tr.previous_private.end(),p.private_.begin(),p.private_.end());
        throw;
    }
}
} // namespace
