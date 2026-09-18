// SPDX-License-Identifier: MIT
// REAL Windows Vulkan -> exported allocation -> L0/SYCL Level4 -> Vulkan
// transfer-consumer probe. Uploads/readbacks below are DIAGNOSTICS outside the
// measured island, not the production zero-copy data path.
#include "level4_zerocopy_win32.hpp"
#include "level4_reference.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <algorithm>
using namespace maple_w2;namespace l4=maple_w2::level4;namespace zc=l4::zc;
namespace {
void ck(VkResult r,const char*m){if(r!=VK_SUCCESS)throw std::runtime_error(std::string(m)+" "+std::to_string(int(r)));}
void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
sycl::device select_device(const std::string&name){for(const auto&p:sycl::platform::get_platforms())for(auto d:p.get_devices(sycl::info::device_type::gpu))if(d.get_backend()==sycl::backend::ext_oneapi_level_zero&&d.get_info<sycl::info::device::vendor_id>()==0x8086&&d.get_info<sycl::info::device::name>().find(name)!=std::string::npos)return d;throw std::runtime_error("no matching LevelZero GPU; no OpenCL/CPU fallback");}
struct VulkanOwner {
 VkInstance instance=VK_NULL_HANDLE;VkPhysicalDevice physical=VK_NULL_HANDLE;VkDevice device=VK_NULL_HANDLE;VkQueue queue=VK_NULL_HANDLE;uint32_t family=0;
 std::shared_ptr<std::recursive_mutex> mutex=std::make_shared<std::recursive_mutex>();
 ~VulkanOwner(){if(device){(void)vkDeviceWaitIdle(device);vkDestroyDevice(device,nullptr);}if(instance)vkDestroyInstance(instance,nullptr);}
 static std::shared_ptr<VulkanOwner>create(const std::string&name,bool validation){
  auto v=std::make_shared<VulkanOwner>();VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="Maple Level4 zero-copy probe";app.apiVersion=VK_API_VERSION_1_1;
  const char*layers[]={"VK_LAYER_KHRONOS_validation"};VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ii.pApplicationInfo=&app;if(validation){ii.enabledLayerCount=1;ii.ppEnabledLayerNames=layers;}
  ck(vkCreateInstance(&ii,nullptr,&v->instance),"vkCreateInstance");uint32_t n=0;ck(vkEnumeratePhysicalDevices(v->instance,&n,nullptr),"enumerate");std::vector<VkPhysicalDevice>devices(n);ck(vkEnumeratePhysicalDevices(v->instance,&n,devices.data()),"enumerate");
  for(auto p:devices){VkPhysicalDeviceProperties prop{};vkGetPhysicalDeviceProperties(p,&prop);if(prop.vendorID==0x8086&&std::string(prop.deviceName).find(name)!=std::string::npos){need(!v->physical,"ambiguous Vulkan device name; select a unique adapter");v->physical=p;}}
  need(v->physical!=VK_NULL_HANDLE,"no matching Vulkan adapter");uint32_t nq=0;vkGetPhysicalDeviceQueueFamilyProperties(v->physical,&nq,nullptr);std::vector<VkQueueFamilyProperties>qs(nq);vkGetPhysicalDeviceQueueFamilyProperties(v->physical,&nq,qs.data());v->family=UINT32_MAX;
  for(uint32_t i=0;i<nq;++i)if(qs[i].queueFlags&VK_QUEUE_COMPUTE_BIT){v->family=i;break;}need(v->family!=UINT32_MAX,"no compute queue");
  float priority=1.f;VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qi.queueFamilyIndex=v->family;qi.queueCount=1;qi.pQueuePriorities=&priority;
  const char*exts[]={VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME};VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};di.queueCreateInfoCount=1;di.pQueueCreateInfos=&qi;di.enabledExtensionCount=1;di.ppEnabledExtensionNames=exts;
  ck(vkCreateDevice(v->physical,&di,nullptr,&v->device),"vkCreateDevice(external-memory)");vkGetDeviceQueue(v->device,v->family,0,&v->queue);return v;
 }
};
struct Cmd {
 std::shared_ptr<VulkanOwner>v;VkCommandPool pool=VK_NULL_HANDLE;VkCommandBuffer cmd=VK_NULL_HANDLE;VkFence fence=VK_NULL_HANDLE;bool pending=false;
 explicit Cmd(std::shared_ptr<VulkanOwner>x):v(std::move(x)){
  VkCommandPoolCreateInfo p{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};p.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;p.queueFamilyIndex=v->family;ck(vkCreateCommandPool(v->device,&p,nullptr,&pool),"pool");
  VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};a.commandPool=pool;a.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;a.commandBufferCount=1;ck(vkAllocateCommandBuffers(v->device,&a,&cmd),"cmd alloc");VkFenceCreateInfo f{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};ck(vkCreateFence(v->device,&f,nullptr,&fence),"fence");
 }
 ~Cmd(){if(v->device)(void)vkQueueWaitIdle(v->queue);if(fence)vkDestroyFence(v->device,fence,nullptr);if(pool)vkDestroyCommandPool(v->device,pool,nullptr);}
 void wait(){if(pending){ck(vkWaitForFences(v->device,1,&fence,VK_TRUE,60'000'000'000ull),"diagnostic fence");pending=false;}}
 void begin(){wait();ck(vkResetCommandBuffer(cmd,0),"reset");VkCommandBufferBeginInfo b{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};b.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;ck(vkBeginCommandBuffer(cmd,&b),"begin");}
 void submit(bool wait_now){ck(vkEndCommandBuffer(cmd),"end");ck(vkResetFences(v->device,1,&fence),"fence reset");VkSubmitInfo s{VK_STRUCTURE_TYPE_SUBMIT_INFO};s.commandBufferCount=1;s.pCommandBuffers=&cmd;{std::lock_guard<std::recursive_mutex>l(*v->mutex);ck(vkQueueSubmit(v->queue,1,&s,fence),"diagnostic submit");}pending=true;if(wait_now)wait();}
};
struct Readback {
 std::shared_ptr<VulkanOwner>v;VkBuffer buffer=VK_NULL_HANDLE;VkDeviceMemory memory=VK_NULL_HANDLE;void*mapped=nullptr;bool coherent=false;size_t bytes;
 Readback(std::shared_ptr<VulkanOwner>x,size_t size):v(std::move(x)),bytes(size){
  VkBufferCreateInfo b{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};b.size=bytes;b.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT;b.sharingMode=VK_SHARING_MODE_EXCLUSIVE;ck(vkCreateBuffer(v->device,&b,nullptr,&buffer),"readback buffer");VkMemoryRequirements req{};vkGetBufferMemoryRequirements(v->device,buffer,&req);VkPhysicalDeviceMemoryProperties p{};vkGetPhysicalDeviceMemoryProperties(v->physical,&p);uint32_t type=UINT32_MAX;
  for(uint32_t i=0;i<p.memoryTypeCount;++i)if((req.memoryTypeBits&(1u<<i))&&(p.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)){type=i;if(p.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)break;}
  need(type!=UINT32_MAX,"no readback memory");coherent=p.memoryTypes[type].propertyFlags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  VkMemoryAllocateInfo a{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};a.allocationSize=req.size;a.memoryTypeIndex=type;ck(vkAllocateMemory(v->device,&a,nullptr,&memory),"readback alloc");ck(vkBindBufferMemory(v->device,buffer,memory,0),"readback bind");ck(vkMapMemory(v->device,memory,0,VK_WHOLE_SIZE,0,&mapped),"readback map");
 }
 ~Readback(){if(v->device)(void)vkQueueWaitIdle(v->queue);if(mapped)vkUnmapMemory(v->device,memory);if(buffer)vkDestroyBuffer(v->device,buffer,nullptr);if(memory)vkFreeMemory(v->device,memory,nullptr);}
 void invalidate(){if(!coherent){VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};r.memory=memory;r.size=VK_WHOLE_SIZE;ck(vkInvalidateMappedMemoryRanges(v->device,1,&r),"invalidate");}}
};
struct Usm {
 sycl::queue q;void*p=nullptr;size_t bytes;
 Usm(sycl::queue qq,size_t n):q(std::move(qq)),bytes(n){p=sycl::malloc_device<uint8_t>(n,q);if(!p)throw std::bad_alloc();}
 ~Usm(){if(p){try{q.wait_and_throw();sycl::free(p,q);}catch(...){std::cerr<<"probe USM teardown failed\n";}}}
};
uint32_t bits(float v){uint32_t b;std::memcpy(&b,&v,4);return b;}
}
int main(int argc,char**argv)try{
 std::string match="A750",out="build/level4-zero-copy",entry="advance",schedule="direct";uint32_t tokens=1,repeats=6;bool maple=false,a8=false,validation=false;
 for(int i=1;i<argc;++i){std::string a=argv[i];auto val=[&](){if(++i>=argc)throw std::invalid_argument("missing option");return std::string(argv[i]);};
  if(a=="--device")match=val();else if(a=="--out")out=val();else if(a=="--tokens")tokens=uint32_t(std::stoul(val()));else if(a=="--repeats")repeats=uint32_t(std::stoul(val()));else if(a=="--entry")entry=val();else if(a=="--schedule")schedule=val();else if(a=="--maple")maple=true;else if(a=="--dense-a8")a8=true;else if(a=="--validation")validation=true;else throw std::invalid_argument("unknown argument: "+a);}
 need(repeats&&repeats<=1000,"invalid repeats");l4::Config c;if(!maple){c.hidden=c.attention=c.q_width=c.ffn=256;c.kv_width=128;c.experts=8;c.topk=2;}c.tokens=tokens;
 need(entry=="advance"||entry=="bootstrap"||entry=="terminal","invalid entry");c.entry=entry=="advance"?l4::Entry::advance:entry=="bootstrap"?l4::Entry::bootstrap:l4::Entry::terminal;
 auto direct_plan=l4::make_plan(c,1,1,1,1,1,1,true),packed_plan=l4::make_plan(c);
 auto d=select_device(match);auto async=[](sycl::exception_list e){for(auto x:e)std::rethrow_exception(x);};sycl::context sc(d);sycl::queue q(sc,d,async,sycl::property_list{sycl::property::queue::enable_profiling{}});
 run_s2s8_probe(q);auto vk=VulkanOwner::create(match,validation);auto interop=zc::Context::create(q,vk->physical,vk->device,vk->queue,vk->family,vk->mutex,vk);
 std::filesystem::create_directories(out);std::vector<std::shared_ptr<Usm>>storage;std::vector<zc::PrivateRegion>regions;
 auto allocate=[&](size_t n){auto a=std::make_shared<Usm>(q,n);storage.push_back(a);regions.push_back({a->p,n,a});return a->p;};
 auto upload=[&](const void*src,size_t n){auto p=allocate(n);q.memcpy(p,src,n).wait_and_throw();return p;};
 auto weight=[&](Shape s,uint32_t seed){auto bytes=repack_s2tile8(l4::synthetic_weight(s,seed),s);return static_cast<const uint8_t*>(upload(bytes.data(),bytes.size()));};
 l4::Weights wt;
 if(l4::has_post(c.entry)){wt.o=weight({c.attention,c.hidden,1},1);wt.gate=weight({c.hidden,c.ffn,c.experts},5);wt.up=weight({c.hidden,c.ffn,c.experts},6);wt.down=weight({c.ffn,c.hidden,c.experts},7);}
 if(l4::has_qkv(c.entry)){wt.q=weight({c.hidden,c.q_width,1},2);wt.k=weight({c.hidden,c.kv_width,1},3);wt.v=weight({c.hidden,c.kv_width,1},4);}
 std::vector<float>gamma(c.hidden,1.f),next(c.hidden,1.0625f),router(size_t(c.hidden)*c.experts);for(size_t i=0;i<router.size();++i)router[i]=float(int((i*13)%97)-48)/4096.f;
 if(l4::has_post(c.entry)){wt.ffn_norm=static_cast<float*>(upload(gamma.data(),gamma.size()*4));wt.router=static_cast<float*>(upload(router.data(),router.size()*4));}
 if(l4::has_qkv(c.entry))wt.next_attn_norm=static_cast<float*>(upload(next.data(),next.size()*4));
 wt.current_layer=c.entry==l4::Entry::bootstrap?-1:0;wt.next_layer=c.entry==l4::Entry::terminal?-1:(c.entry==l4::Entry::bootstrap?0:1);
 auto ws=l4::bind_workspace(allocate(direct_plan.bytes),direct_plan.bytes,direct_plan);auto pws=l4::bind_workspace(allocate(packed_plan.bytes),packed_plan.bytes,packed_plan);
 // Nonzero tensor offsets, one exported arena, disjoint residual ping-pong.
 size_t total=64;auto slot=[&](uint32_t width){auto offset=total;total=l4::aligned(total+size_t(tokens)*width*4);return offset;};
 auto ox=slot(c.hidden),oa=slot(c.attention),oh=slot(c.hidden),oq=slot(c.q_width),ok=slot(c.kv_width),ov=slot(c.kv_width);auto io=interop->create_buffer(total);
 auto read_view=[&](size_t off,uint32_t width){return l4::contiguous_read(io->at<float>(off,size_t(tokens)*width),tokens,width);};auto write_view=[&](size_t off,uint32_t width){return l4::contiguous_write(io->at<float>(off,size_t(tokens)*width),tokens,width);};
 l4::Bindings bindings{read_view(ox,c.hidden),read_view(oa,c.attention),write_view(oh,c.hidden),write_view(oq,c.q_width),write_view(ok,c.kv_width),write_view(ov,c.kv_width)};
 // Reference inputs exactly match Vulkan fill patterns; no host read of imported pointer.
 std::vector<float>x(size_t(tokens)*c.hidden),a(size_t(tokens)*c.attention);for(uint32_t t=0;t<tokens;++t){std::fill(x.begin()+size_t(t)*c.hidden,x.begin()+size_t(t+1)*c.hidden,float(int(t%17)-8)/128);std::fill(a.begin()+size_t(t)*c.attention,a.begin()+size_t(t+1)*c.attention,float(int(t%13)-6)/128);}
 l4::Bindings pb{l4::contiguous_read(static_cast<float*>(upload(x.data(),x.size()*4)),tokens,c.hidden),l4::contiguous_read(static_cast<float*>(upload(a.data(),a.size()*4)),tokens,c.attention)};
 pb.hidden=l4::contiguous_write(static_cast<float*>(allocate(size_t(tokens)*c.hidden*4)),tokens,c.hidden);pb.q=l4::contiguous_write(static_cast<float*>(allocate(size_t(tokens)*c.q_width*4)),tokens,c.q_width);pb.k=l4::contiguous_write(static_cast<float*>(allocate(size_t(tokens)*c.kv_width*4)),tokens,c.kv_width);pb.v=l4::contiguous_write(static_cast<float*>(allocate(size_t(tokens)*c.kv_width*4)),tokens,c.kv_width);
 l4::Options opt;if(a8)opt.o_precision=opt.qkv_precision=l4::DensePrecision::a8;
 if(schedule=="grouped")opt.moe.schedule=MoeSchedule::grouped;else if(schedule=="auto"){opt.moe.schedule=MoeSchedule::auto_select;opt.moe.auto_policy={64,2};}else need(schedule=="direct","unknown schedule");
 auto reference=l4::enqueue(q,wt,pb,opt,pws);reference.done.wait_and_throw();
 struct Output{const char*name;size_t offset;uint32_t width;float*baseline;};std::vector<Output>outputs{{"hidden",oh,c.hidden,pb.hidden.data}};
 if(l4::has_qkv(c.entry)){outputs.push_back({"q",oq,c.q_width,pb.q.data});outputs.push_back({"k",ok,c.kv_width,pb.k.data});outputs.push_back({"v",ov,c.kv_width,pb.v.data});}
 std::vector<std::vector<float>>expected;for(const auto&o:outputs){std::vector<float>v(size_t(tokens)*o.width);q.memcpy(v.data(),o.baseline,v.size()*4).wait_and_throw();expected.push_back(std::move(v));}
 l4::GraphSemantics sem;sem.raw_fa_before_o=sem.raw_qkv_before_qk_norm=sem.sequential_residual=sem.maple_clamped_swiglu=sem.weights_validated=true;
 auto prepared=interop->prepare(wt,bindings,opt,ws,sem,{io},regions);const auto cold=interop->counters();
 Cmd cmd(vk);Readback rb(vk,total);std::ofstream samples(std::filesystem::path(out)/"samples.csv"),numerical(std::filesystem::path(out)/"numerical.csv");
 need(bool(samples)&&bool(numerical),"cannot open output CSVs");
 samples<<"repeat,inbound_host_us,sycl_host_us,acquire_submit_us,total_host_us,execute_wall_us,gpu_sycl_span_us,boundary_d2d_bytes,imports,exports\n"<<std::setprecision(12);
 numerical<<"repeat,output,max_abs,nmse,max_abs_over_rms,finite,bitwise_equal\n"<<std::setprecision(12);
 for(uint32_t rep=0;rep<repeats+1;++rep){
  cmd.begin();for(uint32_t t=0;t<tokens;++t){vkCmdFillBuffer(cmd.cmd,io->vk_buffer(),ox+size_t(t)*c.hidden*4,size_t(c.hidden)*4,bits(x[size_t(t)*c.hidden]));if(l4::has_post(c.entry))vkCmdFillBuffer(cmd.cmd,io->vk_buffer(),oa+size_t(t)*c.attention*4,size_t(c.attention)*4,bits(a[size_t(t)*c.attention]));}cmd.submit(false);
  const auto call_start=std::chrono::steady_clock::now();auto result=interop->execute(prepared);const auto call_end=std::chrono::steady_clock::now();const double execute_wall=std::chrono::duration<double,std::micro>(call_end-call_start).count();const auto stats=interop->counters();need(stats.imports==cold.imports&&stats.exports==cold.exports&&stats.allocations==cold.allocations,"hot-path reallocation/reimport");need(!result.run.boundary_traffic.bytes(),"unexpected device boundary copy");
  // Vulkan, not SYCL, consumes the returned outputs. Host readback solely checks
  // cross-API visibility and is outside result.timing.total_host_us.
  cmd.begin();for(const auto&o:outputs){VkBufferCopy copy{o.offset,o.offset,size_t(tokens)*o.width*4};vkCmdCopyBuffer(cmd.cmd,io->vk_buffer(),rb.buffer,1,&copy);}
  VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;barrier.buffer=rb.buffer;barrier.size=total;
  vkCmdPipelineBarrier(cmd.cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&barrier,0,nullptr);cmd.submit(true);rb.invalidate();
  for(size_t i=0;i<outputs.size();++i){const auto&o=outputs[i];std::vector<float>got(size_t(tokens)*o.width);std::memcpy(got.data(),static_cast<uint8_t*>(rb.mapped)+o.offset,got.size()*4);auto error=errors(got,expected[i]);const bool equal=std::memcmp(got.data(),expected[i].data(),got.size()*4)==0;numerical<<rep<<','<<o.name<<','<<error.max_abs<<','<<error.nmse<<','<<error.max_abs_over_rms<<','<<error.finite<<','<<equal<<'\n';need(error.finite&&error.nmse<1e-6&&error.max_abs_over_rms<.01,"cross-API output comparison failed");}
  std::vector<int32_t>flags(tokens);q.memcpy(flags.data(),result.run.status,size_t(tokens)*4).wait_and_throw();need(std::all_of(flags.begin(),flags.end(),[](int32_t s){return !s;}),"nonzero Level4 status");
  uint64_t first=UINT64_MAX,last=0;for(size_t i=0;i<result.run.count;++i){auto e=result.run.stages[i].event;first=std::min(first,e.get_profiling_info<sycl::info::event_profiling::command_start>());last=std::max(last,e.get_profiling_info<sycl::info::event_profiling::command_end>());}
  if(rep)samples<<rep-1<<','<<result.timing.inbound_host_us<<','<<result.timing.sycl_host_us<<','<<result.timing.acquire_submit_us<<','<<result.timing.total_host_us<<','<<execute_wall<<','<<double(last-first)/1000<<','<<result.run.boundary_traffic.bytes()<<','<<stats.imports<<','<<stats.exports<<'\n';
 }
 samples.flush();numerical.flush();need(bool(samples)&&bool(numerical),"failed writing result CSVs");
 interop->drain();auto stats=interop->counters();std::ofstream scope(std::filesystem::path(out)/"SCOPE.txt");
 scope<<"Real Vulkan->LevelZero/SYCL Level4->Vulkan transfer consumer; single matching device verified by LUID.\nSynthetic weights + per-token uniform GPU-filled input. NOT model quality / representative MoE performance.\nDiagnostic baseline uploads and Vulkan readbacks excluded from island timing. Host-fenced, not wait-free.\nDirect boundary bytes=0; native imports="<<stats.imports<<" exports="<<stats.exports<<" allocations="<<stats.allocations<<"\n"<<"direct_scratch="<<direct_plan.bytes<<" packed_scratch="<<packed_plan.bytes<<" Q="<<tokens<<" entry="<<entry<<"\n";
 std::cout<<"LEVEL4_ZERO_COPY_PROBE PASS: actual Vulkan->L0->Vulkan roundtrip; NOT llama-server E2E\n";return 0;
}catch(const std::exception&e){std::cerr<<"LEVEL4_ZERO_COPY_PROBE FAIL: "<<e.what()<<'\n';return 1;}
