// SPDX-License-Identifier: MIT
// SYCL-only probe. Neither arm measures a Vulkan handoff.
#include "maple_level4.hpp"
#include "level4_reference.hpp"
#include "a8_contract_audit.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <cstring>
using namespace maple_w2;namespace l4=maple_w2::level4;
using Clock=std::chrono::steady_clock;
template<class T>struct Mem {
 sycl::queue&q;T*p=nullptr;size_t n;
 Mem(sycl::queue&qq,size_t nn):q(qq),n(nn){p=sycl::malloc_device<T>(n,q);if(!p)throw std::bad_alloc();}
 ~Mem(){if(p){try{q.wait_and_throw();sycl::free(p,q);}catch(...){}}}
 Mem(const Mem&)=delete;
 void upload(const std::vector<T>&a){if(a.size()!=n)throw std::invalid_argument("upload size");q.memcpy(p,a.data(),n*sizeof(T)).wait_and_throw();}
};
template<class T>std::vector<T>read(sycl::queue&q,const T*p,size_t n){std::vector<T>a(n);q.memcpy(a.data(),p,n*sizeof(T)).wait_and_throw();return a;}
static sycl::device device(const std::string&match) {
 for(const auto&p:sycl::platform::get_platforms())for(const auto&d:p.get_devices(sycl::info::device_type::gpu))
  if(d.get_backend()==sycl::backend::ext_oneapi_level_zero&&d.get_info<sycl::info::device::vendor_id>()==0x8086&&d.get_info<sycl::info::device::name>().find(match)!=std::string::npos)return d;
 throw std::runtime_error("No matching Intel LevelZero GPU; no fallback");
}
static bool ok(const Error&e){return e.finite&&e.nmse<1e-6&&e.max_abs_over_rms<.01;}
static void require(bool x,const std::string&m){if(!x)throw std::runtime_error(m);}
struct Weight {
 Shape s;std::vector<uint8_t>native;Mem<uint8_t>gpu;
 Weight(sycl::queue&q,Shape sh,unsigned seed):s(sh),native(l4::synthetic_weight(sh,seed)),gpu(q,nbytes(sh)){gpu.upload(repack_s2tile8(native,s));}
};
static std::vector<float>sample(const std::vector<float>&v,uint32_t width,const std::vector<uint32_t>&ts) {
 std::vector<float>r;for(auto t:ts)r.insert(r.end(),v.begin()+size_t(t)*width,v.begin()+size_t(t+1)*width);return r;
}
static QuantizedHost quant_sample(sycl::queue&q,A8Workspace w,uint32_t rows,uint32_t width,const std::vector<uint32_t>&ts) {
 auto a=read(q,w.q,size_t(rows)*width);auto d=read(q,w.scales,size_t(rows)*width/32);
 QuantizedHost r{uint32_t(ts.size()),width,32,{},{}};for(auto t:ts){r.q.insert(r.q.end(),a.begin()+size_t(t)*width,a.begin()+size_t(t+1)*width);r.scale.insert(r.scale.end(),d.begin()+size_t(t)*width/32,d.begin()+size_t(t+1)*width/32);}return r;
}
static void metric(std::ofstream&f,const char*n,const std::vector<float>&got,const std::vector<float>&ref){auto e=errors(got,ref);f<<n<<','<<e.nmse<<','<<e.max_abs<<','<<e.max_abs_over_rms<<','<<e.finite<<'\n';require(ok(e),std::string(n)+" numerical failure");}
int main(int argc,char**argv)try{
 uint32_t tokens=1,repeats=12;bool maple=false,dense_a8=false,ordered=false;std::string match="A750",out="build/level4-probe",schedule="direct",entry="advance";
 for(int i=1;i<argc;++i){std::string a=argv[i];auto val=[&](){if(++i>=argc)throw std::invalid_argument("missing argument");return std::string(argv[i]);};
  if(a=="--tokens")tokens=uint32_t(std::stoul(val()));else if(a=="--repeats")repeats=uint32_t(std::stoul(val()));else if(a=="--device")match=val();else if(a=="--out")out=val();else if(a=="--schedule")schedule=val();else if(a=="--entry")entry=val();else if(a=="--maple")maple=true;else if(a=="--dense-a8")dense_a8=true;else if(a=="--in-order")ordered=true;else throw std::invalid_argument("unknown argument "+a);}
 require(repeats>0&&repeats<=1000,"invalid repeats");
 l4::Config c;if(!maple){c.hidden=c.attention=c.q_width=c.ffn=256;c.kv_width=128;c.experts=8;c.topk=2;}c.tokens=tokens;
 c.entry=entry=="bootstrap"?l4::Entry::bootstrap:entry=="terminal"?l4::Entry::terminal:l4::Entry::advance;
 require(entry=="bootstrap"||entry=="terminal"||entry=="advance","unknown entry");
 auto plan=l4::make_plan(c);auto d=device(match);auto async=[](sycl::exception_list es){for(auto e:es)std::rethrow_exception(e);};
 sycl::queue q=ordered?sycl::queue(d,async,sycl::property_list{sycl::property::queue::enable_profiling{},sycl::property::queue::in_order{}}):sycl::queue(d,async,sycl::property_list{sycl::property::queue::enable_profiling{}});
 run_s2s8_probe(q);std::filesystem::create_directories(out);
 std::cout<<"Level4 SYCL-only probe, synthetic weights/activations, NOT llama/Vulkan E2E\n"<<d.get_info<sycl::info::device::name>()<<" Q="<<tokens<<" arena="<<plan.bytes<<"\n";
 // malloc_device has device allocation alignment; bind_workspace checks >=64 alignment.
 Mem<uint8_t>arena(q,plan.bytes);auto ws=l4::bind_workspace(arena.p,arena.n,plan);
 Weight wo(q,{c.attention,c.hidden,1},1),wq(q,{c.hidden,c.q_width,1},2),wk(q,{c.hidden,c.kv_width,1},3),wv(q,{c.hidden,c.kv_width,1},4),wg(q,{c.hidden,c.ffn,c.experts},5),wu(q,{c.hidden,c.ffn,c.experts},6),wd(q,{c.ffn,c.hidden,c.experts},7);
 const size_t nx=size_t(tokens)*c.hidden,na=size_t(tokens)*c.attention;
 std::vector<float>xh(nx),ah(na),gamma(c.hidden,1.f),next(c.hidden,1.0625f),router(size_t(c.hidden)*c.experts);
 for(size_t i=0;i<nx;++i)xh[i]=float(int((i*17)%113)-56)/128.f;for(size_t i=0;i<na;++i)ah[i]=float(int((i*11)%107)-53)/128.f;
 for(size_t i=0;i<router.size();++i)router[i]=float(int((i*13)%97)-48)/4096.f;
 Mem<float>dx(q,nx),da(q,na),dg(q,c.hidden),dn(q,c.hidden),dr(q,router.size()),ho(q,nx),qo(q,size_t(tokens)*c.q_width),ko(q,size_t(tokens)*c.kv_width),vo(q,size_t(tokens)*c.kv_width);
 dx.upload(xh);da.upload(ah);dg.upload(gamma);dn.upload(next);dr.upload(router);
 l4::Weights wt{wo.gpu.p,wq.gpu.p,wk.gpu.p,wv.gpu.p,wg.gpu.p,wu.gpu.p,wd.gpu.p,dg.p,dn.p,dr.p};
 wt.current_layer=c.entry==l4::Entry::bootstrap?-1:0;wt.next_layer=c.entry==l4::Entry::terminal?-1:(c.entry==l4::Entry::bootstrap?0:1);
 l4::Bindings b{l4::contiguous_read(dx.p,tokens,c.hidden),l4::contiguous_read(da.p,tokens,c.attention),l4::contiguous_write(ho.p,tokens,c.hidden),l4::contiguous_write(qo.p,tokens,c.q_width),l4::contiguous_write(ko.p,tokens,c.kv_width),l4::contiguous_write(vo.p,tokens,c.kv_width)};
 l4::Options opt;if(dense_a8)opt.o_precision=opt.qkv_precision=l4::DensePrecision::a8;
 if(schedule=="grouped")opt.moe.schedule=MoeSchedule::grouped;else if(schedule=="auto"){opt.moe.schedule=MoeSchedule::auto_select;opt.moe.auto_policy={64,2};}else require(schedule=="direct","unknown schedule");
 // Check numerical stages on the exact GPU intermediates (4 sample tokens max).
 auto run=l4::enqueue(q,wt,b,opt,ws);run.done.wait_and_throw();
 auto flags=read(q,run.status,tokens);require(std::all_of(flags.begin(),flags.end(),[](int32_t x){return !x;}),"nonzero island status");
 std::vector<uint32_t>ts{0};if(tokens>1)ts.push_back(tokens-1);if(tokens>3){ts.push_back(tokens/2);ts.push_back(tokens/3);}
 std::ofstream numerical(std::filesystem::path(out)/"numerical.csv");numerical<<std::setprecision(12)<<"stage,nmse,max_abs,max_abs_over_rms,finite\n";
 auto cp=[&](const char*n,size_t nval){return read(q,ws.get<float>(n),nval);};
 if(l4::has_post(c.entry)) {
  auto og=cp("o",nx),fr=cp("ff_residual",nx),fn=cp("ff_norm",nx),mo=cp("moe_out",nx);
  std::vector<int32_t>zero(ts.size(),0);auto expected=dense_a8?reference_a8(wo.native,wo.s,quant_sample(q,ws.a8("o_a8"),tokens,c.attention,ts),zero,uint32_t(ts.size()),1,false):reference(wo.native,wo.s,sample(ah,c.attention,ts),zero,uint32_t(ts.size()),1,false,true);
  metric(numerical,"o_same_activation",sample(og,c.hidden,ts),expected);metric(numerical,"ffn_norm",fn,l4::norm_reference(xh,og,gamma,c.hidden,c.ffn_epsilon));
  auto log=cp("router_logits",size_t(tokens)*c.experts);metric(numerical,"router_logits",log,l4::router_logits_reference(fn,router,c.hidden,c.experts));
  auto ids=read(q,ws.get<int32_t>("ids"),size_t(tokens)*c.topk);auto routes=cp("routes",ids.size());
  for(uint32_t t=0;t<tokens;++t){int32_t ri[8];float rs[8];l4::route_reference(log.data()+size_t(t)*c.experts,c.experts,c.topk,ri,rs);for(uint32_t k=0;k<c.topk;++k){require(ri[k]==ids[size_t(t)*c.topk+k],"topk mismatch");require(std::abs(rs[k]-routes[size_t(t)*c.topk+k])<1e-5,"route score mismatch");}}
  auto gm=ws.moe();std::vector<int32_t>si;std::vector<uint32_t>selection;for(auto t:ts)for(uint32_t k=0;k<c.topk;++k){si.push_back(ids[size_t(t)*c.topk+k]);selection.push_back(t*c.topk+k);}
  auto iq=quant_sample(q,gm.input_a8,tokens,c.hidden,ts);auto ga=read(q,gm.gate,size_t(tokens)*c.topk*c.ffn),up=read(q,gm.up,ga.size()),hidden=read(q,gm.hidden,ga.size()),down=read(q,gm.down,size_t(tokens)*c.topk*c.hidden);
  metric(numerical,"gate_same_q8",sample(ga,c.ffn,selection),reference_a8(wg.native,wg.s,iq,si,uint32_t(ts.size()),c.topk,false));
  metric(numerical,"up_same_q8",sample(up,c.ffn,selection),reference_a8(wu.native,wu.s,iq,si,uint32_t(ts.size()),c.topk,false));
  metric(numerical,"swiglu",hidden,swiglu_reference(ga,up));
  metric(numerical,"down_same_q8",sample(down,c.hidden,selection),reference_a8(wd.native,wd.s,quant_sample(q,gm.hidden_a8,tokens*c.topk,c.ffn,selection),si,uint32_t(ts.size()),c.topk,true));
  metric(numerical,"weighted_sum",mo,weighted_sum_reference(down,routes,tokens,c.topk,c.hidden));
 }
 auto hidden=read(q,ho.p,nx);
 if(l4::has_qkv(c.entry)) {
  auto norm=cp("next_norm",nx);metric(numerical,"next_norm",norm,l4::norm_reference(hidden,{},next,c.hidden,c.next_epsilon));
  std::vector<int32_t>zero(ts.size(),0);QuantizedHost aq_{};if(dense_a8)aq_=quant_sample(q,ws.a8("qkv_a8"),tokens,c.hidden,ts);
  for(auto p:std::array<std::pair<const char*,Weight*>,3>{{{"q",&wq},{"k",&wk},{"v",&wv}}}){auto y=cp(p.first,size_t(tokens)*p.second->s.m);
   auto expected=dense_a8?reference_a8(p.second->native,p.second->s,aq_,zero,uint32_t(ts.size()),1,false):reference(p.second->native,p.second->s,sample(norm,c.hidden,ts),zero,uint32_t(ts.size()),1,false,true);
   metric(numerical,p.first,sample(y,p.second->s.m,ts),expected);}
 }
 // Ordered AB/BA: same mathematics, segmented arm deliberately host-waits per stage.
 // This tests orchestration only; it is NOT an interop speedup measurement.
 std::ofstream samples(std::filesystem::path(out)/"samples.csv"),stages(std::filesystem::path(out)/"stages.csv");
 samples<<"repeat,arm,wall_us,gpu_span_us,kernel_sum_us\n";stages<<"repeat,arm,index,stage,parents,start_ns,end_ns\n";
 samples<<std::setprecision(12);std::vector<float>reference_hidden,reference_q;
 for(uint32_t rep=0;rep<repeats+2;++rep)for(uint32_t arm=0;arm<2;++arm){bool split=((rep&1)?1-arm:arm)!=0;opt.diagnostic_stage_waits=split;
  auto t0=Clock::now();auto rr=l4::enqueue(q,wt,b,opt,ws);rr.done.wait_and_throw();auto t1=Clock::now();
  if(rep<2)continue;
  uint64_t first=UINT64_MAX,last=0,total=0;for(size_t i=0;i<rr.count;++i){auto a=rr.stages[i].event.get_profiling_info<sycl::info::event_profiling::command_start>(),z=rr.stages[i].event.get_profiling_info<sycl::info::event_profiling::command_end>();require(z>=a,"bad timestamp");first=std::min(first,a);last=std::max(last,z);total+=z-a;
   stages<<rep-2<<','<<(split?"segmented_host_waits":"level4_single_enqueue")<<','<<i<<','<<rr.stages[i].name<<','<<rr.stages[i].parents<<','<<a<<','<<z<<'\n';}
  samples<<rep-2<<','<<(split?"segmented_host_waits":"level4_single_enqueue")<<','<<std::chrono::duration<double,std::micro>(t1-t0).count()<<','<<double(last-first)/1000<<','<<double(total)/1000<<'\n';
  auto now=read(q,ho.p,nx);if(reference_hidden.empty())reference_hidden=now;require(std::memcmp(now.data(),reference_hidden.data(),nx*4)==0,"orchestration changed hidden bits");
  if(l4::has_qkv(c.entry)){auto qq=read(q,qo.p,size_t(tokens)*c.q_width);if(reference_q.empty())reference_q=qq;require(std::memcmp(qq.data(),reference_q.data(),qq.size()*4)==0,"orchestration changed Q bits");}
 }
 std::ofstream scope(std::filesystem::path(out)/"SCOPE.txt");scope<<"Level4 SYCL-only synthetic single-island probe\nNo Vulkan import, no ggml graph rewrite, no model-level quality gate.\nQ="<<tokens<<" hidden="<<c.hidden<<" ffn="<<c.ffn<<" experts="<<c.experts<<" topk="<<c.topk<<" entry="<<entry<<" dense="<<(dense_a8?"A8G32":"A16")<<"\nCPU projection samples="<<ts.size()<<"; orchestration bit compare=all hidden and Q outputs.\n";
 std::cout<<"LEVEL4_PROBE PASS (SYCL-only; not Vulkan/model E2E)\n";return 0;
}catch(const std::exception&e){std::cerr<<"LEVEL4_PROBE FAIL: "<<e.what()<<'\n';return 1;}
