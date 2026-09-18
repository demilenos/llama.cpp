// SPDX-License-Identifier: MIT
// Actual src/maple_level4.cpp lambda/DAG execution with ONE-LANE mock collectives.
// Existing DPAS and MoE replaced by explicit CPU reference stubs. NOT GPU evidence.
#include "level4_bridge.hpp"
#include "level4_reference.hpp"
#include <iostream>
#include <set>
#include <new>
using namespace maple_w2;namespace l4=maple_w2::level4;
namespace maple_w2 {
static sycl::event stub(sycl::queue&q,const std::vector<sycl::event>&d){return q.submit([&](sycl::handler&h){h.depends_on(d);});}
static std::vector<uint8_t> native(const uint8_t*p,Shape s,Layout l){std::vector<uint8_t>x(p,p+nbytes(s));return l==Layout::s2tile8?unpack_s2tile8(x,s):x;}
Run enqueue(sycl::queue&q,const DeviceProblem&p,Options o,float*,const std::vector<sycl::event>&d) {
 Run r;r.main=stub(q,d);r.done=r.main;if(o.split_k>1){r.has_reduce=true;r.done=stub(q,{r.main});}
 if(sycl::mock_execute){std::vector<float>x((const float*)p.x,(const float*)p.x+size_t(p.tokens)*p.shape.k);std::vector<int32_t>ids(p.ids,p.ids+size_t(p.tokens)*p.topk);
  auto a=reference(native(p.w0,p.shape,p.layout),p.shape,x,ids,p.tokens,p.topk,p.per_selection,true);std::copy(a.begin(),a.end(),p.y0);
  if(p.w1){a=reference(native(p.w1,p.shape,p.layout),p.shape,x,ids,p.tokens,p.topk,p.per_selection,true);std::copy(a.begin(),a.end(),p.y1);}}
 return r;
}
sycl::event enqueue_a8_quant(sycl::queue&q,const void*x,ActivationType,uint32_t rows,uint32_t k,uint32_t g,A8Workspace w,const std::vector<sycl::event>&d){
 auto e=stub(q,d);if(sycl::mock_execute){auto a=quantize_a8(std::vector<float>((const float*)x,(const float*)x+size_t(rows)*k),rows,k,g);std::copy(a.q.begin(),a.q.end(),w.q);std::copy(a.scale.begin(),a.scale.end(),w.scales);std::fill(w.invalid,w.invalid+a.scale.size(),0);}return e;
}
A8Run enqueue_a8(sycl::queue&q,const DeviceProblem&p,A8Options o,A8Workspace w,float*,const std::vector<sycl::event>&d) {
 A8Run r;r.main=stub(q,d);r.done=r.main;if(o.kernel.split_k>1){r.has_reduce=true;r.done=stub(q,{r.main});}
 if(sycl::mock_execute){size_t rows=size_t(p.tokens)*(p.per_selection?p.topk:1);QuantizedHost a{uint32_t(rows),p.shape.k,o.group,std::vector<int8_t>(w.q,w.q+rows*p.shape.k),std::vector<float>(w.scales,w.scales+rows*p.shape.k/o.group)};
  std::vector<int32_t>ids(p.ids,p.ids+size_t(p.tokens)*p.topk);auto y=reference_a8(native(p.w0,p.shape,p.layout),p.shape,a,ids,p.tokens,p.topk,p.per_selection);std::copy(y.begin(),y.end(),p.y0);
  if(p.w1){y=reference_a8(native(p.w1,p.shape,p.layout),p.shape,a,ids,p.tokens,p.topk,p.per_selection);std::copy(y.begin(),y.end(),p.y1);}}
 return r;
}
MoeRun enqueue_moe(sycl::queue&q,const MoeProblem&p,const MoeOptions&o,const MoeWorkspace&w,const std::vector<sycl::event>&d) {
 MoeRun r;r.add("CPU_STUB_moe_reference",stub(q,d));
 if(sycl::mock_execute){std::vector<float>x(p.x,p.x+size_t(p.tokens)*p.gate_shape.k);std::vector<int32_t>ids(p.ids,p.ids+size_t(p.tokens)*p.topk);std::vector<float>routes(p.routes,p.routes+ids.size());
  auto a=quantize_a8(x,p.tokens,p.gate_shape.k,32);std::copy(a.q.begin(),a.q.end(),w.input_a8.q);std::copy(a.scale.begin(),a.scale.end(),w.input_a8.scales);std::fill(w.input_a8.invalid,w.input_a8.invalid+a.scale.size(),0);
  auto gate=reference_a8(native(p.gate,p.gate_shape,p.layout),p.gate_shape,a,ids,p.tokens,p.topk,false),up=reference_a8(native(p.up,p.gate_shape,p.layout),p.gate_shape,a,ids,p.tokens,p.topk,false);
  std::copy(gate.begin(),gate.end(),w.gate);std::copy(up.begin(),up.end(),w.up);auto h=swiglu_reference(gate,up,o.clamp);std::copy(h.begin(),h.end(),w.hidden);std::fill(w.hidden_invalid,w.hidden_invalid+h.size(),0);
  auto aq=quantize_a8(h,uint32_t(ids.size()),p.gate_shape.m,32);std::copy(aq.q.begin(),aq.q.end(),w.hidden_a8.q);std::copy(aq.scale.begin(),aq.scale.end(),w.hidden_a8.scales);std::fill(w.hidden_a8.invalid,w.hidden_a8.invalid+aq.scale.size(),0);
  Shape ds{p.gate_shape.m,p.gate_shape.k,p.gate_shape.experts};auto down=reference_a8(native(p.down,ds,p.layout),ds,aq,ids,p.tokens,p.topk,true);std::copy(down.begin(),down.end(),w.down);
  auto y=weighted_sum_reference(down,routes,p.tokens,p.topk,p.gate_shape.k);std::copy(y.begin(),y.end(),p.y);std::fill(w.output_invalid,w.output_invalid+y.size(),0);}
 return r;
}
}
static size_t checks=0;void need(bool b){++checks;if(!b)throw std::runtime_error("Level4 host assertion "+std::to_string(checks));}
static bool path(const sycl::queue&q,int src,int dst){std::vector<int>todo{dst};std::set<int>seen;while(!todo.empty()){int x=todo.back();todo.pop_back();if(x==src)return true;if(x==0||!seen.insert(x).second)continue;for(auto e:q.graph.at(x-1))todo.push_back(e.number);}return false;}
struct Arena {void*p;size_t n;Arena(size_t sz):p(::operator new(sz,std::align_val_t(64))),n(sz){std::memset(p,0,sz);}~Arena(){::operator delete(p,std::align_val_t(64));}};
int main()try {
 for(auto mode:{l4::Entry::bootstrap,l4::Entry::advance,l4::Entry::terminal})for(auto precision:{l4::DensePrecision::a16,l4::DensePrecision::a8})for(uint32_t tokens:{1u,3u}) {
  l4::Config c{tokens,256,256,256,128,256,8,2,mode};auto plan=l4::make_plan(c);Arena mem(plan.bytes);auto ws=l4::bind_workspace(mem.p,mem.n,plan);
  if(l4::has_post(mode)){auto tile_ws=ws.moe().tiles;need(tile_ws.capacity==maple_w2::expert_tile_capacity(size_t(tokens)*c.topk,c.experts,4));need(tile_ws.capacity>=maple_w2::expert_tile_capacity(size_t(tokens)*c.topk,c.experts,8));need(tile_ws.expert_capacity==c.experts);}
  auto mk=[&](Shape s,unsigned seed){return repack_s2tile8(l4::synthetic_weight(s,seed),s);};
  auto wo=mk({256,256,1},1),wq=mk({256,256,1},2),wk=mk({256,128,1},3),wv=mk({256,128,1},4),wg=mk({256,256,8},5),wu=mk({256,256,8},6),wd=mk({256,256,8},7);
  std::vector<float>gamma(256,1.f),next(256,1.125f),router(8*256),x(size_t(tokens)*256),a(x.size());
  for(size_t i=0;i<x.size();++i){x[i]=float(int(i%47)-23)/64.f;a[i]=float(int(i%29)-14)/32.f;}
  for(size_t i=0;i<router.size();++i)router[i]=float(int((i*13)%59)-29)/2048.f;
  std::vector<float>ho(x.size(),-99),qo(x.size(),-99),ko(size_t(tokens)*128,-99),vo(ko.size(),-99);
  l4::Weights wt{wo.data(),wq.data(),wk.data(),wv.data(),wg.data(),wu.data(),wd.data(),gamma.data(),next.data(),router.data()};
  wt.current_layer=mode==l4::Entry::bootstrap?-1:0;wt.next_layer=mode==l4::Entry::terminal?-1:(mode==l4::Entry::bootstrap?0:1);
  l4::Bindings b{l4::contiguous_read(x.data(),tokens,256),l4::contiguous_read(a.data(),tokens,256),l4::contiguous_write(ho.data(),tokens,256),l4::contiguous_write(qo.data(),tokens,256),l4::contiguous_write(ko.data(),tokens,128),l4::contiguous_write(vo.data(),tokens,128)};
  // Output to head-major storage, not just contiguous token-major tensors.
  b.q.head_dim=64;b.q.head_stride=size_t(tokens)*64;b.q.token_stride=64;
  l4::Options o;o.o_precision=o.qkv_precision=precision;
  sycl::queue q;sycl::mock_execute=true;auto root=q.submit([](sycl::handler&){});sycl::mock_waits=0;
  auto r=l4::enqueue(q,wt,b,o,ws,{root});need(sycl::mock_waits==0);
  for(size_t j=0;j<r.count;++j){need(path(q,root.number,r.stages[j].event.number));need(path(q,r.stages[j].event.number,r.done.number));for(size_t k=0;k<j;++k)if((r.stages[j].parents>>k)&1)need(path(q,r.stages[k].event.number,r.stages[j].event.number));}
  need(r.count<56);for(auto v:ho)need(std::isfinite(v));for(uint32_t t=0;t<tokens;++t)need(r.status[t]==0);
  if(l4::has_qkv(mode)) {auto expected=l4::norm_reference(ho,{},next,256,c.next_epsilon);std::vector<float>actual(ws.get<float>("next_norm"),ws.get<float>("next_norm")+x.size());need(errors(actual,expected).nmse<1e-11);
   for(uint32_t t=0;t<tokens;++t)for(uint32_t k=0;k<256;++k)need(qo[l4::offset(b.q,t,k)]==ws.get<float>("q")[size_t(t)*256+k]);}
  if(l4::has_post(mode)){auto norm=std::vector<float>(ws.get<float>("ff_norm"),ws.get<float>("ff_norm")+x.size());auto logits=l4::router_logits_reference(norm,router,256,8);auto actual=std::vector<float>(ws.get<float>("router_logits"),ws.get<float>("router_logits")+tokens*8);need(errors(actual,logits).nmse<1e-10);
   for(uint32_t t=0;t<tokens;++t){int32_t ids[2];float sc[2];l4::route_reference(actual.data()+t*8,8,2,ids,sc);for(unsigned k=0;k<2;++k){need(ws.get<int32_t>("ids")[t*2+k]==ids[k]);need(std::abs(ws.get<float>("routes")[t*2+k]-sc[k])<1e-7f);}}}
  // Interleaved QKV output views overlap in bounding ranges but not in elements.
  if(l4::has_qkv(mode)) {
    std::vector<float> combined(size_t(tokens)*512,-77);auto ib=b;
    ib.q={combined.data(),combined.size(),64,512,64,1};ib.k={combined.data()+256,combined.size()-256,64,512,64,1};ib.v={combined.data()+384,combined.size()-384,64,512,64,1};
    auto ir=l4::enqueue(q,wt,ib,o,ws);for(uint32_t t=0;t<tokens;++t)for(uint32_t j=0;j<256;++j)need(combined[size_t(t)*512+j]==ws.get<float>("q")[size_t(t)*256+j]);
  }
  // Bridge has exactly two external callbacks, one SYCL tail wait, no HOST tensor copies.
  l4::GraphSemantics sem;sem.raw_fa_before_o=sem.raw_qkv_before_qk_norm=sem.sequential_residual=sem.maple_clamped_swiglu=sem.weights_validated=sem.same_device_context=true;
  int enter=0,exit=0,poison=0;l4::VulkanHandoff hand{[&]{++enter;},[&]{++exit;},[&]{++poison;},std::make_shared<int>(1),sem};l4::BridgeCounters count;sycl::mock_waits=0;
  l4::execute_host_fenced(q,wt,b,o,ws,hand,count);need(enter==1&&exit==1&&poison==0&&sycl::mock_waits==1);need(count.inbound==1&&count.outbound==1&&count.bytes_h2d==0&&count.bytes_d2h==0);
  auto invalid=hand;invalid.semantics.control_vector=true;size_t old=q.graph.size();bool threw=false;try{l4::execute_host_fenced(q,wt,b,o,ws,invalid,count);}catch(...){threw=true;}need(threw&&q.graph.size()==old);
  invalid=hand;invalid.acquire_after_sycl=[](){throw std::runtime_error("mock acquire failed");};threw=false;try{l4::execute_host_fenced(q,wt,b,o,ws,invalid,count);}catch(...){threw=true;}need(threw&&poison==1);
  // Wrong metadata rejected BEFORE any submission.
  auto bad=b;bad.hidden.elements=1;old=q.graph.size();threw=false;try{l4::enqueue(q,wt,bad,o,ws);}catch(...){threw=true;}need(threw&&q.graph.size()==old);
  if(l4::has_qkv(mode)){bad=b;bad.k.data=bad.v.data;old=q.graph.size();threw=false;try{l4::enqueue(q,wt,bad,o,ws);}catch(...){threw=true;}need(threw&&q.graph.size()==old);}
 }
 // Actual production router selection lambda: ties/nonfinite status.
 {sycl::queue q;float log[8]={0,0,0,0,0,0,0,0},sc[8];int32_t ids[8],bad=0;l4::enqueue_router_select(q,log,ids,sc,&bad,1,8,8,{});for(int i=0;i<8;++i){need(ids[i]==i);need(sc[i]==.125f);}log[2]=std::numeric_limits<float>::quiet_NaN();l4::enqueue_router_select(q,log,ids,sc,&bad,1,8,8,{});need(bad!=0);for(auto i:ids)need(i==-1);}
 {Arena a(256);l4::ImportedAllocationView v{a.p,256,7,std::make_shared<int>(0)};need(v.at<float>(64,16,7)==(float*)((uint8_t*)a.p+64));bool t=false;try{v.at<float>(64,16,6);}catch(...){t=true;}need(t);t=false;try{v.at<float>(252,2,7);}catch(...){t=true;}need(t);}
 std::cout<<"LEVEL4 host/DAG + scalar-lambda tests PASS checks="<<checks<<"; DPAS/MoE CPU stubs; NOT SYCL device compilation\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
