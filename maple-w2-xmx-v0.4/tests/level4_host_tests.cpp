// SPDX-License-Identifier: MIT
// Actual src/maple_level4.cpp lambda/DAG execution with ONE-LANE mock collectives.
// Existing DPAS and MoE replaced by explicit CPU reference stubs. NOT GPU evidence.
#include "level4_bridge.hpp"
#include "level4_reference.hpp"
#include <iostream>
#include <set>
#include <new>
using namespace maple_w2;namespace l4=maple_w2::level4;
#include "level4_cpu_stubs.hpp"
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
  // Strict zero-copy boundary uses the SAME computation with direct imported-
  // equivalent pointers. This is scalar host execution, not external-memory proof.
  {
    const auto ref_h=ho;std::vector<float>ref_q(qo.size());
    if(l4::has_qkv(mode))for(uint32_t t=0;t<tokens;++t)for(uint32_t j=0;j<256;++j)ref_q[size_t(t)*256+j]=qo[l4::offset(b.q,t,j)];
    auto dp=l4::make_plan(c,1,1,1,1,1,1,true);Arena dm(dp.bytes);auto dw=l4::bind_workspace(dm.p,dm.n,dp);
    need(dp.bytes<plan.bytes);need(dw.get<float>("residual")==nullptr);need(dw.get<float>("hidden_out")==nullptr);
    auto db=b;db.q=l4::contiguous_write(qo.data(),tokens,256);auto dop=o;dop.boundary=l4::BoundaryPolicy::direct_required;
    sycl::mock_waits=0;auto direct=l4::enqueue(q,wt,db,dop,dw,{root});need(sycl::mock_waits==0);
    need(direct.boundary_traffic.bytes()==0);need(direct.boundary_traffic.direct_inputs==(l4::has_post(mode)?2u:1u));
    need(direct.boundary_traffic.direct_outputs==(l4::has_qkv(mode)?4u:1u));
    need(ho==ref_h);if(l4::has_qkv(mode))need(qo==ref_q);
    for(size_t j=0;j<direct.count;++j){std::string name=direct.stages[j].name;need(name.find("pack_")==std::string::npos);need(name!="export_hidden_raw_qkv");need(path(q,root.number,direct.stages[j].event.number));need(path(q,direct.stages[j].event.number,direct.done.number));}
    auto alias=db;alias.hidden.data=x.data();auto old=q.graph.size();bool threw=false;
    try{l4::enqueue(q,wt,alias,dop,dw);}catch(...){threw=true;}need(threw&&q.graph.size()==old);
    if(tokens>1&&l4::has_qkv(mode)){old=q.graph.size();threw=false;try{l4::enqueue(q,wt,b,dop,dw);}catch(...){threw=true;}need(threw&&q.graph.size()==old);}
    old=q.graph.size();threw=false;try{l4::enqueue(q,wt,db,o,dw);}catch(...){threw=true;}need(threw&&q.graph.size()==old);
    // Restore packed reference storage for following legacy tests.
    l4::enqueue(q,wt,b,o,ws);
  }
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
