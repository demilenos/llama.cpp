// SPDX-License-Identifier: MIT
#include "level4_contract.hpp"
#include "expert_tiles_reference.hpp"
#include <iostream>
#include <set>
using namespace maple_w2::level4; using maple_w2::expert_tile_capacity;
static size_t checks=0;void need(bool b){++checks;if(!b)throw std::runtime_error("contract assertion "+std::to_string(checks));}
template<class F>void rejects(F f){bool threw=false;try{f();}catch(const std::exception&){threw=true;}need(threw);}
int main()try{
 for(auto entry:{Entry::bootstrap,Entry::advance,Entry::terminal})for(auto q:{1u,13u,64u,184u,2048u})for(auto g:{1u,2u,4u})for(auto d:{1u,2u,4u}){
  Config c;c.entry=entry;c.tokens=q;auto p=make_plan(c,4,2,4,4,g,d);size_t last=0;
  std::vector<Span>s;for(auto&kv:p.spans)s.push_back(kv.second);std::sort(s.begin(),s.end(),[](auto a,auto b){return a.offset<b.offset;});
  for(auto x:s){if(!x.count)continue;need(x.offset%64==0);need(x.offset>=last);last=x.offset+x.bytes();need(last<=p.bytes);}need(p.bytes%64==0);  if(has_post(entry)) {
   const size_t jobs=size_t(q)*c.topk,cap4=expert_tile_capacity(jobs,c.experts,4),cap8=expert_tile_capacity(jobs,c.experts,8);
   need(p.at("gemm_tiles.expert").count==cap4);
   need(p.at("gemm_tiles.begin").count==cap4&&p.at("gemm_tiles.rows").count==cap4);
   need(p.at("gemm_tiles.count").count==1&&p.at("gemm_tiles.invalid").count==c.experts+1&&cap4>=cap8);
  }
 }
 float dummy=0;for(uint32_t q:{1u,3u,13u})for(uint32_t heads:{1u,4u,16u}) {
  uint32_t d=128,w=heads*d;for(bool hm:{false,true}){WriteView v{&dummy,size_t(q)*w,d,hm?d:w,hm?size_t(q)*d:d,1};check_write(v,q,w);
   std::set<size_t>addr;for(uint32_t t=0;t<q;++t)for(uint32_t x=0;x<w;++x)need(addr.insert(offset(v,t,x)).second);need(*addr.rbegin()+1==v.elements);}
 }
 rejects([&]{Config c;c.tokens=2049;make_plan(c);});rejects([&]{Config c;c.topk=9;make_plan(c);});
 rejects([&]{Config c;c.ffn_epsilon=0;make_plan(c);});rejects([&]{make_plan(Config{},3);});
 rejects([&]{plus_size(SIZE_MAX,1);});rejects([&]{WriteView v{&dummy,1024,128,1,1,1};check_write(v,2,256);});
 {
  std::vector<float> fused(3*3072);WriteView q{fused.data(),fused.size(),128,3072,128,1};
  WriteView k{fused.data()+2048,fused.size()-2048,128,3072,128,1};
  WriteView v{fused.data()+2560,fused.size()-2560,128,3072,128,1};
  need(disjoint_token_slices(q,2048,k,512));need(disjoint_token_slices(q,2048,v,512));need(disjoint_token_slices(k,512,v,512));
  k.data=fused.data()+2047;need(!disjoint_token_slices(q,2048,k,512));
 }
 GraphSemantics s;s.raw_fa_before_o=s.raw_qkv_before_qk_norm=s.sequential_residual=s.maple_clamped_swiglu=s.weights_validated=s.same_device_context=true;
 need(!reject_graph(Config{},s));for(auto bit:{&GraphSemantics::active_lora,&GraphSemantics::control_vector,&GraphSemantics::projection_bias,&GraphSemantics::extra_scale_or_rotation,&GraphSemantics::graph_has_external_interior_consumers,&GraphSemantics::final_row_selection_inside}){auto z=s;z.*bit=true;need(reject_graph(Config{},z));}
 for(unsigned e:{1u,8u,256u}){std::vector<float>logits(e,0),routes(std::min(e,8u));std::vector<int32_t>ids(routes.size());route_reference(logits.data(),e,uint32_t(ids.size()),ids.data(),routes.data());for(size_t i=0;i<ids.size();++i){need(ids[i]==int32_t(i));need(std::abs(routes[i]-1.f/ids.size())<1e-6f);}}
 std::cout<<"LEVEL4 CPU contract PASS checks="<<checks<<"; no GPU execution\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
