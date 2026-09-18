// SPDX-License-Identifier: MIT
// Shared HOST-ONLY DPAS/MoE reference stubs. Never linked into a production target.
#pragma once
#include "level4_reference.hpp"
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
