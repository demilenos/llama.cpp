// SPDX-License-Identifier: MIT
#include "maple_level4.hpp"
#include "dg2_validation.hpp"
#include <cstring>
#include <limits>
namespace maple_w2::level4 {
namespace {
#ifdef MAPLE_LEVEL4_SCALAR_TEST
// Host test of the actual lambda bodies with one-lane collective doubles.
// NEVER define this on a SYCL/GPU target. It is not a GPU compilation substitute.
constexpr uint32_t WG=1;
#else
constexpr uint32_t WG=64;
#endif
class PackInput;class PackOutput;class InitStatus;class ResidualNorm;
class RouterLogits;class RouterSelect;class FinalStatus;class CopyBootstrap;
A8Workspace aq(const Workspace&w,const char*n){return w.a8(n);}
ExpertTokenTileView tiles(const Workspace&w,const char*n,uint32_t t) {
    return {w.get<int32_t>(std::string(n)+".first"),w.get<int32_t>(std::string(n)+".length"),
        w.get<int32_t>(std::string(n)+".expert"),w.get<int32_t>(std::string(n)+".total"),
        w.plan.at(std::string(n)+".first").count,t};
}
void check_precision(DensePrecision p) {
    if(p!=DensePrecision::a16&&p!=DensePrecision::a8)throw std::invalid_argument("unknown dense precision");
}
void check_kernel(maple_w2::Options o,uint32_t k,uint32_t expected) {
    check_split(k,o.split_k);if(o.split_k!=expected||!o.local_size||o.local_size>32)
        throw std::invalid_argument("Level4 options mismatch workspace split/local");
}
struct AddressRange {uintptr_t begin;size_t bytes;};
bool overlaps(AddressRange a,AddressRange b) {
    if(a.bytes>UINTPTR_MAX-a.begin||b.bytes>UINTPTR_MAX-b.begin)throw std::invalid_argument("pointer range wraps");
    return a.bytes&&b.bytes&&a.begin<b.begin+b.bytes&&b.begin<a.begin+a.bytes;
}
AddressRange addr(const void*p,size_t n){return {reinterpret_cast<uintptr_t>(p),n};}
}
MoeWorkspace Workspace::moe()const {
    const auto&c=plan.config;MoeWorkspace m;
    m.gate=get<float>("gate");m.up=get<float>("up");m.hidden=get<float>("hidden");m.down=get<float>("down");
    m.gate_split=get<float>("gate_split");m.down_split=get<float>("down_split");m.input_a8=a8("input_a8");m.hidden_a8=a8("hidden_a8");
    m.gate_status=get<int32_t>("gate_status");m.down_status=get<int32_t>("down_status");
    m.hidden_invalid=get<int32_t>("hidden_invalid");m.output_invalid=get<int32_t>("output_invalid");
    m.grouping={get<int32_t>("counts"),get<int32_t>("offsets"),get<int32_t>("sorted_to_job"),size_t(c.tokens)*c.topk,c.experts};
    m.tiles={get<int32_t>("gemm_tiles.expert"),get<int32_t>("gemm_tiles.begin"),get<int32_t>("gemm_tiles.rows"),get<int32_t>("gemm_tiles.count"),get<int32_t>("gemm_tiles.invalid"),plan.at("gemm_tiles.expert").count,c.experts};
    m.gate_tiles=tiles(*this,"gate_tiles",plan.gate_tile);m.down_tiles=tiles(*this,"down_tiles",plan.down_tile);return m;
}
Workspace bind_workspace(void*p,size_t n,Plan plan) {
    if(!p||(reinterpret_cast<uintptr_t>(p)&63)||n<plan.bytes)throw std::invalid_argument("Level4 arena needs 64-byte alignment and plan.bytes capacity");
    // Reconstruct to reject caller-edited region offsets before any pointer is formed.
    const auto canonical=make_plan(plan.config,plan.gate_split,plan.down_split,plan.o_split,plan.qkv_split,plan.gate_tile,plan.down_tile);
    if(canonical.bytes!=plan.bytes||canonical.spans.size()!=plan.spans.size())throw std::invalid_argument("noncanonical Level4 plan");
    for(const auto&kv:canonical.spans){const auto&s=plan.at(kv.first);if(s.offset!=kv.second.offset||s.count!=kv.second.count||s.element_bytes!=kv.second.element_bytes)throw std::invalid_argument("modified Level4 plan");}
    return {static_cast<uint8_t*>(p),n,std::move(plan)};
}
uint64_t Run::add(const char*n,const sycl::event&e,uint64_t parents) {
    if(count>=stages.size())throw std::logic_error("Level4 event capacity");
    const auto bit=uint64_t(1)<<count;
    stages[count++]={n,e,parents};done=e;return bit;
}
void validate(const Weights&wt,const Bindings&b,const Options&o,const Workspace&w) {
    const auto&c=w.plan.config;check_config(c);
    if(c.entry==Entry::advance && (wt.current_layer<0||wt.current_layer==INT32_MAX||wt.next_layer!=wt.current_layer+1))
        throw std::invalid_argument("advance needs O/FFN layer L and norm/QKV layer L+1");
    if(c.entry==Entry::bootstrap && (wt.current_layer!=-1||wt.next_layer<0))
        throw std::invalid_argument("bootstrap binds only its target norm/QKV layer");
    if(c.entry==Entry::terminal && (wt.current_layer<0||wt.next_layer!=-1))
        throw std::invalid_argument("terminal must not bind a next layer");
    if(!w.base||(reinterpret_cast<uintptr_t>(w.base)&63)||w.capacity<w.plan.bytes)throw std::invalid_argument("bad Level4 arena");
    if(wt.dense_layout!=Layout::native_tq2&&wt.dense_layout!=Layout::s2tile8)throw std::invalid_argument("unsupported dense layout");
    check_precision(o.o_precision);check_precision(o.qkv_precision);
    const size_t residual_n=required_elements(b.residual,c.tokens,c.hidden);check_write(b.hidden,c.tokens,c.hidden);
    std::vector<AddressRange> outputs{addr(b.hidden.data,checked_mul(required_elements(b.hidden,c.tokens,c.hidden),4))};
    std::vector<std::pair<WriteView,uint32_t>> output_views{{b.hidden,c.hidden}};
    std::vector<AddressRange> readonly{addr(b.residual.data,checked_mul(residual_n,4))};
    if(has_post(c.entry)) {
        readonly.push_back(addr(b.attention.data,checked_mul(required_elements(b.attention,c.tokens,c.attention),4)));
        if(!wt.o||!wt.gate||!wt.up||!wt.down||!wt.ffn_norm||!wt.router)throw std::invalid_argument("missing current-layer O/FFN weights");
        if(wt.moe_layout!=Layout::s2tile8)throw std::invalid_argument("Level4 MoE uses validated v0.4 s2tile8 path");
        if(o.moe.path!=MoePath::a8_gluquant||o.moe.input_group!=32||o.moe.hidden_group!=32||o.moe.input_prequantized)
            throw std::invalid_argument("Level4 preserves W2A8 G32/H32 gluquant; external prequantized input not supported");
        if(o.moe.diagnostic_host_waits)throw std::invalid_argument("use Level4 diagnostic waits, not hidden MoE waits");
        check_kernel(o.o_kernel,c.attention,w.plan.o_split);check_kernel(o.moe.gate_kernel,c.hidden,w.plan.gate_split);check_kernel(o.moe.down_kernel,c.ffn,w.plan.down_split);
        if(o.moe.gate_tokens_per_tile!=w.plan.gate_tile||o.moe.down_tokens_per_tile!=w.plan.down_tile)
            throw std::invalid_argument("Level4 token tile differs from workspace plan");
        if(!std::isfinite(o.moe.clamp)||o.moe.clamp!=7.f)throw std::invalid_argument("this Level4 graph contract is Maple clamp=7");
        // Resolve dispatch before submissions; catches invalid options and policy.
        (void)choose_moe_schedule(o.moe.schedule,o.moe.expert_grouping,c.tokens,c.topk,c.experts,true,o.moe.auto_policy);
        for(auto p:{wt.gate,wt.up})readonly.push_back(addr(p,nbytes({c.hidden,c.ffn,c.experts})));
        readonly.push_back(addr(wt.down,nbytes({c.ffn,c.hidden,c.experts})));
        readonly.push_back(addr(wt.o,nbytes({c.attention,c.hidden,1})));
        readonly.push_back(addr(wt.ffn_norm,size_t(c.hidden)*4));readonly.push_back(addr(wt.router,size_t(c.experts)*c.hidden*4));
    }
    if(has_qkv(c.entry)) {
        if(!wt.q||!wt.k||!wt.v||!wt.next_attn_norm)throw std::invalid_argument("missing NEXT-layer norm/QKV weights");
        check_kernel(o.qkv_kernel,c.hidden,w.plan.qkv_split);
        for(auto item:std::array<std::pair<WriteView,uint32_t>,3>{{{b.q,c.q_width},{b.k,c.kv_width},{b.v,c.kv_width}}}) {
            check_write(item.first,c.tokens,item.second);output_views.push_back(item);outputs.push_back(addr(item.first.data,checked_mul(required_elements(item.first,c.tokens,item.second),4)));
        }
        readonly.push_back(addr(wt.q,nbytes({c.hidden,c.q_width,1})));
        readonly.push_back(addr(wt.k,nbytes({c.hidden,c.kv_width,1})));readonly.push_back(addr(wt.v,nbytes({c.hidden,c.kv_width,1})));
        readonly.push_back(addr(wt.next_attn_norm,size_t(c.hidden)*4));
    }
    const auto arena=addr(w.base,w.plan.bytes);
    for(size_t i=0;i<outputs.size();++i){if(overlaps(outputs[i],arena))throw std::invalid_argument("output aliases Level4 scratch");
        for(size_t j=0;j<i;++j)if(overlaps(outputs[i],outputs[j])&&!disjoint_token_slices(output_views[i].first,output_views[i].second,output_views[j].first,output_views[j].second))throw std::invalid_argument("overlapping Level4 output bindings");}
    for(const auto&r:readonly){if(overlaps(r,arena))throw std::invalid_argument("input/weight aliases Level4 scratch");}
    // Input/output alias is only allowed for the residual view. Immutable weights
    // must never be overwritten by an output. Everything is packed before outputs.
    const size_t first_weight=has_post(c.entry)?2:1;
    for(size_t i=first_weight;i<readonly.size();++i)for(const auto&o_:outputs)
        if(overlaps(readonly[i],o_))throw std::invalid_argument("output overwrites immutable weight");
}
sycl::event enqueue_residual_norm(sycl::queue&q,const float*a,const float*b,const float*g,float*r,float*y,int32_t*bad,
 uint32_t tokens,uint32_t width,float eps,const std::vector<sycl::event>&deps) {
    if(!a||!bad||(!r&&!y)||!tokens||!width||!std::isfinite(eps)||eps<=0)throw std::invalid_argument("invalid residual/norm view");
    return q.submit([&](sycl::handler&h){h.depends_on(deps);h.parallel_for<ResidualNorm>(
        sycl::nd_range<1>{sycl::range<1>(size_t(tokens)*WG),sycl::range<1>(WG)},[=](sycl::nd_item<1>it){
        size_t t=it.get_group_linear_id();uint32_t lane=uint32_t(it.get_local_linear_id());float s=0;int invalid=0;
        for(uint32_t k=lane;k<width;k+=WG){float z=a[t*width+k]+(b?b[t*width+k]:0.f);s+=z*z;invalid|=!sycl::isfinite(z)||(g&&!sycl::isfinite(g[k]));}
        s=sycl::reduce_over_group(it.get_group(),s,sycl::plus<float>());invalid=sycl::reduce_over_group(it.get_group(),invalid,sycl::maximum<int>());
        invalid|=!sycl::isfinite(s);float inv=y?1.f/sycl::sqrt(s/float(width)+eps):1.f;
        if(lane==0)bad[t]=invalid;
        for(uint32_t k=lane;k<width;k+=WG){float z=a[t*width+k]+(b?b[t*width+k]:0.f);if(r)r[t*width+k]=z;if(y)y[t*width+k]=(z*inv)*(g?g[k]:1.f);}
    });});
}
sycl::event enqueue_router_logits(sycl::queue&q,const float*x,const float*w,float*y,uint32_t tokens,uint32_t width,uint32_t experts,const std::vector<sycl::event>&deps) {
    if(!x||!w||!y||!tokens||!width||!experts||experts>256)throw std::invalid_argument("invalid router logits view");
    return q.submit([&](sycl::handler&h){h.depends_on(deps);h.parallel_for<RouterLogits>(
        sycl::nd_range<1>{sycl::range<1>(size_t(tokens)*experts*WG),sycl::range<1>(WG)},[=](sycl::nd_item<1>it){
        size_t job=it.get_group_linear_id(),t=job/experts,e=job%experts;uint32_t lane=uint32_t(it.get_local_linear_id());float s=0;
        for(uint32_t k=lane;k<width;k+=WG)s+=x[t*width+k]*w[e*width+k];
        s=sycl::reduce_over_group(it.get_group(),s,sycl::plus<float>());if(lane==0)y[job]=s;
    });});
}
sycl::event enqueue_router_select(sycl::queue&q,const float*l,int32_t*ids,float*routes,int32_t*bad,uint32_t tokens,uint32_t e,uint32_t k,const std::vector<sycl::event>&deps) {
    if(!l||!ids||!routes||!bad||!tokens||!e||e>256||!k||k>8||k>e)throw std::invalid_argument("invalid router selection view");
    return q.submit([&](sycl::handler&h){h.depends_on(deps);h.parallel_for<RouterSelect>(sycl::range<1>(tokens),[=](sycl::id<1>ii){
        const size_t t=ii[0];const float neg=-std::numeric_limits<float>::infinity();float best[8];int32_t bi[8];
        for(uint32_t j=0;j<8;++j){best[j]=neg;bi[j]=-1;}float mx=neg;int invalid=0;
        for(uint32_t i=0;i<e;++i){float z=l[t*e+i];invalid|=!sycl::isfinite(z);mx=sycl::fmax(mx,z);
            // Strict > and ascending expert traversal preserve lower-ID ties.
            for(uint32_t j=0;j<k;++j)if(z>best[j]){for(uint32_t n=k-1;n>j;--n){best[n]=best[n-1];bi[n]=bi[n-1];}best[j]=z;bi[j]=int32_t(i);break;}}
        float den=0;for(uint32_t i=0;i<e;++i)den+=sycl::exp(l[t*e+i]-mx);
        float selected=0;float p[8];for(uint32_t j=0;j<k;++j){p[j]=sycl::exp(best[j]-mx)/den;selected+=p[j];}
        invalid|=!sycl::isfinite(den)||!sycl::isfinite(selected)||!(selected>0.f);bad[t]=invalid;
        for(uint32_t j=0;j<k;++j){ids[t*k+j]=invalid?-1:bi[j];routes[t*k+j]=invalid?0.f:p[j]/(selected+1e-20f);}
    });});
}
Run enqueue(sycl::queue&q,const Weights&wt,const Bindings&b,const Options&o,const Workspace&w,const std::vector<sycl::event>&deps) {
    validate(wt,b,o,w);validate_dg2_device(q.get_device());const auto c=w.plan.config;Run r;r.status=w.get<int32_t>("status");
    auto events=[&](uint64_t bits){std::vector<sycl::event>v;if(!bits)return deps;for(size_t i=0;i<r.count;++i)if((bits>>i)&1u)v.push_back(r.stages[i].event);return v;};
    auto add=[&](const char*n,sycl::event e,uint64_t p=0){auto bit=r.add(n,e,p);if(o.diagnostic_stage_waits)e.wait_and_throw();return bit;};
    const size_t jobs=size_t(c.tokens)*c.topk;
    auto zero=w.get<int32_t>("ids_zero"),stat=r.status,nb=w.get<int32_t>("norm_bad"),nx=w.get<int32_t>("next_bad"),rb=w.get<int32_t>("router_bad");
    auto so=w.get<int32_t>("dense_o_status"),sq=w.get<int32_t>("dense_q_status"),skv=w.get<int32_t>("dense_kv_status");
    auto sg=has_post(c.entry)?w.get<int32_t>("gate_status"):nullptr,sd=has_post(c.entry)?w.get<int32_t>("down_status"):nullptr;
    auto init=q.submit([&](sycl::handler&h){h.depends_on(deps);h.parallel_for<InitStatus>(sycl::range<1>(jobs),[=](sycl::id<1>ii){size_t i=ii[0];if(i<c.tokens){zero[i]=0;stat[i]=0;nb[i]=0;nx[i]=0;rb[i]=0;so[i]=0;sq[i]=0;skv[i]=0;}if(sg){sg[i]=0;sd[i]=0;}});});
    const uint64_t ib=add("init_status",init);
    auto pack=[&](const char*name,ReadView view,float*out,uint32_t width){auto ev=q.submit([&](sycl::handler&h){h.depends_on(deps);h.parallel_for<PackInput>(sycl::range<1>(size_t(c.tokens)*width),[=](sycl::id<1>ii){size_t i=ii[0];out[i]=view.data[offset(view,i/width,i%width)];});});return add(name,ev);};
    const uint64_t rp=pack("pack_residual",b.residual,w.get<float>("residual"),c.hidden);
    // Reuse the unmodified v0.4 dense-compatible topk=1 primitive. Q and K/V have
    // different output widths; pair K+V, never pretend Q/K shapes are equal.
    auto project=[&](const char*main_name,const char*reduce_name,const uint8_t*w0,const uint8_t*w1,const float*x,float*y0,float*y1,
                     uint32_t k,uint32_t m,int32_t*status,DensePrecision precision,maple_w2::Options ko,A8Workspace a,float*scratch,uint64_t parent){
        DeviceProblem p{{k,m,1},c.tokens,1,false,wt.dense_layout,w0,w1,x,zero,y0,y1,status};
        if(precision==DensePrecision::a8){auto z=enqueue_a8(q,p,{32,A8Mode::prequantized,ko},a,scratch,events(parent));auto bit=add(main_name,z.main,parent);return z.has_reduce?add(reduce_name,z.done,bit):bit;}
        auto z=maple_w2::enqueue(q,p,ko,scratch,events(parent));auto bit=add(main_name,z.main,parent);return z.has_reduce?add(reduce_name,z.done,bit):bit;
    };
    uint64_t post=rp|ib;MoeWorkspace mw{};
    if(has_post(c.entry)) {
        auto ap=pack("pack_fa_pre_o",b.attention,w.get<float>("attention"),c.attention);uint64_t oa=ap|ib;
        auto ow=aq(w,"o_a8");if(o.o_precision==DensePrecision::a8){auto e=enqueue_a8_quant(q,w.get<float>("attention"),ActivationType::f32,c.tokens,c.attention,32,ow,events(oa));oa=add("o_input_quant",e,oa);}
        auto op=project("o_projection","o_reduce",wt.o,nullptr,w.get<float>("attention"),w.get<float>("o"),nullptr,c.attention,c.hidden,so,o.o_precision,o.o_kernel,ow,w.get<float>("o_split"),oa);
        auto ff=enqueue_residual_norm(q,w.get<float>("residual"),w.get<float>("o"),wt.ffn_norm,w.get<float>("ff_residual"),w.get<float>("ff_norm"),nb,c.tokens,c.hidden,c.ffn_epsilon,events(op|rp));
        auto fb=add("residual_ffn_norm",ff,op|rp);
        auto logits=enqueue_router_logits(q,w.get<float>("ff_norm"),wt.router,w.get<float>("router_logits"),c.tokens,c.hidden,c.experts,events(fb));auto lb=add("router_logits_f32",logits,fb);
        auto sel=enqueue_router_select(q,w.get<float>("router_logits"),w.get<int32_t>("ids"),w.get<float>("routes"),rb,c.tokens,c.experts,c.topk,events(lb));auto sb=add("router_topk",sel,lb);
        mw=w.moe();MoeProblem mp{{c.hidden,c.ffn,c.experts},c.tokens,c.topk,wt.moe_layout,wt.gate,wt.up,wt.down,w.get<float>("ff_norm"),w.get<float>("routes"),w.get<int32_t>("ids"),w.get<float>("moe_out")};
        auto mr=enqueue_moe(q,mp,o.moe,mw,events(sb|ib));r.dispatch=mr.dispatch;const size_t start=r.count;
        for(size_t j=0;j<mr.count;++j){uint64_t parent=mr.stages[j].parents?(uint64_t(mr.stages[j].parents)<<start):(sb|ib);add(mr.stages[j].label,mr.stages[j].event,parent);}
        const uint64_t mb=uint64_t(1)<<(r.count-1);
        auto e=enqueue_residual_norm(q,w.get<float>("ff_residual"),w.get<float>("moe_out"),has_qkv(c.entry)?wt.next_attn_norm:nullptr,
            w.get<float>("hidden_out"),has_qkv(c.entry)?w.get<float>("next_norm"):nullptr,nx,c.tokens,c.hidden,c.next_epsilon,events(mb));
        post=add(has_qkv(c.entry)?"residual_next_attn_norm":"residual_terminal",e,mb);
    } else {
        auto e=enqueue_residual_norm(q,w.get<float>("residual"),nullptr,wt.next_attn_norm,w.get<float>("hidden_out"),w.get<float>("next_norm"),nx,c.tokens,c.hidden,c.next_epsilon,events(rp|ib));
        post=add("bootstrap_attn_norm",e,rp|ib);
    }
    uint64_t done=post;
    if(has_qkv(c.entry)) {
        auto a=aq(w,"qkv_a8");uint64_t p=post;
        if(o.qkv_precision==DensePrecision::a8){auto e=enqueue_a8_quant(q,w.get<float>("next_norm"),ActivationType::f32,c.tokens,c.hidden,32,a,events(p));p=add("qkv_input_quant_once",e,p);}
        auto qp=project("q_projection","q_reduce",wt.q,nullptr,w.get<float>("next_norm"),w.get<float>("q"),nullptr,c.hidden,c.q_width,sq,o.qkv_precision,o.qkv_kernel,a,w.get<float>("q_split"),p|ib);
        auto kvp=project("kv_pair_projection","kv_reduce",wt.k,wt.v,w.get<float>("next_norm"),w.get<float>("k"),w.get<float>("v"),c.hidden,c.kv_width,skv,o.qkv_precision,o.qkv_kernel,a,w.get<float>("kv_split"),p|ib);
        done=qp|kvp;
    }
    // Final per-token status merges quantizer and MoE device flags without host readback.
    const auto oi=has_post(c.entry)&&o.o_precision==DensePrecision::a8?aq(w,"o_a8").invalid:nullptr;
    const auto qi=has_qkv(c.entry)&&o.qkv_precision==DensePrecision::a8?aq(w,"qkv_a8").invalid:nullptr;
    const auto mi=has_post(c.entry)?mw.input_a8.invalid:nullptr,hi=has_post(c.entry)?mw.hidden_a8.invalid:nullptr;
    const auto hv=has_post(c.entry)?mw.hidden_invalid:nullptr,ov=has_post(c.entry)?mw.output_invalid:nullptr;
    const float* hx=w.get<float>("hidden_out"),*qv=has_qkv(c.entry)?w.get<float>("q"):nullptr,*kv=has_qkv(c.entry)?w.get<float>("k"):nullptr,*vv=has_qkv(c.entry)?w.get<float>("v"):nullptr;
    auto status_event=q.submit([&](sycl::handler&h){h.depends_on(events(done));h.parallel_for<FinalStatus>(sycl::range<1>(c.tokens),[=](sycl::id<1>ii){size_t t=ii[0];int v=nb[t]|nx[t]|rb[t]|so[t]|sq[t]|skv[t];
        if(oi)for(uint32_t j=0;j<c.attention/32;++j)v|=oi[t*(c.attention/32)+j];
        if(qi)for(uint32_t j=0;j<c.hidden/32;++j)v|=qi[t*(c.hidden/32)+j];
        if(mi){for(uint32_t j=0;j<c.hidden/32;++j)v|=mi[t*(c.hidden/32)+j];
            for(uint32_t a=0;a<c.topk;++a){size_t job=t*c.topk+a;v|=sg[job]|sd[job];for(uint32_t j=0;j<c.ffn/32;++j)v|=hi[job*(c.ffn/32)+j];for(uint32_t j=0;j<c.ffn;++j)v|=hv[job*c.ffn+j];}
            for(uint32_t j=0;j<c.hidden;++j)v|=ov[t*c.hidden+j];}
        for(uint32_t j=0;j<c.hidden;++j)v|=!sycl::isfinite(hx[t*c.hidden+j]);
        if(qv){for(uint32_t j=0;j<c.q_width;++j)v|=!sycl::isfinite(qv[t*c.q_width+j]);for(uint32_t j=0;j<c.kv_width;++j)v|=!sycl::isfinite(kv[t*c.kv_width+j])||!sycl::isfinite(vv[t*c.kv_width+j]);}
        stat[t]=v;
    });});auto st=add("island_status",status_event,done);
    const size_t row_width=size_t(c.hidden)+(has_qkv(c.entry)?size_t(c.q_width)+2ull*c.kv_width:0);
    auto output=q.submit([&](sycl::handler&h){h.depends_on(status_event);h.parallel_for<PackOutput>(sycl::range<1>(size_t(c.tokens)*row_width),[=](sycl::id<1>ii){size_t t=ii[0]/row_width,col=ii[0]%row_width;float value;WriteView out;size_t oc=col;
        if(col<c.hidden){value=hx[t*c.hidden+col];out=b.hidden;}
        else if((col-=c.hidden)<c.q_width){value=qv[t*c.q_width+col];out=b.q;oc=col;}
        else if((col-=c.q_width)<c.kv_width){value=kv[t*c.kv_width+col];out=b.k;oc=col;}
        else {col-=c.kv_width;value=vv[t*c.kv_width+col];out=b.v;oc=col;}
        if(!sycl::isfinite(value)||stat[t])value=std::numeric_limits<float>::quiet_NaN();
        out.data[offset(out,t,oc)]=value;
    });});add("export_hidden_raw_qkv",output,st);return r;
}
} // namespace
