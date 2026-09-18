// SPDX-License-Identifier: MIT
#include "level4_zerocopy_contract.hpp"
#include "level4_contract.hpp"
#include <iostream>
#include <vector>
namespace z=maple_w2::level4::zc;namespace l=maple_w2::level4;
int main()try{size_t checks=0;auto need=[&](bool b){++checks;if(!b)throw std::runtime_error("contract check failed");};auto reject=[&](auto fn){bool b=false;try{fn();}catch(const std::exception&){b=true;}need(b);};
 z::BufferExtent e{8192,1024,4096,8};need(e.resolve(64,128,64,8)==1088);reject([&]{e.resolve(0,4,4,7);});reject([&]{e.resolve(4092,8,4,8);});reject([&]{e.resolve(SIZE_MAX,8,4,8);});reject([&]{e.resolve(1,8,4,8);});reject([&]{e.resolve(0,0,4,8);});
 z::DeviceIdentity a{0x8086,0x56a1,1,{1,2,3,4,5,6,7,8},true},b=a;z::require_same_device(a,b);need(true);b.luid[7]++;reject([&]{z::require_same_device(a,b);});b=a;b.node_mask=2;reject([&]{z::require_same_device(a,b);});b=a;b.luid_valid=false;reject([&]{z::require_same_device(a,b);});
 z::OwnershipState s;reject([&]{s.acquire();});s.release();reject([&]{s.release();});s.released();s.acquire();s.release();s.released();s.acquire();need(s.value()==z::Ownership::acquire_submitted);s.poison();reject([&]{s.check();});reject([&]{s.release();});
 for(auto entry:{l::Entry::bootstrap,l::Entry::advance,l::Entry::terminal})for(uint32_t q:{1u,13u,184u,2048u}){l::Config c;c.tokens=q;c.entry=entry;auto p=l::make_plan(c),d=l::make_plan(c,1,1,1,1,1,1,true);need(d.bytes<p.bytes);size_t saved=size_t(q)*c.hidden*4*2;if(l::has_post(entry))saved+=size_t(q)*c.attention*4;if(l::has_qkv(entry))saved+=size_t(q)*(c.q_width+2*c.kv_width)*4;need(p.bytes-d.bytes==saved);}
 std::cout<<"zero-copy portable contract PASS checks="<<checks<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
