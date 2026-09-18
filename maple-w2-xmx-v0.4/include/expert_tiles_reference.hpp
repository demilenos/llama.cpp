// SPDX-License-Identifier: MIT
#pragma once
#include "expert_grouping_reference.hpp"
namespace maple_w2 {
inline void validate_token_tile(uint32_t r) {
    if(r!=4 && r!=8) throw std::invalid_argument("multi-token DPAS requires repeat 4 or 8");
}
// sum_e ceil(count[e]/R) <= min(J, ceil(J/R) + B - 1), B=E+1 incl. invalid bucket.
// A host-known bound avoids reading a GPU count back before every launch.
inline size_t expert_tile_capacity(size_t jobs,uint32_t experts,uint32_t r) {
    validate_grouping_size(jobs,experts);validate_token_tile(r);
    return std::min(jobs,(jobs+r-1)/r+experts);
}
struct ExpertTilesHost {
    std::vector<int32_t> expert,begin,rows;
};
inline ExpertTilesHost make_expert_tiles_reference(const ExpertGroupingHost& g,uint32_t r) {
    validate_token_tile(r);
    if(g.counts.size()<2 || g.counts.size()>257 || g.offsets.size()!=g.counts.size()+1 || g.offsets.front()!=0)
        throw std::invalid_argument("bad grouped plan shape");
    const size_t jobs=g.sorted_to_job.size();
    validate_grouping_size(jobs,uint32_t(g.counts.size()-1));
    ExpertTilesHost t;
    for(size_t e=0;e<g.counts.size();++e) {
        if(g.counts[e]<0 || g.offsets[e]<0 || int64_t(g.offsets[e])+g.counts[e]!=g.offsets[e+1] || size_t(g.offsets[e+1])>jobs)
            throw std::invalid_argument("bad grouped plan counts/offsets");
        for(int32_t j=0;j<g.counts[e];j+=int32_t(r)) {
            t.expert.push_back(int32_t(e));t.begin.push_back(g.offsets[e]+j);
            t.rows.push_back(std::min(int32_t(r),g.counts[e]-j));
        }
    }
    if(size_t(g.offsets.back())!=jobs)throw std::invalid_argument("group plan does not cover jobs");
    if(t.expert.size()>expert_tile_capacity(jobs,uint32_t(g.counts.size()-1),r))
        throw std::logic_error("tile bound violated");
    return t;
}
} // namespace maple_w2
