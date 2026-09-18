// Verifies the CUDA device accessor's element mapping matches the CPU codec exactly.
// The CUDA kernels cannot be compiled here (no nvcc), but this is where a mapping bug
// would hide, and it is pure integer logic, so it is checkable on the host.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

#define QK_PTQ1_0 128
struct block_ptq1_0 { uint8_t qs[24]; uint8_t qh[2]; uint16_t d; };

// --- transcribed verbatim from ggml-cuda/dequantize.cuh ---
static int ptq1_0_trit(const block_ptq1_0 * x, const int e) {
    uint8_t b; int n;
    if (e < 80)       { b = x->qs[e & 15];              n = e >> 4; }
    else if (e < 120) { const int t = e - 80; b = x->qs[16 + (t & 7)]; n = t >> 3; }
    else              { const int t = e - 120; b = x->qh[t & 1];       n = t >> 1; }
    uint32_t v = b;
    for (int i = 0; i < 4; ++i) if (i < n) v = (v * 3) & 0xFF;
    return (int)((v * 3) >> 8) - 1;
}

// Vulkan Q8-DMMV closed-form digit decoder.  This is intentionally independent
// of the iterative CUDA accessor so the integer-dot shader's math is checked here.
static int ptq1_0_trit_vulkan_q8(const block_ptq1_0 * x, const int e) {
    uint8_t b; int n;
    if (e < 80)       { b = x->qs[e & 15];                  n = e >> 4; }
    else if (e < 120) { const int t = e - 80; b = x->qs[16 + (t & 7)]; n = t >> 3; }
    else              { const int t = e - 120; b = x->qh[t & 1];       n = t >> 1; }

    static const uint32_t pow3[5] = {3, 9, 27, 81, 243};
    const uint32_t v = (uint32_t(b) * pow3[n]) >> 8;
    const uint32_t mod3 = v - 3u * ((v * 171u) >> 9);
    return int(mod3) - 1;
}

// --- the CPU reference traversal from ggml-quants.c dequantize_row_ptq1_0 ---
static void cpu_ref(const block_ptq1_0 * x, int * out) {
    const uint8_t pow3[6] = {1,3,9,27,81,243};
    const size_t stages[3] = {32,16,8};
    int o = 0; size_t j = 0;
    for (size_t s = 0; s < 3; ++s) {
        const size_t c = stages[s];
        for (; j + c <= sizeof(x->qs); j += c)
            for (size_t n = 0; n < 5; ++n)
                for (size_t m = 0; m < c; ++m) {
                    uint8_t q = x->qs[j+m] * pow3[n];
                    out[o++] = (int)(((uint16_t)q * 3) >> 8) - 1;
                }
    }
    for (size_t n = 0; n < 4; ++n)
        for (size_t h = 0; h < sizeof(x->qh); ++h) {
            uint8_t q = x->qh[h] * pow3[n];
            out[o++] = (int)(((uint16_t)q * 3) >> 8) - 1;
        }
}

int main(void) {
    unsigned seed = 12345;
    long bad = 0, total = 0;
    for (int trial = 0; trial < 20000; ++trial) {
        block_ptq1_0 blk;
        for (int i = 0; i < 24; ++i) { seed = seed*1103515245u+12345u; blk.qs[i] = (seed>>16)&0xFF; }
        for (int i = 0; i <  2; ++i) { seed = seed*1103515245u+12345u; blk.qh[i] = (seed>>16)&0xFF; }
        int ref[QK_PTQ1_0]; cpu_ref(&blk, ref);
        for (int e = 0; e < QK_PTQ1_0; ++e) {
            ++total;
            const int cuda = ptq1_0_trit(&blk, e);
            const int vkq8 = ptq1_0_trit_vulkan_q8(&blk, e);
            if (cuda != ref[e] || vkq8 != ref[e]) {
                if (bad < 5) printf("  MISMATCH blk%d e=%d cuda=%d vkq8=%d cpu=%d\n", trial, e, cuda, vkq8, ref[e]);
                ++bad;
            }
        }
    }
    printf("  checked %ld element positions across 20000 random blocks\n", total);
    printf("  mismatches: %ld\n", bad);
    printf("  %s\n", bad==0 ? "CUDA and Vulkan-Q8 element mappings MATCH the CPU codec exactly" : "MAPPING BUG");
    return bad != 0;
}
