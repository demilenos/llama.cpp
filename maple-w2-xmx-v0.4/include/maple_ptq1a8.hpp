// SPDX-License-Identifier: MIT
#pragma once

#include "w2a8_reference.hpp"

#include <array>

namespace maple_w2 {

// Prism PTQ1_0: 128 ternary weights per block.
// Native block = 24 base-3 bytes + 2 high bytes + one fp16 scale.
constexpr uint32_t ptq1_block_k = 128;
constexpr uint32_t ptq1_block_bytes = 28;

// XMX-facing PTQ1 tile:
//   512 B signed-INT2 VNNI codes for K256 x N8
//    32 B scales: [row0 lo128, row0 hi128, ... row7 lo128, row7 hi128]
constexpr uint32_t ptq1_tile_bytes = 544;

inline size_t ptq1_nbytes(Shape s) {
    if (!s.k || s.k % 256 || !s.m || s.m % 8 || !s.experts) {
        throw std::invalid_argument("PTQ1 XMX requires K % 256 == 0, M % 8 == 0 and positive dimensions");
    }
    return checked_mul(checked_mul(checked_mul(s.experts, s.m), s.k / ptq1_block_k), ptq1_block_bytes);
}

inline size_t ptq1_native_offset(Shape s, uint32_t e, uint32_t r, uint32_t b128) {
    return ((size_t(e) * s.m + r) * (s.k / ptq1_block_k) + b128) * ptq1_block_bytes;
}

inline size_t ptq1_tile_offset(Shape s, uint32_t e, uint32_t rt, uint32_t kb256) {
    return ((size_t(e) * (s.m / 8) + rt) * (s.k / 256) + kb256) * ptq1_tile_bytes;
}

// Matches Prism's dequantize_row_ptq1_0 exactly: uint8 overflow is intentional.
inline int ptq1_trit(uint8_t packed, unsigned n) {
    static constexpr uint8_t pow3[5] = {1, 3, 9, 27, 81};
    if (n >= 5) {
        throw std::invalid_argument("PTQ1 trit index out of range");
    }
    const uint8_t q = uint8_t(uint16_t(packed) * pow3[n]);
    const int xi = int((uint16_t(q) * 3u) >> 8);
    return xi - 1;
}

inline void decode_ptq1_block(const uint8_t * block, int8_t * dst128) {
    if (!block || !dst128) {
        throw std::invalid_argument("null PTQ1 block");
    }

    size_t j = 0;
    size_t out = 0;
    constexpr std::array<size_t, 3> stages = {32, 16, 8};

    for (size_t c : stages) {
        for (; j + c <= 24; j += c) {
            for (unsigned n = 0; n < 5; ++n) {
                for (size_t m = 0; m < c; ++m) {
                    const int v = ptq1_trit(block[j + m], n);
                    if (v < -1 || v > 1) {
                        throw std::invalid_argument("invalid PTQ1 ternary code");
                    }
                    dst128[out++] = int8_t(v);
                }
            }
        }
    }

    for (unsigned n = 0; n < 4; ++n) {
        for (size_t h = 0; h < 2; ++h) {
            const int v = ptq1_trit(block[24 + h], n);
            if (v < -1 || v > 1) {
                throw std::invalid_argument("invalid PTQ1 high ternary code");
            }
            dst128[out++] = int8_t(v);
        }
    }

    if (out != ptq1_block_k) {
        throw std::logic_error("PTQ1 decoder did not produce 128 values");
    }
}

inline uint16_t ptq1_scale_bits(const uint8_t * block) {
    return u16(block + 26);
}

inline void validate_ptq1(const std::vector<uint8_t> & w, Shape s) {
    if (w.size() != ptq1_nbytes(s)) {
        throw std::invalid_argument("PTQ1 payload length mismatch");
    }
    for (size_t off = 0; off < w.size(); off += ptq1_block_bytes) {
        const uint16_t d = ptq1_scale_bits(w.data() + off);
        if ((d & 0x7c00u) == 0x7c00u) {
            throw std::invalid_argument("non-finite PTQ1 weight scale");
        }
    }
}

// One-time host repack for bring-up. No dequantized weight tensor is materialized:
// PTQ1 ternary codes become the exact signed-INT2 VNNI bytes consumed by the
// existing W2A8 DPAS kernel. The only layout extension is retaining both K128 scales.
inline std::vector<uint8_t> repack_ptq1_s2tile8(const std::vector<uint8_t> & w, Shape s) {
    validate_ptq1(w, s);
    const size_t tiles = checked_mul(checked_mul(s.experts, s.m / 8), s.k / 256);
    std::vector<uint8_t> out(checked_mul(tiles, size_t(ptq1_tile_bytes)), 0);

    for (uint32_t e = 0; e < s.experts; ++e) {
        for (uint32_t rt = 0; rt < s.m / 8; ++rt) {
            for (uint32_t kb = 0; kb < s.k / 256; ++kb) {
                uint8_t * dst = out.data() + ptq1_tile_offset(s, e, rt, kb);

                for (unsigned r = 0; r < 8; ++r) {
                    std::array<int8_t, ptq1_block_k> lo{};
                    std::array<int8_t, ptq1_block_k> hi{};
                    const uint8_t * b0 = w.data() + ptq1_native_offset(s, e, rt * 8 + r, kb * 2);
                    const uint8_t * b1 = w.data() + ptq1_native_offset(s, e, rt * 8 + r, kb * 2 + 1);
                    decode_ptq1_block(b0, lo.data());
                    decode_ptq1_block(b1, hi.data());

                    put16(dst + 512 + r * 4 + 0, ptq1_scale_bits(b0));
                    put16(dst + 512 + r * 4 + 2, ptq1_scale_bits(b1));

                    for (unsigned k = 0; k < 256; ++k) {
                        const int v = k < 128 ? int(lo[k]) : int(hi[k - 128]);
                        const uint32_t signed_bits = uint32_t(v) & 3u; // -1,0,+1 -> 3,0,1
                        const unsigned word = (k / 16) * 8 + r;
                        const unsigned shift = 2 * (k % 16);
                        uint32_t bits = u32(dst + word * 4);
                        bits |= signed_bits << shift;
                        put32(dst + word * 4, bits);
                    }
                }
            }
        }
    }
    return out;
}

} // namespace maple_w2
