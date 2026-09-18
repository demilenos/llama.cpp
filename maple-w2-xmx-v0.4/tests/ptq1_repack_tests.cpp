// SPDX-License-Identifier: MIT
#include "maple_ptq1a8.hpp"

#include <array>
#include <cassert>
#include <cstring>
#include <iostream>

namespace {

using maple_w2::Shape;

void encode_ptq1_block(const std::array<int8_t, 128> & values, uint16_t scale, uint8_t * block) {
    std::memset(block, 0, maple_w2::ptq1_block_bytes);

    size_t j = 0;
    size_t xoff = 0;
    constexpr std::array<size_t, 3> stages = {32, 16, 8};

    for (size_t c : stages) {
        for (; j + c <= 24; j += c) {
            for (size_t m = 0; m < c; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    const int xi = int(values[xoff + m + n * c]) + 1;
                    assert(xi >= 0 && xi <= 2);
                    q = uint8_t(q * 3 + xi);
                }
                block[j + m] = uint8_t((uint16_t(q) * 256u + 242u) / 243u);
            }
            xoff += 5 * c;
        }
    }

    for (size_t h = 0; h < 2; ++h) {
        uint8_t q = 0;
        for (size_t m = 0; m < 4; ++m) {
            const int xi = int(values[xoff + h + m * 2]) + 1;
            assert(xi >= 0 && xi <= 2);
            q = uint8_t(q * 3 + xi);
        }
        q = uint8_t(q * 3);
        block[24 + h] = uint8_t((uint16_t(q) * 256u + 242u) / 243u);
    }
    xoff += 8;
    assert(xoff == 128);

    maple_w2::put16(block + 26, scale);
}

std::array<int8_t, 128> make_values(unsigned row, unsigned half) {
    std::array<int8_t, 128> values{};
    for (unsigned i = 0; i < values.size(); ++i) {
        values[i] = int8_t(int((i * 7 + row * 5 + half * 11) % 3) - 1);
    }
    return values;
}

} // namespace

int main() {
    {
        const auto input = make_values(3, 1);
        std::array<uint8_t, maple_w2::ptq1_block_bytes> block{};
        encode_ptq1_block(input, maple_w2::float_to_half(0.75f), block.data());

        std::array<int8_t, 128> decoded{};
        maple_w2::decode_ptq1_block(block.data(), decoded.data());
        assert(decoded == input);

        for (uint32_t i = 0; i < maple_w2::ptq1_block_k; ++i) {
            const auto coord = maple_w2::ptq1_trit_coord(i);
            assert(coord.byte < 26);
            assert(coord.trit < 5);
            assert(maple_w2::ptq1_trit(block[coord.byte], coord.trit) == input[i]);
        }
    }

    const Shape shape{256, 8, 1};
    std::vector<uint8_t> native(maple_w2::ptq1_nbytes(shape), 0);
    std::array<std::array<std::array<int8_t, 128>, 2>, 8> expected{};
    std::array<std::array<uint16_t, 2>, 8> scales{};

    for (unsigned row = 0; row < 8; ++row) {
        for (unsigned half = 0; half < 2; ++half) {
            expected[row][half] = make_values(row, half);
            scales[row][half] = maple_w2::float_to_half(0.125f * float(1 + row * 2 + half));
            uint8_t * block = native.data() + maple_w2::ptq1_native_offset(shape, 0, row, half);
            encode_ptq1_block(expected[row][half], scales[row][half], block);
        }
    }

    const auto tile = maple_w2::repack_ptq1_s2tile8(native, shape);
    assert(tile.size() == maple_w2::ptq1_tile_bytes);

    for (unsigned row = 0; row < 8; ++row) {
        assert(maple_w2::u16(tile.data() + 512 + row * 4 + 0) == scales[row][0]);
        assert(maple_w2::u16(tile.data() + 512 + row * 4 + 2) == scales[row][1]);

        for (unsigned k = 0; k < 256; ++k) {
            const int expected_value = k < 128 ? expected[row][0][k] : expected[row][1][k - 128];
            assert(maple_w2::s2tile8_at(tile.data(), k, row) == expected_value);
        }
    }

    {
        auto bad = native;
        maple_w2::put16(bad.data() + 26, 0x7c00u);
        bool rejected = false;
        try {
            (void) maple_w2::repack_ptq1_s2tile8(bad, shape);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        assert(rejected);
    }

    std::cout << "PTQ1_REPACK PASS bytes=" << tile.size() << "\n";
    return 0;
}
