// SPDX-License-Identifier: MIT
#pragma once
#include "level4_zerocopy_contract.hpp"
#if defined(_WIN32) && !defined(MAPLE_ZC_NATIVE_MOCK)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#include <d3d12.h>
#include <wrl/client.h>
#include <cstdio>
namespace maple_w2::level4::zc {
inline void require_same_windows_device(DeviceIdentity a, DeviceIdentity b,
        const uint8_t * vk_uuid, const uint8_t * ze_uuid, bool ze_subdevice) {
    if (a.node_mask == b.node_mask) { require_same_device(a, b); return; }
    bool nonzero = false; for (size_t i = 0; i < 16; ++i) nonzero |= vk_uuid[i] != 0;
    if (ze_subdevice || a.node_mask != 1 || !b.node_mask || (b.node_mask & (b.node_mask - 1)) ||
        !nonzero || std::memcmp(vk_uuid, ze_uuid, 16)) { require_same_device(a, b); return; }
    const auto original_mask = b.node_mask;
    b.node_mask = 1;
    require_same_device(a, b);
    LUID luid{}; std::memcpy(&luid, a.luid.data(), 8);
    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    Microsoft::WRL::ComPtr<ID3D12Device> device;
    DXGI_ADAPTER_DESC1 desc{};
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter))) ||
        FAILED(adapter->GetDesc1(&desc)) || desc.VendorId != a.vendor || desc.DeviceId != a.device ||
        FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))) || device->GetNodeCount() != 1)
        throw std::invalid_argument("node mask mismatch without single-node DXGI identity proof");
    std::fprintf(stderr, "LEVEL4_ZC_IDENTITY single_node_dxgi_verified=1 uuid_match=1 ze_reported_mask=%u effective_mask=1\n", original_mask);
}
}
#endif
