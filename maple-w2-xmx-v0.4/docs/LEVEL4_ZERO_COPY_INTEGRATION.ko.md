# Level4 zero-copy 연결 계약

## 1. 기존 queue/context를 재사용

`integration/level4_zerocopy_win32.hpp`의 실제 구현을 사용합니다.
`integration/level4_win32_memory.hpp`의 이전 최소 helper와 중복 import하지 마십시오.

```cpp
namespace l4 = maple_w2::level4;
namespace zc = l4::zc;

// 아래 Vk/SYCL handles와 owner는 기존 llama.cpp backend가 소유한 실제 값.
// queue_guard는 이 queue를 사용하는 기존 모든 vkQueueSubmit/WaitIdle과 공유.
auto interop = zc::Context::create(
    existing_sycl_queue, vk_physical_device, vk_device, vk_compute_queue,
    vk_compute_queue_family, queue_guard, backend_device_owner);
```

새 SYCL context를 몰래 만들지 않습니다. 호출 queue가 명시적인 single-device Level Zero context여야 합니다.
Vulkan과 ZE의 vendor/device, LUID, nodeMask를 대조합니다. 이름/ordinal은 동등성 증거로 사용하지 않습니다.
Vulkan 1.1+, `VK_KHR_external_memory_win32`, ZE OPAQUE_WIN32 import capability가 필요합니다.
타일/linked-adapter 불일치는 fail-closed입니다.

## 2. Shared arena를 한 번 export/import

```cpp
auto shared_io = interop->create_buffer(io_bytes); // device-local, exclusive, dedicated VkBuffer arena
VkBuffer vk_io = shared_io->vk_buffer();
float *x = shared_io->at<float>(x_offset_bytes, size_t(tokens)*hidden);
float *fa = shared_io->at<float>(fa_offset_bytes, size_t(tokens)*attention_width);
float *h = shared_io->at<float>(h_offset_bytes, size_t(tokens)*hidden);
float *q = shared_io->at<float>(q_offset_bytes, size_t(tokens)*q_width);
float *k = shared_io->at<float>(k_offset_bytes, size_t(tokens)*kv_width);
float *v = shared_io->at<float>(v_offset_bytes, size_t(tokens)*kv_width);
```

메타데이터를 연결할 때 실제 Vulkan tensor가 **이 `vk_io`와 해당 offset을 사용하도록** allocator에 등록해야 합니다.
기존 tensor가 다른 non-exportable allocation에 남아 있으면 shared arena를 하나 더 만든 것만으로 zero-copy가 되지 않습니다.

기존 allocator가 exportable allocation을 이미 제공하면 재할당하지 않고:

```cpp
auto shared_io = interop->adopt(zc::ExportedBuffer{
    actual_vk_buffer, actual_vk_device_memory,
    {allocation_size_bytes, vk_bind_offset_bytes, vk_buffer_size_bytes, allocation_generation},
    allocation_and_buffer_lifetime_owner
});
```

`source_owner`는 실제 VkBuffer와 VkDeviceMemory를 모두 살려두는 `shared_ptr` lease여야 합니다.
메모리가 재생성되면 generation을 변경하고 old import를 in-flight 중에 교체하지 않습니다.
동일 `(VkDeviceMemory,generation)`은 캐시하므로 반복 `adopt`도 재-import하지 않습니다.

주소는 다음 식 하나로 계산합니다.

```
ZE imported allocation base + VkBindBufferMemory offset + tensor-relative VkBuffer offset
```

**`VkDeviceAddress`를 `float*`로 캐스팅하지 않습니다.** 실제 external import에서 얻은 pointer입니다.
Win32 NT HANDLE은 import 직후 한 번 `CloseHandle`, import mapping은 fence/drain 후 `zeMemFree`,
그 다음 Vulkan 원본 memory를 해제합니다. import와 queue/device의 실제 수명이 함께 유지됩니다.

## 3. ggml view adapter

`integration/ggml_level4_zerocopy.hpp`:

```cpp
zc::TensorBinding binding { ggml_boundary_tensor, shared_io, resolved_offset_inside_vk_buffer };
auto view = binding.read(tokens, logical_width); // 또는 .write(...)
```

resolved_offset에는 **이미 ggml view offset이 반영**되어 있어야 합니다. 이 helper는 `view_offs`를 다시 더하지 않습니다.
지원되는 logical shape는 기존 Level4 adapter와 같은 F32 `[D,Q]` / `[Dh,H,Q]`이고,
직접 실행에서는 전체 token row가 연속이어야 합니다. head-major `[Dh,Q,H]`를 이름만 보고 재해석하지 않습니다.
Q1은 token stride가 의미 없지만 head/lane stride는 그대로 검증합니다.

입력 residual/FA와 출력 hidden/Q/K/V는 disjoint하게 두십시오. 특히 hidden residual은 **ping-pong buffer**입니다.
기존 packed 실행에서 허용되던 alias는 direct 모드에서 조용히 허용하지 않습니다.
출력을 이미 caller가 사용하는 fused-QKV parent buffer의 비연속 view에 쓰려면 직접지원이 아니라 layout 변경부터 필요합니다.

## 4. Workspace / weights 준비

```cpp
auto plan = l4::make_plan(config, gate_split, down_split, o_split, qkv_split,
                          gate_token_tile, down_token_tile, true /* direct boundary */);
auto ws = l4::bind_workspace(device_scratch, scratch_capacity, plan);

l4::Bindings b{
    l4::contiguous_read(x,tokens,hidden),
    l4::contiguous_read(fa,tokens,attention_width),
    l4::contiguous_write(h,tokens,hidden),
    l4::contiguous_write(q,tokens,q_width),
    l4::contiguous_write(k,tokens,kv_width),
    l4::contiguous_write(v,tokens,kv_width)
};
std::vector<zc::PrivateRegion> private_regions = {
    {device_scratch, scratch_capacity, scratch_owner},
    {w2_weight_arena, w2_weight_bytes, w2_weight_owner},
    {norm_router_arena, norm_router_bytes, norm_router_owner}
};
auto prepared = interop->prepare(weights, b, options, ws, graph_semantics,
                                 {shared_io}, private_regions);
```

private region도 동일 ZE context/device allocation인지 확인합니다. imported 메모리를 private로 위장하면 거부합니다.
Vulkan과 공유하는 weights도 있다면 `shared` 목록에 포함하고 private로 넣지 않습니다.
같은 VkBuffer는 목록에 한 번만 넣으며 서로 겹치는 별도 VkBuffer alias는 이 첫 transport에서 지원하지 않습니다.

`Weights` layout/native TQ2/s2tile8 의미는 기존 Level4와 동일합니다.
**GGUF native TQ2를 s2tile8이라고 표시해 raw pointer만 연결하면 안 됩니다.**
필요한 load-time repack/upload는 별도의 cold setup이며 이번 zero-copy의 boundary 비용과 구분합니다.
O/MoE는 현재 layer, norm/QKV는 다음 layer의 weights를 연결합니다.

## 5. 런타임: submit → execute → consume

```cpp
// (A) 외부 graph consumer/LoRA/bias/row-selection을 검증한 기존 Level4 matcher 필요.
// (B) 현재 Vulkan command buffer에 기록만 한 producer를 반드시 실제 queue에 submit.
submit_existing_vulkan_producers();

// (C) 내부 구현이 release barrier+fence, 실제 Level4, acquire submit까지 수행.
auto result = interop->execute(prepared);

// (D) execute가 제출한 acquire 뒤에, 같은 queue로 head norm/RoPE/KV/FA를 제출.
submit_existing_vulkan_consumers();
```

여기의 submit 함수 이름은 **호출부가 연결할 지점**이며 llama.cpp에 존재한다고 가정한 함수가 아닙니다.
`Context::execute()`는 실제 구현입니다. 별도 ownership callback을 중복 호출하지 마십시오.

동기화 순서:
1. 공유 recursive mutex 획득. 해당 queue의 다른 host submit과 충돌하지 않게 함.
2. 이전 acquire command buffer retirement 확인.
3. 모든 공유 buffer에 local→EXTERNAL release barrier를 기록/submit.
4. 그 release fence 완료를 기다림: 이전 producer가 제출되어 있으면 이 fence가 함께 포괄.
5. 실제 Level4 SYCL chain 실행 및 마지막 event 대기. 중간 mandatory host wait 없음.
6. EXTERNAL→local acquire barrier submit. 여기서 바로 acquire fence를 host wait하지 않음.
7. 반환 후 같은 queue의 consumer가 acquire 뒤에 실행.

**Zero-copy지만 host-fenced입니다.** external GPU semaphore/timeline 기능은 이번에 구현하지 않았습니다.
`result.run.done`은 SYCL 완료이지 이후 Vulkan consumer 완료가 아닙니다.
동일 request의 scratch 재사용은 다음 `execute`의 순서가 보장하며,
서로 다른 in-flight request가 arena를 공유하려면 별도의 명시적인 직렬화가 필요합니다.
다중 VkQueue나 서로 다른 context에서 동시 사용하는 자원은 별도 설계 없이는 지원하지 않습니다.

## 6. 실패 / lifetime

`prepare()`의 shape/export/lease 문제는 제출 전에 실패하므로 원래 graph 경로로 선택을 되돌릴 수 있습니다.
`execute()`가 ownership 전환 후 실패하면 context를 poisoned로 표시합니다.
**같은 output을 소비하거나 거기에 fallback을 덧실행하지 마십시오.** 호출 요청을 실패 처리하고 drain/재생성합니다.

캐시는 mapping 하나를 영구 무한 보존하는 구조가 아니라 weak registry + 실제 buffer/prepared/in-flight strong leases입니다.
이전 후속 Vulkan 소비자는 다음 release fence 또는 explicit `drain()`이 끝날 때까지 lease로 보호합니다.
`drain()`은 queue를 완료하고 transport의 이전 in-flight lease를 푸는 cold lifecycle 연산입니다.
GPU 완료를 증명할 수 없는 teardown 오류는 owner를 격리하여 use-after-free를 피하고 로그를 남깁니다.
모든 외부 submit이 동일 queue/mutex/lifetime 규약을 지켜야 하는 책임은 호출부에도 있습니다.

## 7. 범위 제한과 관측점

기존 Level4 stage pack/export가 제거된 것은 `Run.boundary_traffic.bytes()==0`와 실제 stage label로 검사합니다.
이 값은 코드가 제출한 copy byte count이지 WDDM/PCIe hardware telemetry가 아닙니다.
probe는 내부 `total_host_us`와 외부 `execute_wall_us`를 따로 기록하여 validation/lock/retire 비용을 누락하지 않습니다.
`inbound_host_us`에는 아직 끝나지 않은 Vulkan producer 대기도 포함되므로 순수 API 전환 비용과 동일하지 않습니다.
모든 전역/중간 메모리 traffic이 사라지는 것도 아닙니다. RMSNorm/MoE/quantization은 원래 계산대로 수행합니다.

동일 config에서 direct plan은 boundary staging span만 제거합니다.
Maple advance Q2048는 72 MiB를 회수하고, workspace의 다른 항목은 유지합니다.
실제 모델 weights·KV·export arena·allocator rounding은 이 workspace 수치와 별도입니다.

## 공식 API 근거

- Level Zero Win32 external import descriptor와 `zeMemAllocDevice` pNext:
  https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/mem.html
- Vulkan Win32 HANDLE lifetime / `vkGetMemoryWin32HandleKHR`:
  https://docs.vulkan.org/refpages/latest/refpages/source/vkGetMemoryWin32HandleKHR.html
- Vulkan external queue family ownership transfer:
  https://docs.vulkan.org/spec/latest/chapters/synchronization.html
- Level Zero device LUID/node-mask properties:
  https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/device.html
- SYCL USM와 context:
  https://github.khronos.org/SYCL_Reference/iface/usm_basic_concept.html

위 명세 확인은 실기 지원의 증거가 아닙니다. 현재 드라이버에 대한 실제 roundtrip은 제공한 Windows probe로 확인합니다.
