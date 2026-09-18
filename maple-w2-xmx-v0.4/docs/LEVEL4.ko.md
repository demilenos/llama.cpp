# Maple W2 XMX v0.4 — Level 4 island upgrade v1

Interop boundary를 넓혀 비용을 줄이자는 설계와 Level 4 선택은 사용자의 제안이다. 이 변경은 해당 경계를 구현한다. Level 5/6(RoPE·KV 이동)은 구현하지 않는다.

## 정확한 구현 범위

`src/maple_level4.cpp`는 실제 SYCL kernel과 기존 XMX v0.4 호출을 이어 붙인 실행 모듈이다. 설계만 적은 스켈레톤이 아니다. 다만 사용자 **현재 merged llama.cpp의 graph optimizer에 자동으로 등록된 구현은 아니다.** 첨부된 최신 full ZIP은 결과·소스 hash bundle이며 현재 `ggml-vulkan.cpp`와 graph fusion hook의 원본은 없다. 공개 이전 fork를 현 merged tree로 간주해 덮어쓰지 않았다.

이 패키지에서 추가한 것은 실행 모듈, scratch planner, shared-memory/handoff 연결 API, CPU 및 GPU probe다. **실제 ggml graph span recognizer와 현재 allocator callbacks 연결은 별도 호출부 작업이다.** Level 4 모듈 파일이 컴파일된 것과 서버가 Level 4를 사용한다는 것은 다르다.

## 세 진입점

- `bootstrap`: 입력 hidden → 해당 layer attention RMSNorm → raw Q/K/V. 보통 layer 0.
- `advance`: layer L의 pre-O FA → O_L → residual_L 합산 → FFN RMSNorm_L → FP32 router_L → W2A8 MoE_L → residual 합산 → attention RMSNorm_(L+1) → raw Q/K/V_(L+1).
- `terminal`: 마지막 layer pre-O FA → O → residual/norm/router/MoE/residual → hidden. 다음 QKV는 호출하지 않는다. 최종 norm/head/sampler는 기존 실행기로 남긴다.

`Weights.current_layer/next_layer`가 위 조합을 만족하지 않으면 **첫 GPU submit 전에 거부**한다. Bootstrap은 current=-1, next=대상 layer; advance는 next=current+1; terminal은 next=-1.

출력 hidden은 다음 attention의 residual용이다. 반드시 함께 보존한다. QKV만 반환하고 hidden을 버리면 다음 residual을 재구성할 수 없다.

## Vulkan에 남는 기능

Q/K head norm, SWA에 적용되는 RoPE, 기타 attention-side rotation, KV write/cache/state/attention, mask, prefix checkpoint, 모델 최종 head·sampling. 이 모듈은 KV 또는 prefix state를 소유/수정하지 않는다.

특히 공개 Maple graph의 `build_attn(...wo...)`는 O projection까지 포함한다. 그 반환값을 이 모듈의 `attention`에 넣으면 O가 두 번 적용된다. O 앞의 tensor를 경계로 잡아야 한다. `integration/ggml_level4_contract.hpp`의 `is_plain_o_projection`은 plain graph에서 해당 source 연결을 검사한다.

QKV의 출력 폭은 Maple에서 서로 다르다. 여기서는 기존 arithmetic을 재사용하여 **Q 한 경로 + K/V Pair 경로**로 실행한다. A8 모드의 QKV input quant는 한 번만 실행하고 두 경로가 공유한다. 가중치를 새 QKV 거대 tensor로 복제하지 않는다. '단일 SYCL island'와 '단일 GPU kernel'은 다르다.

## 기존 코드 보존

`src/maple_moe.cpp`, `src/maple_w2a8.cpp`, `src/maple_w2a8_grouped.cpp`, `src/maple_w2a16.cpp`, `src/expert_grouping.cpp`, `tools/grouping_compare.cpp`를 덮어쓰지 않는다. 기존 T1/T2/T4, static grouping, split-K, G32/H32, RNE, gluquant 경로를 호출한다. 이전 결과 bundle에서 달랐던 merged `maple_moe.cpp`와 `grouping_compare.cpp`도 보존 대상이다.

새 RMSNorm/router는 일반 SYCL FP32 kernel이다. Router는 FP32 dot → 전체 softmax → top-k → 재정규화다. 동점은 lower expert ID 우선이다. 이 tie-break와 reduction 순서가 기존 ggml와 bitwise 동일하다고 가정하지 않는다. 실제 router IDs/scores와 최종 logits 비교가 별도로 필요하다.

## 정밀도

MoE는 현재 W2A8 G32/H32 gluquant를 유지한다. 새 O/QKV는 **기본 A16**, A8 G32는 각각 명시적으로 선택한다. 새로운 Q/K activation 양자화까지 '기존에 검증됐다'고 취급하지 않기 위해서다. A16도 기존 Vulkan F32/FP16 reduction과 bitwise 동일하다는 보증은 없다.

입출력/Norm/Router는 F32다. Norm 가중치가 F16이면 loader에서 작은 norm vector만 F32로 변환해 상주시킨다. TQ2 원본은 load 시 strict ternary/finite scale 검증을 거쳐야 한다. Bias, active LoRA, control vector, 별도의 scale/rotation이 island 내부에 있으면 기본 recognizer contract가 거부한다. 조용히 생략하지 않는다.

## 실행/메모리 API

`make_plan()`이 필요한 scratch 크기와 offset을 계산하고, `bind_workspace()`가 미리 할당한 64-byte-aligned GPU arena에 view를 만든다. `enqueue()` 안에서는 GPU allocation/import, host H2D/D2H, 필수 host wait를 수행하지 않는다. host metadata vector/map 사용은 있다.

실제 입력이 `[D,Q]` 또는 `[head_dim,heads,Q]`이고 head/token stride가 다르면 GPU pack으로 내부 contiguous view를 만든다. QKV 출력은 별도 배열 또는 token-interleaved QKV slice를 지원한다. 외부 allocation을 공유하되 필요한 **GPU 안의 layout copy는 남는다.** Zero-copy interop을 모든 device-local copy의 제거와 혼동하지 않는다.

아래 scratch는 weights/KV/외부 I/O를 제외한 **한 in-flight island** 기준이다. 동일 arena는 layer/chunk 사이에 재사용하고 동시 요청에는 별도 arena 또는 완료 dependency가 필요하다.

| Q | 모든 split=1 | gate split4/down split2 |
|---:|---:|---:|
| 1 | 233,920 B | 496,064 B |
| 13 | 3,006,784 B | 6,414,656 B |
| 64 | 14,792,832 B | 31,570,048 B |
| 184 | 42,525,568 B | 90,760,064 B |
| 512 | 118,327,424 B | 252,545,152 B |
| 2048 | 473,303,168 B | 1,010,174,080 B |

Q2048은 기본 약 451.4 MiB, split4/2 약 963.4 MiB다. 기존 MoE 진단 plane까지 유지하므로 VRAM이 부족하면 Q/ubatch와 split scratch를 포함해 판단한다. 아직 tensor lifetime aliasing으로 메모리를 극단 압축하지 않았다.

## Handoff와 import

`level4_bridge.hpp::execute_host_fenced()`는 기존 allocator에서 import된 동일 물리 메모리를 사용한다. 호출자는 기존 Vulkan 구현의 실제 `release_and_wait`, `acquire_after_sycl`, `poison`, allocation lease를 제공해야 한다. 누락된 callback은 거부한다. 다만 no-op lambda가 실제 GPU barrier를 기록했는지 API가 자동 증명하는 것은 아니므로, 호출부 roundtrip 검증이 필요하다.

- 진입: Vulkan write 완료 + 필요한 external ownership release를 한 번 처리.
- 내부: 한 SYCL enqueue API로 전체 chain 제출. mandatory stage wait 없음.
- 종료: SYCL done을 한 번 기다린 뒤 Vulkan acquire를 한 번 처리.

이 초기 경로는 **CPU fence relay**다. GPU semaphore-only 동기화로 구현했다고 주장하지 않는다. Host-fence 자체의 runtime 비용은 실측해야 한다. SPIR-V/Vulkan semaphore를 SYCL event로 cast하지 않는다.

`integration/level4_win32_memory.hpp`는 선택적 loader-time OPAQUE_WIN32 → Level Zero import helper다. SYCL queue의 native context/device에 import하며 borrowed HANDLE을 duplicate한다. 기존 shared allocator가 이미 import owner를 관리한다면 이 helper로 재-import하지 말고 기존 view/lease를 사용한다. Wrapper는 Vulkan export allocation이나 graph hook을 생성하지 않는다.

실제 physical device 일치와 allocation exportability는 호출부에서 검증한다. `VkBuffer binding offset + tensor view offset`까지 합친 범위를 imported base에서 계산해야 한다. Buffer handle/device address를 USM pointer로 재해석하지 않는다. Generation이 바뀐 arena view를 재사용하면 거부한다. 호출자는 반환 후에도 출력의 마지막 Vulkan consumer가 완료될 때까지 allocation lease를 유지해야 한다. Acquire barrier를 기록한 것만으로 downstream GPU 사용이 끝난 것은 아니다.

Submission 실패 후 즉시 같은 output에 Vulkan fallback을 실행하지 않는다. queue를 drain하고 region을 poison한 뒤 원래 오류를 전달한다. 상태 plane은 device에 남으며 invalid 계산은 NaN을 출력한다. `Run.done`은 완료이지 모델 품질 합격이 아니다.

## ggml 연결 순서

1. 현 graph에서 의미가 확인된 구간만 선택한다. Interior tensor에 외부 consumer가 있으면 거부한다.
2. 모델 로드 단계에 weight 변환/검증 및 imported allocation cache를 준비한다. 매 호출 import 금지.
3. `GraphSemantics`를 실제 graph 분석 결과로 채운다. 기본값 false를 무조건 true로 바꾸는 것은 검증이 아니다.
4. layer L의 O/FFN과 layer L+1의 norm/QKV를 `Weights`에 바인딩한다.
5. 실제 tensor stride와 import offset을 `Bindings`로 매핑한다.
6. existing Vulkan release/acquire 구현으로 `VulkanHandoff`를 연결하고 `execute_host_fenced`를 호출한다.
7. 이 호출이 맡은 ggml interior nodes만 원래 executor에서 skip한다. Attention/KV 영역은 그대로 실행한다.
8. 최종 layer의 `inp_out_ids` 처리는 경계 밖에서 보존한다. 선택된 FA/residual을 함께 gather한 뒤 terminal을 호출하거나 해당 tail은 기존 Vulkan에 둔다. 선택을 조용히 버리지 않는다.

이 패키지는 구간명만으로 node를 skip하는 위험한 자동 패치를 하지 않는다. CMake에 library가 추가됐다고 dispatcher 등록이 끝난 것은 아니다.

## 검증 순서

CPU → SYCL-only probe → imported-memory roundtrip → 실제 graph route-ID/hidden/QKV parity → 실제 logits/장문 → profiler-off E2E. 각 단계의 합격을 다음 단계로 확대하지 않는다.

`level4_probe`의 AB/BA 두 arm은 같은 SYCL 구현을 사용한다. 하나는 stage마다 host wait, 다른 하나는 마지막에만 wait한다. 이것은 orchestration 검증이지 Vulkan interop 절감 실측이 아니다. 타이밍 출력 후의 D2H 검사/CPU reference는 성능 측정 구간 밖이다.

`stages.csv`는 raw native timestamps와 dependency bit mask를 보존한다. `analyze_level4_trace.py`는 interval union, overlap, gap을 계산한다. Stage sum ≠ elapsed이며 서로 다른 API의 clock을 그냥 빼지 않는다.

## 원본 확인에 사용한 자료

- 사용자 업로드 `maple-w2-xmx-v0.4-source.zip`
- 이전 배포 `maple-w2-xmx-v0.4-vulkan-port-v1.zip`
- 사용자 실측 `maple-w2-xmx-v0.4-full-20260918.zip` (raw logs/hash; 최신 merged source 원문 아님)
- 공개 graph 참고: demilenos/llama.cpp, `fed6590fd2e64da55baa4a31788dfb12454b1844`, src/models/maple.cpp. 현재 merged tree와 동일하다고 가정하지 않음.
- https://huggingface.co/deepgrove/maple-preview/blob/main/modeling_maple.py
- https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/mem.html
- https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-0/intel-oneapi-level-zero-backend-specification.html
