이 저장소의 llama runtime 통합 및 실기 검증은 [runtime 문서](../docs/maple-level4-zero-copy.md)를 참조하십시오. 아래 내용과 manifest는 원본 v1 overlay의 설명입니다.

# Level4 zero-copy v1 — Vulkan ↔ Level Zero/SYCL

## 범위와 상태

**대상은 기존 XMX 0.4 + Vulkan architecture port + Level4 upgrade v1 모듈입니다.**
사용자가 정한 Level4 경계는 그대로 유지합니다.

```
Vulkan FA(pre-O) → SYCL O / residual / norm / router / MoE /
residual / next norm / next QKV → Vulkan head norm / RoPE / KV / FA
```

이번 변경은 기존 callback 명세에 그치지 않습니다. 실제 `VkDeviceMemory` export,
`zeMemAllocDevice` external import, import 캐시, Level4 실행, Vulkan ownership acquire를 구현합니다.
연속 F32 경계에서는 기존 GPU input pack / output scatter도 제거합니다.

**현재 확인한 것은 CPU/host-double 검증입니다. Intel SYCL SDK 컴파일, Windows/A750 실기,
실제 외부 메모리 cache coherency, 전체 llama-server graph 실행 및 속도는 아직 미검증입니다.**
검증 보고서에 이를 분리했습니다. DLL/EXE 바이너리는 배포하지 않습니다.

## 적용

ZIP은 현재 소스와 **다른 디렉터리**에 풉니다. `--root`에는 `include/maple_moe.hpp`,
`src/maple_moe.cpp`, `src/maple_level4.cpp`가 있는 XMX 모듈 디렉터리를 지정합니다.

```powershell
# 먼저 파일 hash/충돌 확인. 실제 파일은 바꾸지 않습니다.
python .\tools\apply_level4_zero_copy.py --root "현재 XMX 모듈 경로"

# dry-run 확인 후 적용
python .\tools\apply_level4_zero_copy.py --root "현재 XMX 모듈 경로" --apply
```

모르는 수정본은 덮어쓰지 않습니다. CMake의 zero-copy include 추가만 기존 사용자 CMake 변경을 보존해
append할 수 있습니다. 이전 파일은 `build/level4-zero-copy-backup-*`에 보관합니다.
기존 `maple_moe.cpp`, DPAS, grouping, T1/T2/T4 코드는 수정하지 않습니다.
`Options`, `Run`, `Plan` 헤더가 바뀌었으므로 관련 모듈·호출부는 **같이 재빌드**하십시오.

일반 unified diff도 제공합니다. 패치 경로는 **XMX 모듈 루트 기준**입니다.
llama.cpp 루트에서 적용하는 경우 실제 모듈 하위 경로에 `--directory`를 맞추거나 위 적용기를 사용합니다.
완성된 소스 파일들을 원본 llama.cpp 루트에 무작정 덮어쓰지 마십시오.

## Windows 실기 probe

필수: MSVC + Intel oneAPI DPC++/ESIMD, Vulkan SDK, Level Zero SDK.
`VULKAN_SDK`, `LEVEL_ZERO_V1_SDK_PATH` 또는 `LEVEL_ZERO_SDK_PATH`가 실제 SDK 루트를 가리켜야 합니다.

적용한 **모듈 디렉터리**에서:

```powershell
# 작은 형상, native Vulkan fill → Level4 → Vulkan readback 소비
.\build_level4_zero_copy.cmd -Device A750 -Validation

# Maple 형상의 단일 island. 이것도 합성 가중치/입력입니다.
.\build_level4_zero_copy.cmd -Device A750 -MapleShape -Tokens 13 -Schedule direct -Validation

# 구조상 첫/마지막 layer 경계도 별도 검사
.\build_level4_zero_copy.cmd -Device A750 -Entry bootstrap -Validation
.\build_level4_zero_copy.cmd -Device A750 -Entry terminal -Validation

# SDK 경로가 환경변수에 없다면 명시
.\build_level4_zero_copy.cmd -Device A750 `
  -VulkanSdk "실제 Vulkan SDK 루트" -LevelZeroSdk "실제 Level Zero SDK 루트"
```

빌드만 하려면 `-BuildOnly`, 새로운 dense O/QKV A8까지 검사하려면 `-DenseA8`입니다.
기본 O/QKV는 기존 Level4의 A16, MoE는 기존 A8 정책입니다.
probe는 `s2×s8` DPAS 검사도 먼저 수행하며 OpenCL/CPU/host-copy로 조용히 우회하지 않습니다.

CMake로 빌드하려면 이미 oneAPI compiler가 선택된 구성에서:

```text
-DMAPLE_BUILD_SYCL=ON -DMAPLE_LEVEL4_ZERO_COPY=ON
-DMAPLE_LEVEL_ZERO_INCLUDE_DIR=<.../include>
-DMAPLE_LEVEL_ZERO_LIBRARY=<.../lib/ze_loader.lib>
```

추가 target은 `maple-level4-zerocopy`와 `maple-level4-zerocopy-probe`입니다.
Linux의 기본 CPU CMake 빌드는 SDK 없이 계약·test-double 검사만 합니다.

## 무엇이 zero-copy인가

| 항목 | 새 production executor |
|---|---|
| Vulkan → CPU tensor readback → SYCL upload | 없음 |
| 매 island Win32 export / ZE import | 없음. setup/cache에서 수행 |
| 연속 F32 input pack | 없음. import pointer 직접 읽기 |
| 연속 F32 hidden/Q/K/V output scatter | 없음. export buffer에 직접 쓰기 |
| O/MoE/QKV 중간 계산·activation quantization | 여전히 실행. 이는 boundary copy가 아님 |
| invalid token 상태 처리 | 작은 direct output status kernel 유지 |
| CPU 대기 | **있음.** inbound Vulkan fence + SYCL tail wait |
| 외부 semaphore 기반 GPU-only handoff | 구현하지 않음 |
| 드라이버 내부 실제 PCIe 전송량 | 아직 측정하지 않음 |

`create_buffer()`는 device-local Vulkan allocation을 export하도록 생성합니다.
`adopt()`는 기존 **이미 exportable인** VkBuffer/메모리를 등록합니다.
기존 non-exportable allocation에 플래그만 나중에 붙여 공유할 수는 없습니다.
공유 arena 하나에 여러 tensor를 넣을 수 있으므로 tensor마다 allocation할 필요는 없습니다.

## 실제 graph 연결

이 패치는 Level4 **모듈**과 실제 전송 계층을 업그레이드합니다.
이전 배포와 마찬가지로 현재 사용자 llama.cpp tree의 전체 graph matcher/노드 skip hook은 포함하지 않습니다.
이미 연결한 Level4 호출부에서는 아래 세 부분을 바꿉니다.

1. graph 준비 시 exporter/기존 exporter adoption을 사용해 boundary tensor arena를 생성하고 `Prepared`를 캐시합니다.
2. 런타임에서 선행 Vulkan producer command buffer를 같은 queue에 **실제로 submit**합니다.
3. 이전 callback bridge 대신 `Context::execute(prepared)`를 호출하고, 같은 Vulkan queue로 후속 노드를 submit합니다.

메모리를 공유하는 것만으로 노드 skip이나 interop 동기화가 자동 완성되지는 않습니다.
`docs/LEVEL4_ZERO_COPY_INTEGRATION.ko.md`에 실제 API 연결 코드를 넣었습니다.

## 결과와 해석

`build/level4-zero-copy-*/`에 `probe.log`, `samples.csv`, `numerical.csv`, `SCOPE.txt`,
source/binary hash stamp, `zero-copy-audit.json`이 남습니다.

- `inbound_host_us`: 이전 acquire retirement + Vulkan release/fence 완료.
- `sycl_host_us`: Level4 enqueue부터 마지막 SYCL event 완료.
- `acquire_submit_us`: Vulkan acquire 기록·제출. consumer 완료 대기 아님.
- `total_host_us`: 위 세 구간 합. 내부 계측 구간이며 prevalidation/lock 획득/반환 시 lease 정리는 제외.
- `execute_wall_us`: 호출 직전~반환 직후의 전체 host wall. 위에서 빠진 비용도 포함.
- `gpu_sycl_span_us`: SYCL event clock의 첫 시작~마지막 종료. host clock에 더하지 않습니다.
- `boundary_d2d_bytes=0`, import/export count 불변을 검사합니다.

probe의 가중치 setup upload와 **진단용 Vulkan readback**은 measured island 밖에 있습니다.
입력은 Vulkan fill로 만든 token별 uniform 값이므로 cache visibility 확인용이지 representative performance 입력이 아닙니다.
합성 baseline과 all hidden/Q/K/V 출력을 비교하며 모델 PPL·logit KLD·장문 코드 품질은 측정하지 않습니다.

## 이전 동작 보존

기존 `Options.boundary` 기본은 `packed`입니다. zero-copy executor만 `direct_required`를 강제합니다.
비연속 view나 입력/출력 alias는 제출 전에 거부하며, 몰래 copy를 넣거나 tolerance를 늘리지 않습니다.
일반 packed Level4 경로와 zero-copy 실행을 같은 소스로 A/B할 수 있습니다.
