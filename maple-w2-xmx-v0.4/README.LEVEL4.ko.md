# Level 4 추가 패키지 — XMX v0.4 + architecture port v1 기준

**추가 실행 모듈 및 연결 API. 현재 llama.cpp graph hook 자동 등록본이나 교체 DLL이 아니다.** 새 파일만 이식하며 이미 병합한 DPAS/MoE 코드는 유지한다. 전체 범위와 미검증 부분은 `docs/LEVEL4.ko.md` 참조.

## 기존 module에 적용

압축을 별도 디렉터리에 풀고, `--root`에 `include/maple_moe.hpp`, `src/maple_moe.cpp`가 있는 **기존 XMX module root**를 지정한다. 전체 llama.cpp root가 반드시 module root인 것은 아니다.

```powershell
python .\tools\apply_level4_upgrade.py --root "C:\AI\llama-src\YOUR_W2_MODULE"
python .\tools\apply_level4_upgrade.py --root "C:\AI\llama-src\YOUR_W2_MODULE" --apply
```

첫 명령은 dry-run이다. 기존 Level4 파일에 다른 수정이 있으면 거부하고, 기존 MoE/DPAS 소스를 덮어쓰지 않는다. Module의 CMake target을 식별하지 못하면 아무것도 변경하지 않으며, `--no-cmake`로 새 파일만 추가한 뒤 현재 빌드에 `src/maple_level4.cpp`를 포함할 수 있다. 신규 target은 `maple-level4`, 기존 dependency는 `maple-w2-xmx`다.

`ggml_level4_contract.hpp`와 `level4_bridge.hpp`는 호출부 연결 API다. 아직 현 ggml의 span recognizer, node skip 및 allocator callbacks를 연결해야 **서버에서 실제 Level4가 실행**된다. 단순히 ZIP을 병합했다고 interop 경계가 줄었다고 판단하지 않는다.

## CPU 확인 (module root)

```powershell
cmake -S . -B build-level4-cpu -DMAPLE_BUILD_SYCL=OFF
cmake --build build-level4-cpu --config Release
ctest --test-dir build-level4-cpu -C Release --output-on-failure
```

## Intel GPU 확인 (module root)

```powershell
.\build_level4_and_test.cmd -Device A750
.\build_level4_and_test.cmd -Device A750 -MapleShape -Tokens 13 -Schedule direct
.\build_level4_and_test.cmd -Device A750 -MapleShape -Tokens 64 -Schedule grouped
```

첫 실행은 H256/E8/top2의 synthetic smoke다. `-MapleShape`는 H2048/FFN512/E256/top8 형상이며 **여전히 합성 가중치·입력**이다. `-DenseA8`를 추가하면 새 O/QKV도 A8로 바뀌므로 정밀도 실험으로 분리한다. MoE G32/H32는 그대로다.

Build-only:

```powershell
.\build_level4_and_test.cmd -Device A750 -BuildOnly
```

한 번 빌드한 실행기로 첫/끝 경계도 확인:

```powershell
.\build\maple-level4-probe.exe --device A750 --entry bootstrap --tokens 3 --out build/l4-bootstrap
.\build\maple-level4-probe.exe --device A750 --entry terminal --tokens 3 --out build/l4-terminal
```

검사 결과: `numerical.csv`, `samples.csv`, `stages.csv`, `summary.json`, `SCOPE.txt`. 빌드 식별자: `build/level4-build.json`. GPU probe 실패 시 더 느슨한 tolerance나 다른 backend로 자동 대체하지 않는다.

선택적 Windows import helper는 `level_zero/ze_api.h` + `sycl/ext/oneapi/backend/level_zero.hpp`, ze_loader.lib가 필요하다. 일반 Level4 GPU probe는 Vulkan SDK나 Win32 import helper를 호출하지 않는다. 따라서 probe PASS도 실제 Vulkan↔LevelZero roundtrip 합격이 아니다.

## API 예시

```cpp
namespace l4 = maple_w2::level4;
l4::Config cfg; cfg.tokens = actual_token_count;
cfg.entry = l4::Entry::advance;
l4::Options opt;
opt.moe.schedule = maple_w2::MoeSchedule::auto_select;
opt.moe.auto_policy = {64, 2}; // 기존 A750 T1 실측 기반의 노출된 provisional policy
// Split/token-tile은 현재 v0.4 설정을 넘긴다. 여기서는 기본 T1/split1.
auto plan = l4::make_plan(cfg, opt.moe.gate_kernel.split_k,
    opt.moe.down_kernel.split_k, opt.o_kernel.split_k,
    opt.qkv_kernel.split_k, opt.moe.gate_tokens_per_tile,
    opt.moe.down_tokens_per_tile);
auto workspace = l4::bind_workspace(resident_sycl_arena, arena_capacity, plan);
// weights.current_layer=L, next_layer=L+1을 포함하여 실제 weights/bindings를 구성.
auto run = l4::enqueue(existing_sycl_queue, weights, bindings, opt, workspace,
                       incoming_sycl_dependencies);
// 실제 Vulkan 연동은 level4_bridge.hpp의 ownership callbacks와 shared lease 사용.
```

기존 helper에 익숙하지 않은 새 부모 구현에서 위 이름을 임의 생성해 즉시 호출하지 않는다. `resident_sycl_arena`와 weight/I/O pointer는 해당 queue context의 실제 USM/imported allocation이어야 한다.

## 현재 검증 수준

이 환경에는 Intel DPC++/GPU/Windows SDK가 없다. C++ host/CPU reference와 test doubles로 검사했다. **SYCL device compile·A750 실행·import·실제 모델 속도/품질은 미검증**이며 상세 내역은 `validation-level4/VALIDATION.ko.md`에 기록한다.
