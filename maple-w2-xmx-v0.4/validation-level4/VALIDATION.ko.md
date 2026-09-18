# Level 4 upgrade v1 — 실제 검증 내역

이 패키지의 검증은 Linux CPU 환경에서 수행했다. **Intel DPC++ device compile, Windows PowerShell 실행, A750, Vulkan/LevelZero roundtrip, 실제 llama.cpp graph 실행 및 모델 품질/속도는 검증하지 않았다.**

| 검사 | 실제 결과 | 로그 |
|---|---|---|
| GCC Release 전체 CPU CTest | 10/10 PASS | ctest-gcc-final.log |
| Clang 17 Release C++ tests | 9/9 PASS | ctest-clang-cpp.log |
| Python tests | 54개 PASS, 이전 48 + 신규 적용기 6 | python-tests-final.log |
| 신규 경계·planner CPU 테스트 | 107,886 assertion checks | sanitizers.log |
| 실제 Level4 lambda/host DAG + CPU Dense/MoE 대체 구현 | 14,976 assertion checks | sanitizers.log |
| 신규 두 C++ test의 ASan+UBSan+leak 검사 | PASS | sanitizers.log |
| 새 source/probe의 Clang host parsing (SYCL test double) | PASS | host-syntax-final.log, probe-host-syntax-final.log |

Assertion 수는 GPU 실행 횟수가 아니다. Level4 host test는 실제 `maple_level4.cpp`의 새 scalar lambda를 WG=1 collective double로 실행하고, 기존 DPAS/MoE는 CPU reference stub으로 대체한다. **WG64의 GPU reduction, device address space, 실제 native DPAS, SYCL compiler/device ABI를 검증한 결과가 아니다.** Probe host parsing 역시 GPU device compilation의 대체가 아니다.

신규 적용기 테스트는 dry-run, 기존 수동 수정 보존, idempotence, hash 불일치, 충돌 파일, 누락 interface, CMake 식별 실패를 검사한다. 실제 배포 overlay를 별도 기준 트리에 적용하고 재빌드했다. 23개 신규 파일의 hash가 일치했고, CMake 이외 기존 145개 파일과 두 수동 변경 sentinel을 모두 보존했다. 적용 후 CTest 10/10 PASS. `overlay-apply-check.log`, `applied-build.log`, `applied-ctest.log` 참조.

## 구현 상태

실제 작성: post-attention pre-O → O/residual/norm/router/MoE/residual/next norm/QKV를 실행하는 SYCL 모듈, bootstrap/terminal, strided F32 boundary pack, resident workspace planner, explicit handoff API, optional Win32 import helper, stage validation·raw timestamp GPU probe.

미완료: 최신 merged ggml graph의 span recognition·interior node skip 등록, 사용자 shared allocator와 ownership callbacks 연결. 이 부분을 자동 적용하는 patch는 없으며, 모듈 추가만으로 llama-server의 경계 수가 줄어들지 않는다. Readme와 integration header가 호출 계약을 제공한다.

SYCL-only probe의 segmented arm은 고의적인 host wait 진단이다. 특히 MoE는 기존 enqueue가 내부 DAG를 먼저 제출하고 반환한 stage events에 wait를 적용하므로 기존 Vulkan 세분 경계 실행의 정확한 복제본이 아니다. 최종 성능은 실제 Vulkan↔SYCL bridge에 연결한 뒤 별도 A/B한다.

## 보존 및 안전성

기존 XMX/DPAS/MoE/grouping 파일은 byte-level 동일하게 보존했다. 패키지는 새 파일만 추가하고 CMake include 한 줄 블록만 append한다. 사용자 실제 merged `maple_moe.cpp`와 `grouping_compare.cpp`를 예전 배포본으로 덮어쓰지 않는다. DLL·모델·서버 설정·Git 이력 변경은 없다.

추가 library의 runtime/GPU 성공을 확인한 뒤에만 실서버 dispatch를 활성화한다. Unsupported graph semantics는 실행 전에 fallback 결정하고, 이미 제출된 island가 실패한 뒤 같은 buffer로 즉시 fallback하지 않는다.
