# CUDA portability audit

Scope: distinguish CUDA-inspired ideas from measured local Vulkan/A750 results. Similarity does not establish a CUDA port, and no matched local CUDA benchmark is available for the Bonsai2 target.

## Validated current evidence

- The CUDA-style hybrid FWHT idea was implemented as a separate A750 Vulkan experiment. Matched TG128 OFF was `14.682107 +/- 0.011661`; hybrid ON was `14.711299 +/- 0.006892` (~0.199% gain).
- The pre-final sign-fusion candidate measured TG128 `14.780515 +/- 0.002712` and TG1024 `14.693389 +/- 0.005619`.
- Whole-graph signed validation now passes 33 cases, including six shapes and fanout fallback, with the signed pipeline. The final compound WG64/rows4 + hybrid/signs + submit configuration measured TG128 `15.864460 +/- 0.007227` and TG1024 `15.687187 +/- 0.007388`. Those results are compound A750 Vulkan configuration results; they cannot be attributed solely to CUDA-derived FWHT or sign fusion.

## Historical snapshots and measured regressions

- The earlier hybrid/sign implementation notes that said GPU validation was pending are historical snapshots and are superseded by the whole-graph result above.
- Vulkan tuning regressions are real but are not CUDA-port regressions: WG128/rows16 TG32 `10.72` and WG64/rows16 `10.25` versus roughly `14.49`; square microkernel WG64/rows8 `115.21 us` versus rows4 `78.43 us`. Provenance: `tg-mmv-wg128-r16-cpu-tg32.log`, `tg-mmv-wg64-r16-cpu-tg32.log`, and the corresponding square microkernel logs.
- CUDA DP4A/MMQ structural ideas remain unvalidated for this target. Existing CUDA `vecdotq.cuh` and `mmq-load-tiles.cuh` use different signed-byte tile and activation-quantization arrangements; importing them would require new numerical and quality validation.

## Remaining genuine port possibilities

- The CUDA hybrid shuffle/shared decomposition (`ggml-cuda/fwht.cu:175-203`) is already ported and measured above; it is not an outstanding item.
- CUDA sign fusion (`fwht.cu:158-165`, graph predicate in `ggml-cuda.cu:3480`) is already ported, whole-graph validated and measured above; it is not an outstanding item.
- CUDA multi-column decode reuse is relevant to N>1 batching and does not directly solve N=1 TG.
- A full CUDA MMQ signed-byte tile port is a larger experiment, not a safe direct transplant.

The current A750 evidence is Vulkan-specific. It does not certify CUDA performance, CUDA hardware utilization, or the CUDA-relative 50% target.

## Provenance details

- CUDA `vecdotq.cuh:817` reuses widened packed bytes across five digits. The Vulkan float byte/lane mapping is independently implemented; related algorithm does not mean literal CUDA kernel transplant.
- The packed-power accessor originated in the other local Vulkan fork, not CUDA. Exact digit equivalence was tested.
- XMX/CM1 tiling originated in Vulkan/local optimized code. Serialized PP512 reached 105.867 tok/s but severely regressed decode; unserialized hangs are failures, not performance results. It is excluded from the production profile and must not be mislabeled a CUDA-port regression.
- Larger independent MMV workgroups also regressed: WG256/R4 13.496084, WG256/R8 14.067849, WG512/R4 11.718320 tok/s. Neither these nor submission-divisor regressions are CUDA ports.
