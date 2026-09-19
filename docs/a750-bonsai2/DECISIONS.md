# Decisions — A750 Bonsai2 27B

Record adopted decisions with evidence; distinguish experiments from adoption.

## 2026-09-19: Decode target and acceptance criteria

- Adopt TG128 >= 15 tok/s as the decode performance target. Keep the original GGUF; require quality regression <= 5% relative to the validated original Prism path on a documented evaluation. A short PPL evaluation is a proxy, not a universal capability guarantee.
- Keep the requested deployment port 9931 and Q8_0 K/V cache. Choose context from measured request-time dedicated VRAM, leaving at least 512 MiB. Idle allocation alone is insufficient: context36608 passed idle (514.5 MiB remaining) but failed during a request (500 MiB).
- CPU token embedding placement remains a candidate, pending TG and memory verification. PP512 r3: GPU105.115613 versus CPU104.992098 tok/s (0.12% lower); this alone does not establish the requested <=5% impact for decode.
- Investigate serialization before writing a new kernel. Current PP workaround (XMX + serialize + maxnodes1) measured TG32=1.685529 tok/s. Removing the workaround is an experiment, not an adopted configuration.
- Use sequential GPU measurements so candidate processes do not contend. Use compact bounded diagnostics and record failed/interrupted runs explicitly.

Existing source changes and earlier evidence are documented in RESULTS.md. No previous tuning result is treated as proof of the new TG128 target.
## Adopted investigation direction: PTQ1 decoder optimization

- Evidence: normal scheduling with noXMX TG128 r2 is 2.998743 ± 0.000349 t/s (pp-fix/tg-default-schedule.log), versus 1.685529 t/s for prior serializedTG32.
- Profile: pp-fix/tg-default-profile.log attributes approximately 90% of time to PTQ1 matvec (about 0.31 s/token).
- Decision: serialization is not the sole root cause. Optimize the existing PTQ1 integer-dot decoder with exact weights; do not apply further weight quantization.
- Adoption gate: require GPU/CPU-reference agreement and model-quality validation before adoption.
- Status: helper-source edit candidate is pending validation and is not an accepted solution.
## Candidate: PTQ1 float byte-reuse decoder

- Investigation: exact float matvec reusing PTQ1 packed bytes, decoding each byte once for five trits plus the qh tail; preserve stored FP16 scales, Hadamard metadata, and weights. No format conversion or extra VRAM.
- Evidence: integer helper CPU test passed at 149.42 us (TG32=8.09 under normal noXMX); float candidate measured TG32=12.64 with noXMX/MMVQ disabled. One tg1 kernel measured 77.39 us; six CPU tests passed (base 5120x5120 plus five cases). No MMVQ activation quantization.
- Status: promising but not final; adoption requires TG128 >= 15 tok/s plus quality validation.

## Backup candidate: PTQ1-to-PQ2 format conversion

- Use only if the PTQ1 decoder candidates fail. Convert exact trits to group-128 PQ2 while preserving each FP16 scale and Hadamard metadata.
- PTQ1 blocks are 28 bytes per 128 weights; PQ2 blocks are 34 bytes, about 1.214x storage: 5.936 GB becomes approximately 7.21 GB. CPU-resident embeddings may reduce GPU pressure.
- Vulkan gaps: add a PQ2 group-128 decoder and q8_1 dispatch by adapting Q2_0’s four 32-element chunks; verify one PQ2 scale per 128 weights and q8_1 scale/sum alignment. Existing Q2_0 q8_1 registrations do not cover PQ2.
- Status: investigation-only fallback; adoption requires CPU-reference, GPU numerical, and model-quality validation.

## Quality gate for decode candidates

- Use WikiText2 PPL ratio <= 1.05 versus the validated baseline as the acceptance proxy, with KLD recorded; broader quality is not certified by this short test.
- Run the candidate with ubatch=1 so the actual TG matvec path is exercised; PP512 PPL does not test the decode path.
## Investigation: A750 N=1024 FWHT shared-memory path

- Float candidate remains below target at TG32=12.64 tok/s; six numerical cases passed, but this is not adoption evidence.
- Audit finding: the Hadamard N=1024 subgroup-32 path uses 32 values per thread across four rows per workgroup. Small TG shapes (5–17 rows) create only 2–5 workgroups.
- Probe candidate: enable the existing shared-memory FWHT kernel only for A750/N=1024 via `GGML_VK_A750_FWHT_SHMEM`, keeping the math unchanged; it uses 256 threads and one row per workgroup.
- Status: pending probe, performance, and correctness validation. Do not claim a dense fallback or actual routing from profile labels alone.
## Candidate evidence: A750 PTQ1 matvec configuration

- Best measured TG32 is 14.49 tok/s with six numeric cases passing.
- Status: pending TG128 validation at the actual target workload and the WikiText2/KLD quality gate; this is not final adoption evidence.
## Quality and TG128 candidate evidence

- ubatch=1 Q8 WikiText2/KLD proxy completed across four chunks: PPL 7.955312 to 7.941050, ratio 0.998207; proxy passes the <=1.05 gate, but this is not universal quality certification.
- TG128 candidate measurement is 14.49 tok/s on both GPU and CPU embedding variants. It remains below the 15 tok/s target; row16 testing is still in progress, so the final configuration is not locked.
## Shared-memory guard and current evidence

- Commit `78573509a` adds a guard that falls back based on the actual maximum shared-memory limit.
- `rows16` with workgroup 128 measured 10.72 tok/s and is rejected; retain `rows8` pending further validation.
- The six existing source commits are mirrored in the versioned README. The ubatch=1 WikiText2/KLD proxy passed (ratio 0.998207), and CPU embedding passed the TG comparison; broader quality and final context behavior remain pending.
## Additional candidate evidence

- `rows16` with workgroup 64 measured 10.25 tok/s and is rejected.
- Compact26 reached TG128 14.69 ± 0.01 tok/s and is promising; quality validation remains pending and there is no final adoption.
## Expanded acceptance criteria

- Require both TG128 >= 15 tok/s and TG1024 >= 15 tok/s; no 1K result is claimed until measured.
- Retain the Q8 KV quality proxy threshold at <= 5% regression.
- Preserve port 9931 and at least 512 MiB request-time dedicated-memory reserve.
- Require CPU-embedding impact <= 5% versus the GPU-embedding comparison.
## Adopted incremental improvement: compact26

- Adopt commit `d42e1e002` as an incremental improvement: TG128 CPU improved from 14.49 to 14.69 ± 0.01 tok/s, with six numerical cases passing and mapping proof recorded.
- Compact26 quality passed the four-chunk, ubatch=1, Q8 proxy: PPL 7.940868 versus baseline 7.955312, ratio 0.998184.
- This is not final target adoption: TG128 >= 15 remains unmet, TG1024 is unmeasured, and broader quality remains uncertified.
## CPU token embedding and CUDA FWHT candidates

- Adopt CPU token embedding as eligible for VRAM optimization in the current compact26 normal configuration: PP512 GPU 77.870159 +/- 0.247380 versus CPU 77.682724 +/- 0.125910 tok/s, approximately 0.241% loss. Earlier TG128 GPU/CPU comparison was 14.49 tok/s for both. Original embedding weights are unchanged; final-context behavior remains pending.
- Candidate only: CUDA FWHT intra-subgroup shuffle path in `ggml-cuda/fwht.cu` lines 175–203. It is not adopted until GPU correctness and performance are verified.
## CUDA hybrid FWHT candidate

- Commit `14c1d20c7` passes the Hadamard-27 check. TG128 with hybrid FWHT is 14.711299 +/- 0.006892 tok/s versus matched OFF 14.682107 +/- 0.011661, about a 0.199% gain.
- TG1024 hybrid is 14.603291 +/- 0.009334 tok/s, below the 15 tok/s requirement. Treat hybrid as a usable incremental candidate with the target still unmet; final sign fusion remains pending.
- Wider MMV settings are rejected as defaults: WG256/R4 13.496084, WG256/R8 14.067849, WG512/R4 11.718320 tok/s. Retain opt-in reproduction controls from commit `f45655370`.
## Sign-fusion candidate and submit controls

- Sign-fusion candidate measured TG128 14.780515 +/- 0.002712 and TG1024 14.693389 +/- 0.005619, with a real profile showing 162 fusions/token.
- Initial 33 tests were nodewise only and are not fusion proof; corrected whole-graph validation remains pending. Independent submit-knob behavior is a Vulkan experiment and must not be treated as a CUDA port.
- Launchers expose submit divisor 40 and max nodes per submit 100 through `GGML_VK_A750_SUBMIT_DIVISOR` and `GGML_VK_MAX_NODES_PER_SUBMIT` for reproducibility.
## Whole-graph sign fusion and submit-grid evidence

- Whole-graph sign validation now passes: commit `bab57b4d3`, signed pipeline `fwht_signed_hybrid_f32`, six shapes plus fanout fallback, recorded in `fwht-signs-wholegraph33.log`.
- Submit knob commit `e54f521cb`. TG128 results: divisor40/nodes100 `14.773073`, divisor16/nodes256 `14.809493`, divisor8/nodes1024 `14.747310`; divisor4/nodes1024 `14.546931` tok/s. Target remains unmet; final MMV compact-grid selection is pending.
- GPU profile attributes 59.47137 ms of PTQ matvec out of 73.4305 ms instrumented (~81%); this is instrumented profile share, not proof of the wall-clock critical path.
## TG128 target candidate: WG64/rows4

- Candidate configuration: MMV workgroup 64, rows 4, signed/hybrid FWHT with FWHT workgroup 256, submit divisor 16, max nodes 256, Q8 KV, CPU embedding.
- TG128 measured `15.864460 +/- 0.007227 tok/s`, exceeding the 15 tok/s target and improving over the WG128/rows8 result near 14.81.
- This is measured candidate evidence, not a production-default decision. TG1024, server validation, and the quality gate remain pending; do not lock final defaults yet.
## Final compact-grid performance evidence

- Final signed MMV WG64/rows4 candidate reaches TG128 `15.864460 +/- 0.007227` and TG1024 `15.687187 +/- 0.007388` tok/s, meeting both measured throughput targets.
- Seven final numeric tests and whole-graph sign validation (33 cases) pass in the `final-signs-mmv64-r4-*` logs.
- CPU quality validation is still running and server validation has not started; production adoption remains pending those gates.
## Final CPU quality proxy

- Final CPU-embedding Q8/FA ubatch=1, four-chunk quality result: PPL 7.941728 versus baseline 7.955312, ratio 0.998293; proxy passes the <=5% threshold.
- This is a bounded quality proxy, not universal capability certification. Server validation and final deployment review remain pending.
## Final embedding, PP, and server-context evidence

- Final TG128 embedding comparison: CPU `15.864460 +/- 0.007227` versus GPU `15.845674 +/- 0.011085` tok/s; CPU is approximately 0.12% faster with no observed loss. Final PP512 r1 is CPU `78.031034` versus GPU `77.649063` tok/s; earlier matched PP3 showed a 0.24% CPU loss.
- Server context 43008 with a 267-token prompt and 128 generated tokens reached 15.605543 tok/s peak dedicated usage 7337.71 MiB, leaving 760.29 MiB; this passes the 512 MiB reserve criterion.
- Context 50176 with approximately 2057 prompt tokens and 128 generated tokens reached peak dedicated usage 7629.10 MiB, leaving 468.90 MiB; reject it for insufficient reserve.
- Context 48640 long-prompt TG1K testing is still in progress and is not final.
## Current CPU-only deployment draft

- CPU quality proxy remains 7.941728 versus 7.955312 baseline (ratio 0.998293). CPU/GPU embedding TG128 is 15.864460 versus 15.845674 tok/s; PP512 r1 is CPU 78.031034 versus GPU 77.649063.
- Normal noXMX PP is approximately 78 tok/s. The separate XMX/serialized PP result around 105.87 tok/s is not combinable with the normal configuration.
- Server context 48640 measured 15.282001 tok/s with 521.50 MiB reserve. Context 49152 failed at 503.97 MiB reserve. Context 48896 observed 512.74 MiB reserve but remains provisional pending confirmation.
- CUDA-relative 50% hardware utilization remains uncertified; no matched CUDA benchmark exists. Older 8080/PID 3700 server evidence is historical.
- Correction: `git rev-list 07c232edf..HEAD` counts 12 source commits after the base, not 13.
- Context 48896 completed at 15.288034 tok/s with 512.74 MiB reserve, but the 4096-word probe left only 500.27 MiB; reject full-use acceptance. Context 41472 near-full testing is in progress.

## 2026-09-19: full-context runtime memory gate and critical-path interpretation

- Reject context 48896 as a deployment default: the 4105-token prompt raises peak adapter dedicated usage to 7597.73 MiB, leaving 500.27 MiB of the 8098 MiB usable budget, below the required 512 MiB. The earlier 2057-token pass is insufficient.
- Test context 41472 near capacity before adoption. Startup graph reservation already models full KV, but execution can additionally allocate FA scratch, pipelines and driver resources; reserve-only startup is not a sufficient acceptance test. Source audit does not uniquely attribute the observed growth.
- Do not adopt the hypothesis that GPU work is no longer on the critical path. MMV WG64/rows4 improves decode by approximately 7%, while submission tuning is small. The instrumented ~81% matvec share is not a wall-clock critical-path measurement and cannot distinguish arithmetic, memory bandwidth, or CPU submission gaps.

## 2026-09-19: near-full 41472 result and larger final candidate

- Context 41472 passed 41225 input plus 128 generated tokens, with 636 valid VRAM samples and no failed samples: peak 7301.89 MiB, reserve 796.11 MiB. Evidence: server-memory-20260919-092539.json.
- Long-context decode is 11.31 tok/s, distinct from short-context TG128/TG1024 measurements exceeding 15 tok/s. Do not generalize the short-context target to all prefix lengths.
- Context 41472 is unnecessarily conservative for capacity. Test 48384 near full next; this is 512 tokens below the known failing 48896 setting. The actual measurements invalidate the earlier linear prompt-memory extrapolation; no such extrapolation is accepted as final memory proof.

## 2026-09-19: reproducible launcher environment and path handling

- Clear and restore inherited GGML_VK_DISABLE_FUSION and GGML_VK_SUBMIT_TRACE in launch/quality scripts so external shell state cannot silently disable signed fusion or add production trace overhead.
- Quote the model path in Start-Process arguments so WorkspaceRoot paths containing spaces are handled correctly.
- Validation: all three PowerShell scripts parse; a mocked launcher verified clearing, restoration, signed-fusion enablement and quoted model argument without launching a competing server. Evidence: launcher-mock-validation.log.

## 2026-09-19: adopted production configuration and final verification

- Adopt context 48384, port 9931, Q8_0 K/V, CPU token embedding, MMV WG64/rows4, signed hybrid FWHT WG256, submit divisor16/max256. Keep original PTQ1 weights and all transformer layers on Vulkan1; CPU embedding does not introduce quantization changes.
- Same-process memory stress passed sequentially: 48137 input +128 generation peak7567.73/reserve530.27 MiB; 2057+1024 peak7575.71/reserve522.29; 4105+128 peak7580.54/reserve517.46. Budget denominator is 8098 MiB usable driver/Vulkan memory, not nominal 8192 MiB. Highest tested accepted context is48384; this is measured headroom, not a guarantee under arbitrary concurrent GPU allocations.
- Final server TG1024 at2057 input:15.277944 tok/s; TG128 at4105 input:15.077958; near-full48137 input:10.868322. The >=15 targets are met for the measured short-context conditions, not at full context.
- Bench r3 TG12815.864460 and TG102415.687187; quality proxy PPL ratio0.998293, seven numeric tests and33 whole-graph sign cases passed. Proxy is not universal quality certification.
- Health endpoint returned ok; OpenAI chat completion returned exactly READY. PID4068 remains listening on127.0.0.1:9931. Source binary version label14c1d20c7 predates source commits bab57b4d3/e54f521cb; it was built with their tested working-tree changes. No code changed after the measured build.
- CUDA-relative utilization50% remains unproven because a matched CUDA measurement/denominator is unavailable. Production PP near78 must not be combined with the separate PP105.87 XMX/serialized result.

## 2026-09-19: memory-budget reporting decision

Report PP graph-buffer delta separately from total dedicated VRAM: matched PP256/512 graph allocations increased0 MiB, but total PP/TG VRAM delta was not measured under matched final conditions. FWHT adds4 KiB/workgroup on-chip shared memory, not VRAM. CPU embedding removes265.23 MiB of GPU model storage. Do not classify existing PP539.52 MiB compute workspace or enlarged KV as tuning-added memory. See MEMORY_BUDGET.md.

## 2026-09-19: measurement archive, deletion pending specific approval

User requested final archival and deletion of measurement originals for later Alchemist reverse engineering. Created archives/alchemist-bonsai2-measurements-20260919-100758.zip (136.05 MiB,355 source files); verified CRC and every file SHA256/size. Original measurements remain intact. Automatic approval review rejected deletion across pp-fix/logs/probe-four/versioned evidence because scope was insufficiently specific. Exact proposal287 files/493.82 MiB is recorded in the adjacent receipt. Preserve current live logs, launch metadata, model, code, fixtures and binaries. Archive helper is copy/verify only and performs no deletion. See ARCHIVE.md.

## 2026-09-19: bind address changed

- Restarted the server with only the listener address changed from 127.0.0.1 to 0.0.0.0. Port remains9931, requested context remains65535 (internal n_ctx_slot65536), Q8 K/V, CPU token embedding and the adopted Vulkan tuning are unchanged.
- Verification: listener LocalAddress=0.0.0.0, PID3168, health=ok. Full-use VRAM headroom at context65535 remains unmeasured; the prior 48384 gate must not be reused as proof for 65535.
- Versioned launcher now accepts -HostAddress and defaults its context to65535; the reproducibility command uses -HostAddress 0.0.0.0.

## 2026-09-19: 1000-character capability and template decision

- A 1,023-character Korean reasoning prompt produced 1,312 characters, and the correction turn produced 1,316; both fail the requested 900-1,100 range despite passing all four headings and the fixed final sentence.
- Reject the arithmetic result as fully correct: the model named C power116,640 won but omitted it from the total twice. Correct all-option first-month cost is9,092,480 won. It also reversed the capacity-versus-demand relation in one sentence.
- Use `reasoning_effort: none` for strict-length ordinary chat. The `low` trial consumed its1,400-token limit without yielding a captured final response. Use only `xhigh`, `medium`, or `low` when reasoning is enabled; `max` is rejected by the GGUF Jinja template with HTTP500.
- The template itself is loaded and applies system/user/assistant roles correctly. Client compatibility and token-budget choices require correction; see CAPABILITY_1000CHAR.md.

## 2026-09-19: server versus client token limit correction

- llama-server exposes `-n/--predict/--n-predict` as the server default generation limit; `-1` means infinity. The current command line explicitly uses `-n -1`, so the server is uncapped by default.
- OpenAI request fields `max_tokens`, `max_completion_tokens`, and llama.cpp `n_predict` are aliases for the per-request generation limit. A client should send `max_tokens:8192` to cap its own request at 8K.
- The earlier 8K server cap was a misapplication and was removed. The versioned and working launchers default `PredictTokens` to -1. No client configuration was changed because no single local client configuration was identified; the raw measurement request files are test fixtures only.

## 2026-09-19: n=5 Vulkan shader timing decision

- Adopt per-dispatch Vulkan timestamp queries for shader attribution. Each query brackets one Vulkan dispatch and is converted with the device timestamp period; no per-dispatch CPU wait is inserted.
- Re-ran the deterministic 512-token prompt with n_predict=5; validation returned tokens_evaluated=512 and tokens_predicted=5 (content qzqzq).
- Request scope contains one 512-token prefill graph and five 1-token decode graphs. Top-10 totals are in archives/bonsai2-vulkan-shader-n5-timing-20260919.zip; archive SHA256 is 9BEAA98C23A966B86EB1776A1D441C5A1A066E743DF0D5C9962069323BFF884D and internal hashes verified.
- Restored baseline ggml-vulkan.dll and source, deleted temporary raw probe files after archive verification, and restarted production on 0.0.0.0:9931 with health=ok.

## 2026-09-19: Bonsai sampling defaults

- Set the Bonsai launcher defaults to temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, and repetition_penalty=1.0.
- llama-server exposes repetition_penalty through the --repeat-penalty option; the launcher uses that spelling.
- Restarted the live server with the updated flags: PID 22080, 0.0.0.0:9931, context 65535, health=ok. The launcher change remains intentionally uncommitted.
