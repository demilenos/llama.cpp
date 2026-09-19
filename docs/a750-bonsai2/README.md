# Bonsai2 27B on Arc A750: current evidence

This report covers the adopted CPU-embedding deployment and its measured limits. The canonical decision log is the mirrored `DECISIONS.md` beside this file; the working copy is `C:/AI/bonsai2_27b/pp-fix/DECISIONS.md`.

The signed WG64/rows4 configuration meets the measured decode targets:

| Workload | Result |
| --- | ---: |
| TG128, r3 | 15.864460 +/- 0.007227 tok/s |
| TG1024, r3 | 15.687187 +/- 0.007388 tok/s |
| PP512, CPU embedding, r1 | 78.031034 tok/s |
| PP512, GPU embedding, r1 | 77.649063 tok/s |
| TG128 CPU embedding | 15.864460 +/- 0.007227 tok/s |
| TG128 GPU embedding | 15.845674 +/- 0.011085 tok/s |

The CPU embedding comparison shows no observed decode loss and is retained for VRAM optimization. Seven numeric tests and the 33-case whole-graph signed-Hadamard validation pass. The quality proxy uses Q8 KV, ubatch 1, four WikiText2 chunks: PPL 7.941728 versus baseline 7.955312, ratio 0.998293. This is a bounded proxy and does not certify broad model capability.

Server memory evidence is still context-sensitive. Context 48640 with a 2057-token prompt and 1024 generated tokens measured 15.282001 tok/s with 521.50 MiB reserve. Context 48896 measured 15.288034 tok/s with 512.74 MiB reserve, but a 4096-word probe left only 500.27 MiB, so full-use acceptance is rejected. Context 49152 left 503.97 MiB and is rejected. Context 41472 passed a near-full probe (41225 input + 128 generation): peak 7301.89 MiB, reserve 796.11 MiB, decode 11.31 tok/s. The current server is bound to 0.0.0.0:9931 (PID3168) with requested context65535 (internal slot65536), Q8_0 K/V, CPU token embedding and all transformer layers on GPU. Health check passed after restart. The earlier 48384 memory stress evidence remains the last measured VRAM gate; 65535 has not yet had a full-use VRAM stress probe. Long-context throughput is lower than the TG benchmarks above.

The versioned launcher is `scripts/a750-bonsai2/start-server.ps1`. Example invocation:

```powershell
.\scripts\a750-bonsai2\start-server.ps1 -WorkspaceRoot C:\AI\bonsai2_27b -HostAddress 0.0.0.0 -ContextSize 65535 -CpuEmbedding
```

The adopted environment is: `GGML_VK_DISABLE_MMVQ=1`, `GGML_VK_PTQ1_MMV_WG=64`, `GGML_VK_PTQ1_MMV_ROWS=4`, `GGML_VK_A750_FWHT_SHMEM=1`, `GGML_VK_A750_FWHT_WG=256`, `GGML_VK_A750_FWHT_HYBRID=1`, `GGML_VK_A750_FWHT_SIGNS=1`, `GGML_VK_A750_SUBMIT_DIVISOR=16`, and `GGML_VK_MAX_NODES_PER_SUBMIT=256`. Production diagnostics and conflicting experimental overrides are cleared: `GGML_VK_PTQ1_FORCE_XMX`, `GGML_VK_SERIALIZE_SUBMISSIONS`, `GGML_VK_PTQ1_TILE`, `GGML_VK_PERF_LOGGER`, and `GGML_VK_PTQ1_PROBE`, `GGML_VK_DISABLE_FUSION`, and `GGML_VK_SUBMIT_TRACE`.

The source inventory contains 12 commits after base `07c232edf`, through `e54f521cb`. Evidence files are in `evidence/`; JSON-origin files are stored with `.log` names where they contain prefixed stderr and are not strict JSON.

The CUDA-relative 50% hardware-utilization claim remains unproven. The repository has no matched local CUDA benchmark with the same model, precision, workload, and hardware-normalized denominator; A750 Vulkan throughput and external PrismML RTX/L4 tok/s references cannot establish that claim.

PP105.87 belongs to a separate XMX/serialized experiment which severely regresses decode; the deployment configuration uses normal scheduling and measured PP near 78 tok/s.


Final server decode: 15.277944 tok/s for 2057 input +1024 generation; 15.077958 for 4105+128; 10.868322 for 48137+128. The 15 tok/s result does not extend to a nearly full context. VRAM reserve uses the 8098 MiB usable driver/Vulkan budget; the minimum sampled reserve was 517.46 MiB, also observed after the chat smoke. Context48384 is the highest tested accepted size, not proof against unrelated concurrent GPU allocations.

The GPU critical-path hypothesis is not established: matvec workgroup tuning gave about7% while submission tuning gave about0.24%. Instrumented per-node timestamps are perturbative and do not distinguish arithmetic, bandwidth or host queue gaps. See DECISIONS.md for the retained reasoning.

The binary build label14c1d20c7 was generated before later source commits, while the tested binary already included their working-tree changes. Full raw memory-probe responses remain under workspace pp-fix; versioned evidence omits repeated prompt/output text and retains timings and memory samples.

The current bind address is intentionally 0.0.0.0 for LAN access. Keep the firewall exposure scoped to the intended network.
