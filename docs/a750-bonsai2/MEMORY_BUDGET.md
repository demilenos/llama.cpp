# PP/TG tuning: incremental memory budget

2026-09-19. Existing logs and source audit only; no server restart or additional GPU workload. Baseline is source07c232edf; distinguish graph allocator buffers, dedicated VRAM, and on-chip shared local memory.

| Item | Before | After | Increment and qualification |
|---|---:|---:|---|
| PP256 Vulkan graph compute buffer |269.6348 MiB|269.6348 MiB|0 MiB measured in allocator logs|
| PP512 Vulkan graph compute buffer |539.5196 MiB|539.5196 MiB|0 MiB measured in allocator logs|
| PP256 host graph compute buffer |5.1349 MiB|5.1349 MiB|0 MiB|
| PP512 host graph compute buffer |10.5196 MiB|10.5196 MiB|0 MiB|
| TG FWHT shared local memory |0 bytes/WG|4096 bytes/WG|+4 KiB/WG on-chip, not VRAM|
| Final TG float MMV shared local memory |baseline route not matched|1024 bytes/WG for N1|Current absolute size, not an established increment|
| PP XMX selected tile shared local memory |baseline route not matched|32768 bytes/WG|Current experimental absolute size, not dedicated VRAM or an established increment|
| Hybrid/signed FWHT versus shared FWHT |4096 bytes/WG|4096 bytes/WG|0 additional shared bytes|
| CPU embedding placement: GPU model buffer |5660.56 MiB|5395.33 MiB|-265.23 MiB GPU model storage; CPU mapped storage265.23 MiB|

The matched PP allocator evidence is the separate XMX/serialized experiment, not the adopted normal-scheduling PP profile. Its buffers were already needed before optimization;539.52 MiB is not an added tuning cost. Full GPU residency, KV and recurrent state are unchanged in that matched comparison.

Source diff introduces no new model-sized or persistent Vulkan tensor/scratch-buffer allocation sites for these optimizations. Signed FWHT reads the existing signs tensor, adds one descriptor and expands push constants16 to24 bytes; it does not create a signs/weight copy. Skipping an intermediate write does not prove graph-reservation savings. Existing scratch paths, compiled pipelines, descriptor pools, driver resources and register spills can still change.

Consequently, exact baseline-to-final dedicated-VRAM increases for PP and TG separately are NOT measured. Reporting zero total additional VRAM would be unsupported. A matched peak comparison requires identical context, KV, embedding placement, batch/ubatch and request. TG logs at batch1/4/32/128 cannot be subtracted to infer optimization overhead.

Current combined deployed profile: context48384, Q8 K/V, CPU embedding, peak adapter dedicated usage7580.54 MiB, reserve517.46 MiB of8098 MiB usable budget. This is a whole-adapter peak, not incremental PP/TG memory. CPU mapped storage is an address-space/buffer size, not a measured physical RAM-residency increase.

Evidence:
- pp-fix/final-baseline-bench.log:2381-2382,2619-2620.
- pp-fix/final-candidate-bench.log:2337-2338,2604-2605.
- pp-fix/tg128-gpu.log:2142; pp-fix/tg128-cpu.log:2143-2144.
- ggml/src/ggml-vulkan/vulkan-shaders/fwht.comp; mul_mat_vec.comp and its shared-reduction include; selected XMX tile source.
- Versioned evidence/server-memory-20260919-095642.json.
