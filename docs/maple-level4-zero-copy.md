# Maple Level4 Vulkan zero-copy runtime

The Level4 graph nodes use a versioned descriptor. The Vulkan backend claims them before the scheduler can copy their boundaries to CPU memory. A CPU callback is a fail-closed diagnostic, not a staging fallback.

The private Vulkan bridge owns exportable dedicated allocations, duplicated Win32 handles, buffer lifetime leases, queue serialization and release/acquire barriers. The SYCL executor imports handles into an explicit single-device Level Zero context. A weak registry shares imports across graph-owned states; strong leases keep the Vulkan buffer alive until the imported mapping is freed. Allocation identity and generation are checked before reuse.

Hidden/Q/K/V occupy separate contiguous planes in the custom output tensor. All graph views use the matching offsets and strides. The module runs with `BoundaryPolicy::direct_required`; no residual/attention upload or hidden/Q/K/V download occurs in the executor. A small status readback remains. This is a host-fenced transport, not an asynchronous external semaphore implementation.

The backend submits Vulkan producers, validates and prepares the executor, releases buffer ownership, runs the Level4 island, and acquires ownership before consumers. Failure after release poisons the device and returns a graph error. It never retries the partially executed node through the CPU callback. Empty terminal nodes are skipped.

## Windows device identity

Normal matching requires Intel vendor/device, valid nonzero LUID and equal single-bit node masks. This A750 driver reports Vulkan mask 1 and Level Zero mask 4, although both UUIDs and LUIDs match. The mismatch is accepted only after all of these additional checks:

- Both nonzero UUIDs match exactly.
- The Level Zero device is a root device, not a subdevice.
- Vulkan reports mask 1 and Level Zero reports a single-bit mask.
- DXGI opens the exact LUID and confirms the same vendor/device.
- D3D12 confirms that adapter has exactly one node.

Linked adapters, UUID mismatch, LUID mismatch and unavailable topology proof remain errors.

## Build and run

Use the oneAPI compiler environment on Windows. Configure the existing Vulkan build with `LLAMA_MAPLE_LEVEL4=ON` and `MAPLE_LEVEL4_ZERO_COPY=ON`. Level Zero headers and `ze_loader.lib` must be available through `MAPLE_LEVEL_ZERO_INCLUDE_DIR` and `MAPLE_LEVEL_ZERO_LIBRARY`. Build `llama-cli`, `llama-completion` and `maple-level4-zerocopy-probe`.

Runtime flags remain `LLAMA_MAPLE_LEVEL4=0|1` and `LLAMA_MAPLE_TOKEN_TILE=1|4|8`. Level4=0 selects the existing Vulkan graph. The tile flag affects the Level4 executor only.

`scripts/bench-maple-zerocopy.ps1` runs upstream non-custom default plus the six custom combinations, sequentially. It uses Vulkan1, FA on, batch/ubatch 512, a local corpus file, 4096 output tokens and identical q8_0 KV settings. It records executable and DLL hashes, commands, timestamps, logs and exit codes. It uses completion with `--no-conversation` to avoid a chat template around the corpus.

## Validation on Arc A750

- Portable contracts: 39 checks passed.
- Native API doubles: 834 checks passed.
- Real Vulkan -> Level Zero -> Vulkan probe with validation enabled: passed. Hidden/Q/K/V were bitwise equal to the packed reference, with zero measured boundary pack bytes.
- Actual model execution reached the Vulkan executor with `CPU_callback=0` and completed Layer 23 with `host_staged=0 boundary_copy_bytes=0` for tile 4 and tile 8, including decode.
- Level4=0 retained the ordinary Vulkan path and produced output.

These short checks establish transport and wiring, not full-model accuracy or performance improvement. Per-island weight uploads, workspace allocation, CPU fences and status readback remain. The 40k/4k seven-case benchmark is separate from these checks.

The module's original `ZERO_COPY_FILES.json` and overlay applicator describe the downloaded v1 archive. This repository also contains v0.5 preservation changes and runtime integration; use the repository patch/commit for this version rather than reapplying that original overlay manifest.
