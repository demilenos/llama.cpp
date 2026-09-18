# Ternary optimized v1 (Alchemist)

## Baseline

Immutable baseline ref used by this line:

- branch: `baseline/ptq1-xmx-n1-8131120`
- commit: `81311206cb89784183d5975dcb2ab5243b59ad1b`
- source line: `llama_maple_optimized/v5`
- host PTQ1 repack regression: `Shape{256, 8, 1}`, element-wise K0..255
- shared PTQ1 trit coordinate/value oracle: 128 values per block

The development branch is `llama_ternary_optimized/v1`.

## v1 execution shape

The baseline dense bridge launches one PTQ1 matmul work-item per token/output M8 tile.
For ncols 2/4 this repeats the same raw PTQ1 K32 base-3 decode for each token.

v1 adds a dense single-expert specialization:

```
M8 output tile
  for K256
    load K128 weight scales
    for K32
      raw PTQ1 -> signed INT2 once
        -> DPAS A8[token 0] -> acc0
        -> DPAS A8[token 1] -> acc1
        -> DPAS A8[token 2] -> acc2   # T4
        -> DPAS A8[token 3] -> acc3   # T4
```

Activation quantization remains staged G32 and happens once per token.

Default routing is intentionally narrow:

- ncols=1: baseline kernel
- ncols=2: T2 shared-decode kernel
- ncols=4: T4 shared-decode kernel
- all other ncols: baseline kernel until correctness and register-pressure data are collected

Runtime controls:

- `GGML_VULKAN_PTQ1_XMX_TILE=1|2|4`
- `GGML_VULKAN_PTQ1_XMX_LOCAL_SIZE=4|8|16|32`

Without overrides, tiled kernels start at local size 8; baseline stays at 4.
Local size is an empirical sweep parameter, not a correctness requirement.

## Regression gates

For ncols 1/2/4 record:

1. numerical correctness against the baseline/golden output
2. final ISA instruction count
3. GRF usage and any scratch/spill traffic
4. kernel latency
5. effective weight bandwidth

The optimization is accepted only when ncols 2/4 reduce the instruction-count slope
without introducing material spill traffic or numerical drift.

## Build

```powershell
.\maple-w2-xmx-v0.4\scripts\build_ternary_alchemist_v1.ps1 -Clean
```

Smoke with a PTQ1 model:

```powershell
.\maple-w2-xmx-v0.4\scripts\build_ternary_alchemist_v1.ps1 `
  -Model C:\path\model.ptq1.gguf `
  -LocalSize 8
```

Force baseline or a tile for A/B:

```powershell
$env:GGML_VULKAN_PTQ1_XMX_TILE = "1"  # baseline
$env:GGML_VULKAN_PTQ1_XMX_TILE = "2"
$env:GGML_VULKAN_PTQ1_XMX_TILE = "4"
```

For IGC JIT dumps, `IGC_ShaderDumpEnable=1` can be used with a custom dump directory.
Treat IGC debug flags as diagnostic interfaces rather than stable runtime API.
