# Maple v0.5 + Level 4 runtime

This fork can execute the supplied Level 4 module from `llama-cli`,
`llama-bench`, and `llama-server`. Enable the build option and opt in at runtime.
The default graph is unchanged when the runtime switch is absent.

## Build (Windows, Intel oneAPI + Vulkan SDK)

Run in an initialized oneAPI/MSVC command prompt:

```bat
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 vs2022
cmake -S . -B build-maple -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DLLAMA_MAPLE_LEVEL4=ON -DLLAMA_OPENSSL=OFF
cmake --build build-maple --target llama-cli llama-server llama-bench --parallel 6
```

Set `VULKAN_SDK` and `SPIRV-Headers_DIR` for your Vulkan SDK if CMake cannot
locate them. Run the binaries in the same initialized environment so Intel
runtime DLLs are available.

```bat
set LLAMA_MAPLE_LEVEL4=1
set LLAMA_MAPLE_TOKEN_TILE=4
build-maple\bin\llama-cli.exe -m C:\AI\models\maple-preview-TQ2_0-head-F16.gguf -ngl 99 -c 512 -b 16 -ub 16 --no-warmup -p "The capital of France is" -n 2
```

`LLAMA_MAPLE_TOKEN_TILE=1|4|8` selects grouped RC1 or native s2/s8 DPAS RC4/RC8.
It is separate from the older token-reuse tile option. Default is RC4.
Use RC1 as the same-quantization comparison, and unset `LLAMA_MAPLE_LEVEL4`
for the original ggml graph. Changing environment variables in a running server
is unsupported; restart the process for each configuration.

## Boundaries and limitations

- Bootstrap: attention RMSNorm and raw Q/K/V.
- Advance: pre-O attention, O, sequential residuals, FFN RMSNorm, FP32 router,
  W2A8 G32/H32 MoE, next attention RMSNorm, next raw Q/K/V.
- Terminal: O through final residual. Final output row selection is applied to
  both residual and pre-O attention before this island.
- Q/K norm, RoPE, attention/KV, output norm/head, and sampling remain in ggml.
- The scheduler places the island on the CPU backend for host transfers, but
  its callback runs the real Intel Arc SYCL/XMX kernels. Completion logs include
  `host_staged=1`. This is **not zero-copy Vulkan/SYCL interop**.
- Packed weights are cached in host RAM for a graph's lifetime and uploaded per
  island. This avoids duplicating the whole model in VRAM, but adds substantial
  host traffic and allocation cost. This path is experimental, not a speed claim.
- O/QKV use A16 activations; MoE uses A8 activations. These change arithmetic
  relative to the ordinary Vulkan graph. RC1/RC4 parity does not establish model
  quality equivalence to the original model.
- Requires an A750/A770 Level Zero device, contiguous TQ2_0 projection and expert
  weights, clamp=7, at most 2048 tokens per ubatch, supported module dimensions,
  and F16/F32 norm/router tensors. Unsupported LoRA, control vectors, fused QKV,
  projection bias/scale or QKV clamps are rejected before graph execution.
- TQ2 payloads are validated before first repacking. Device status and finite
  outputs are checked on every invocation. Failures abort rather than silently
  reusing partially written output or claiming fallback success.

## Source provenance

Base fork commit: `add3706378d9863c88f5f3490b04ee143380b190`.
The existing v0.5 sources were compared with the user-supplied Drive archive;
the prior fork's v0.4 grouping/architecture extensions were preserved.

- `maple-w2-xmx-v0.5-source.zip` SHA256:
  `5e04931b5cedc85c6ceabb3732d60ef766119865f7bcf3f8cffd72dbc3d4dc01`
- `maple-w2-xmx-v0.4-level4-upgrade-v1.patch` SHA256:
  `40b952b107ba69e5974d25f6e88195fcf3c1c715390ccfacd1f9603c0ff7d199`

The Level 4 package's original validation notes describe its original standalone
state. They are not evidence that this integration passed GPU or model tests.
See the integration validation record for measurements from this checkout.

## Integration checks

Run from the initialized compiler environment:

```bat
python scripts\verify-maple-level4.py --server build-maple\bin\llama-server.exe --model C:\AI\models\maple-preview-TQ2_0-head-F16.gguf --out validation-maple
```

The script runs four isolated servers, preserves responses and dispatch logs,
and compares RC4/RC8 against RC1 top-20 log probabilities for two generated tokens.
This is a short model smoke check, not full-logit, long-context, or quality validation.
The recorded run is in [the validation report](validation/maple-level4-20260918/README.md).
