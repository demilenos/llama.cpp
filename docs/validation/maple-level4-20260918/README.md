# Maple integration validation - 2026-09-18

Windows, Intel Arc A750, oneAPI 2026.1.1. Source base add370637.

- Built llama-cli, llama-server and llama-bench with LLAMA_MAPLE_LEVEL4=ON and Vulkan.
- CPU/host suite: 11/11 passed; see cpu-tests.txt.
- Actual GPU RC4/RC8 integer DPAS, grouping and tile-plan probes: passed; see gpu-probe.txt.
- Actual CLI completed bootstrap, 23 advances and terminal through all 24 model layers.
- Final server checks: baseline/RC1/RC4/RC8 each returned ` Paris.` for the same prompt.
- RC4/RC8 matched RC1 output tokens, top-20 token IDs and log probabilities (maximum difference 0).
- The original Vulkan graph had different top-20 distributions: maximum common-entry logprob difference 0.944741249 versus RC1. This is not full-model numerical parity.

Use scripts/verify-maple-level4.py to reproduce the server checks. See results.json for final-run timings, dispatch coverage and binary hashes. The short runs include cold graph/weight preparation and host staging. Level 4 was much slower than ordinary Vulkan (about 15 seconds per generated token here). No performance improvement is claimed. Zero-copy/resident-weight optimization and long-context/model-quality validation remain future work.
