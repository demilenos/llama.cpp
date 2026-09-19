# A750 Vulkan optimized build snapshot

This directory is a runtime artifact snapshot for the Bonsai2 A750 Vulkan server.

Included:
- `bin/`: runtime executables and DLLs from the validated build
- `ggml/src/ggml-vulkan/vulkan-shaders.spv`: compiled Vulkan shader bundle
- `BUILD-MANIFEST.sha256`: integrity hashes for the included files

Excluded from the remote snapshot: CMake/Ninja intermediates, object files, generated C++ copies, tests, and the UI `node_modules` tree. The complete local build tree is about 1.26 GiB and is not a reproducible source checkpoint.

Provenance:
- Remote baseline: `origin/codex/vulkan-a750-ptq1@ee092d834`
- Archived source HEAD: `09eea3e672f4a8427908e7fcff1e188e6ab3a1dc`
- Binary label: `14c1d20c737cd7fa3ab1ad6097ecd5a32af6d8f6`
- Later worktree changes included commits `bab57b4d3` and `e54f521cb`

This is an artifact snapshot, not a source checkout. Rebuild from the source checkout and its recorded commits when source-level reproducibility is required.
