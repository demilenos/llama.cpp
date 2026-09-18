#!/usr/bin/env python3
"""Add only Level4 files. Never replace merged MoE/DPAS sources or reset Git."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ANCHOR = '\n# Level4 attention-to-attention island (additive)\ninclude(cmake/level4.cmake)\n'
def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True, help='Existing XMX MODULE root, containing include/ and src/')
    ap.add_argument('--apply', action='store_true', help='Without this option: dry-run only')
    ap.add_argument('--no-cmake', action='store_true', help='Copy new files only; manually link maple_level4.cpp in parent build')
    args=ap.parse_args()
    payload=Path(__file__).resolve().parents[1]
    root=args.root.resolve()
    if root == payload.resolve():
        raise RuntimeError('Target must be the existing module, not the upgrade payload directory.')
    meta=json.loads((payload/'LEVEL4_FILES.json').read_text(encoding='utf-8'))
    for rel, symbols in {
        'include/maple_moe.hpp':['gate_tokens_per_tile', 'MoeWorkspace', 'enqueue_moe'],
        'include/maple_w2a8.hpp':['enqueue_a8_quant', 'A8Mode'],
        'src/maple_moe.cpp':['enqueue_moe'],
    }.items():
        p=root/rel
        if not p.is_file() or not all(s in p.read_text(encoding='utf-8-sig') for s in symbols):
            raise RuntimeError(f'Missing architecture-v1 interface in {p}; no files changed.')
    changes=[]
    for rel, expected in meta['files'].items():
        rp=Path(rel)
        if rp.is_absolute() or '..' in rp.parts: raise RuntimeError('Unsafe payload path')
        src=payload/rp; dst=root/rp
        data=src.read_bytes()
        if sha(data)!=expected: raise RuntimeError(f'Payload SHA mismatch: {rel}')
        if dst.exists():
            if dst.is_symlink() or not dst.is_file() or dst.read_bytes()!=data:
                raise RuntimeError(f'Existing different Level4 file will NOT be overwritten: {dst}')
        else:
            if not dst.resolve().is_relative_to(root): raise RuntimeError('Path escapes module')
            changes.append((rel,dst,data))
    cmake=root/'CMakeLists.txt'; old_cmake=None; new_cmake=None
    if not args.no_cmake:
        if not cmake.is_file(): raise RuntimeError('No module CMakeLists.txt; use --no-cmake and direct build script.')
        old_cmake=cmake.read_bytes(); text=old_cmake.decode('utf-8-sig')
        if 'include(cmake/level4.cmake)' not in text:
            if 'add_library(maple-w2-xmx' not in text:
                raise RuntimeError('Cannot identify module CMake target. Use --no-cmake; parent ggml linkage is explicit.')
            new_cmake=old_cmake+ANCHOR.encode('utf-8')
    for rel,_,data in changes: print(f'ADD {rel} ({len(data)} bytes)')
    if new_cmake is not None: print('APPEND Level4 CMake include; preserve existing CMake content')
    print('UNCHANGED: src/maple_moe.cpp, all existing DPAS/grouping sources, tools/grouping_compare.cpp')
    if not args.apply: print('DRY RUN. Repeat with --apply to write.'); return 0
    # Stage all bytes on the SAME filesystem. Refuse divergent files before any write.
    made=[]; cmake_written=False
    stage=Path(tempfile.mkdtemp(prefix='.level4-stage-',dir=root))
    try:
        for rel,dst,data in changes:
            p=stage/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        if new_cmake is not None:(stage/'new-cmake.txt').write_bytes(new_cmake)
        for rel,dst,_ in changes:
            dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists(): raise RuntimeError(f'Concurrent write detected: {dst}')
            os.replace(stage/rel,dst);made.append(dst)
        if new_cmake is not None:
            if cmake.read_bytes()!=old_cmake:raise RuntimeError('Concurrent CMake edit detected')
            os.replace(stage/'new-cmake.txt',cmake);cmake_written=True
    except Exception:
        for p in reversed(made): p.unlink(missing_ok=True)
        if cmake_written:cmake.write_bytes(old_cmake)
        raise
    finally:
        shutil.rmtree(stage,ignore_errors=True)
    print('Level4 files installed. Compile/test explicitly; no DLL, launcher, Git history or model changed.')
    return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc: raise SystemExit(f'Level4 upgrade refused: {exc}')
