#!/usr/bin/env python3
"""Dry-run by default. Apply this overlay only to the XMX module with Level4 v1.
Refuses unknown edits, symlinks and unsafe paths. No git reset / DLL replacement.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from datetime import datetime, timezone


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inside(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if not relative or rel.is_absolute() or '..' in rel.parts or '\\' in relative or ':' in relative:
        raise ValueError(f'Unsafe path: {relative}')
    result = root.joinpath(*rel.parts)
    for p in (result, *result.parents):
        if p == root.parent:
            break
        if p.is_symlink():
            raise ValueError(f'Symlink not accepted: {p}')
    if root.resolve() not in result.resolve().parents:
        raise ValueError(f'Path escapes module: {relative}')
    return result


def plan(root: Path, overlay: Path, manifest: dict) -> list[tuple[str, bytes, bytes | None]]:
    if root.resolve() == overlay.resolve():
        raise ValueError('Extract overlay separately; do not apply it in-place to itself')
    for anchor in ('include/maple_moe.hpp', 'src/maple_moe.cpp', 'src/maple_level4.cpp'):
        if not inside(root, anchor).is_file():
            raise ValueError(f'Not a Level4-enabled XMX module: missing {anchor}')
    result, seen = [], set()
    for entry in manifest['files']:
        name = entry['path']
        if name in seen:
            raise ValueError(f'Duplicate manifest path: {name}')
        seen.add(name)
        src, dst = inside(overlay, name), inside(root, name)
        data = src.read_bytes()
        if digest(data) != entry['new_sha256']:
            raise ValueError(f'Overlay corrupt: {name}')
        old = dst.read_bytes() if dst.exists() else None
        if old == data:
            continue
        old_hash = digest(old) if old is not None else None
        if old_hash != entry.get('old_sha256'):
            # One controlled additive CMake modification; preserve user CMake edits.
            if name == 'cmake/level4.cmake' and old is not None:
                marker = 'include("${CMAKE_CURRENT_LIST_DIR}/level4_zerocopy.cmake")'
                text = old.decode('utf-8')
                if marker in text:
                    continue
                data = old + ('\n# Optional Level4 zero-copy transport\n' + marker + '\n').encode('utf-8')
            else:
                raise ValueError(f'Unknown existing edit, refusing overwrite: {name}')
        result.append((name, data, old))
    return result


def apply(root: Path, overlay: Path, manifest: dict, write: bool = False) -> dict:
    updates = plan(root, overlay, manifest)
    summary = {'mode': 'apply' if write else 'dry-run', 'changed_files': [x[0] for x in updates], 'count': len(updates)}
    if not write or not updates:
        return summary
    backup = root / 'build' / ('level4-zero-copy-backup-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f'))
    # No preexisting/symlinked build directory may redirect writes outside module.
    inside(root, backup.relative_to(root).as_posix())
    backup.mkdir(parents=True, exist_ok=False)
    for name, _, old in updates:
        if old is not None:
            dst = backup / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(old)
    written = []
    try:
        for name, data, old in updates:
            dst = inside(root, name)
            now = dst.read_bytes() if dst.exists() else None
            if now != old:
                raise RuntimeError(f'Concurrent change: {name}')
            dst.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=dst.parent, prefix='.level4-zc-', delete=False) as tmp:
                temp = Path(tmp.name)
                tmp.write(data)
            try:
                os.replace(temp, dst)
            finally:
                temp.unlink(missing_ok=True)
            written.append((dst, old, data))
        (backup / 'receipt.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    except Exception:
        for dst, old, new in reversed(written):
            # Do not overwrite a concurrent third-party edit during rollback.
            if dst.exists() and dst.read_bytes() == new:
                if old is None:
                    dst.unlink()
                else:
                    dst.write_bytes(old)
        raise
    summary['backup'] = str(backup)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--overlay', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        root = args.root.absolute()
        overlay = args.overlay.absolute()
        manifest = json.loads((overlay / 'ZERO_COPY_FILES.json').read_text(encoding='utf-8'))
        result = apply(root, overlay, manifest, args.apply)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f'LEVEL4_ZERO_COPY_APPLY FAIL: {exc}')
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
