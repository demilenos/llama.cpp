#!/usr/bin/env python3
"""Audit a *real probe* result folder; never manufactures GPU performance data."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def analyze(root: Path) -> dict:
    scope = (root / 'SCOPE.txt').read_text(encoding='utf-8')
    if 'Real Vulkan->LevelZero/SYCL Level4->Vulkan' not in scope:
        raise ValueError('Missing completed native-probe scope')
    samples, numerical = read_csv(root / 'samples.csv'), read_csv(root / 'numerical.csv')
    if not samples or not numerical:
        raise ValueError('Empty result CSV')
    if [int(r['repeat']) for r in samples] != list(range(len(samples))):
        raise ValueError('Repeat IDs are incomplete/duplicated')
    fields = ('inbound_host_us', 'sycl_host_us', 'acquire_submit_us', 'total_host_us', 'gpu_sycl_span_us', 'execute_wall_us')
    for r in samples:
        for field in fields:
            x = float(r[field])
            if not math.isfinite(x) or x < 0:
                raise ValueError(f'Invalid {field}: {x}')
        parts = sum(float(r[k]) for k in fields[:3])
        if abs(parts - float(r['total_host_us'])) > max(0.05, parts * 1e-6):
            raise ValueError('Host timing boundaries do not sum')
        if float(r['execute_wall_us']) + .05 < float(r['total_host_us']):
            raise ValueError('Outer execute wall cannot be smaller than its instrumented span')
        if int(r['boundary_d2d_bytes']) != 0:
            raise ValueError('A GPU boundary pack/copy was submitted')
    for field in ('imports', 'exports'):
        values = [int(r[field]) for r in samples]
        if len(set(values)) != 1 or values[0] <= 0:
            raise ValueError(f'{field} changed in the measured hot path or was never executed')
    expected = {'hidden'} if 'entry=terminal' in scope else {'hidden', 'q', 'k', 'v'}
    seen: dict[int, set[str]] = {}
    for r in numerical:
        i, name = int(r['repeat']), r['output']
        group = seen.setdefault(i, set())
        if name in group:
            raise ValueError('Duplicated numerical output')
        group.add(name)
        for key in ('max_abs', 'nmse', 'max_abs_over_rms'):
            if not math.isfinite(float(r[key])) or float(r[key]) < 0:
                raise ValueError('Invalid numerical error')
        if int(r['finite']) != 1 or float(r['nmse']) >= 1e-6 or float(r['max_abs_over_rms']) >= .01:
            raise ValueError('Native output comparison failed')
        if int(r['bitwise_equal']) not in (0, 1):
            raise ValueError('Invalid equality flag')
    # Numerical repeat 0 is the cold/warmup run; samples exclude it.
    if set(seen) != set(range(len(samples) + 1)) or any(v != expected for v in seen.values()):
        raise ValueError('Missing warmup/repeat numerical checks')
    return {
        'audit': 'PASS', 'sample_count': len(samples),
        'timing_medians_us': {k: median(float(r[k]) for r in samples) for k in fields},
        'hot_imports': int(samples[0]['imports']), 'hot_exports': int(samples[0]['exports']),
        'boundary_copy_bytes': 0,
        'bitwise_equal_outputs': sum(int(r['bitwise_equal']) for r in numerical),
        'checked_outputs': len(numerical),
        'scope': scope,
        'limitations': [
            'Based on supplied probe outputs; analyzer does not execute a GPU.',
            'Synthetic weights and uniform per-token inputs; no model-quality or representative-performance claim.',
            'Host-fenced import transport; zero-copy is not zero-wait.',
            'GPU/SYCL timing and host timing are different clocks; do not add them.',
            'Diagnostic uploads/readbacks are outside island timing; counters describe submitted software copies, not PCIe bus telemetry.',
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.results)
    except (OSError, ValueError, KeyError) as exc:
        print(f'ZERO_COPY_AUDIT FAIL: {exc}')
        return 1
    (args.results / 'zero-copy-audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
