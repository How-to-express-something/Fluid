"""Compare matched workloads; report medians across per-run statistics."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

MATCH = ['schema_version', 'mode', 'device_name', 'cuda_runtime_version', 'cuda_driver_version',
         'gl_renderer', 'gl_version', 'build_config', 'grid_width', 'grid_height', 'framebuffer_width',
         'framebuffer_height', 'visible', 'vsync_requested', 'color', 'seed', 'frames', 'warmup', 'repeats',
         'timer_pool_size', 'trajectory', 'dt', 'jacobi_iterations', 'diffuse_iterations', 'viscosity',
         'velocity_dissipation', 'density_dissipation', 'force_radius', 'force_strength', 'buoyancy', 'display_gain']

def load(prefix):
    meta = json.loads(Path(prefix + '_metadata.json').read_text(encoding='utf-8'))
    with Path(prefix + '_summary.csv').open(newline='') as file:
        rows = list(csv.DictReader(file))
    return meta, rows


def compare(baseline, candidate):
    before, a = load(baseline)
    after, b = load(candidate)
    mismatch = [key for key in MATCH if before.get(key) != after.get(key)]
    if mismatch:
        raise ValueError('Unmatched workload/environment: ' + ', '.join(mismatch))
    if len(before['results']) != before['repeats'] or len(after['results']) != after['repeats']:
        raise ValueError('Incomplete runs')
    for x, y in zip(before['results'], after['results']):
        for metric in ['density_mass', 'density_checksum', 'density_min', 'density_max']:
            if not math.isclose(x[metric], y[metric], rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError('Final density signature changed; investigate correctness before claiming speedup')
    print('Matched workload. Statistics below are medians across per-run mean/P95 values (not pooled percentiles).')
    print('Pressure backend: ' + before.get('pressure_backend', 'historical/unknown') + ' -> ' + after.get('pressure_backend', 'historical/unknown'))
    print(f'{"metric":24} {"base mean ms":>14} {"new mean ms":>14} {"mean speedup":>14} {"base P95 ms":>14} {"new P95 ms":>14}')
    for metric in ['step_cpu', 'step_gpu_interval', 'publish_cpu', 'draw_submit_cpu', 'draw_gpu', 'swap_cpu', 'recycle_wait_cpu', 'frame_cpu', 'completed_run_wall']:
        left = [r for r in a if r['metric'] == metric]
        right = [r for r in b if r['metric'] == metric]
        if not left and not right:
            continue
        if len(left) != before['repeats'] or len(right) != after['repeats']:
            raise ValueError('Missing timing rows: ' + metric)
        ma = statistics.median(float(r['mean']) for r in left)
        mb = statistics.median(float(r['mean']) for r in right)
        pa = statistics.median(float(r['p95']) for r in left)
        pb = statistics.median(float(r['p95']) for r in right)
        speedup = f'{ma / mb:.3f}x' if mb > 0 else 'n/a'
        print(f'{metric:24} {ma:14.6f} {mb:14.6f} {speedup:>14} {pa:14.6f} {pb:14.6f}')
    print('Density signatures are sanity checks, not proof of full numerical equivalence; run correctness tests too.')
    print('CPU/GPU stage times overlap. Do not add their speedups or interpret work-item throughput as display FPS.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True, help='Report prefix, without _metadata.json')
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    try:
        compare(args.baseline, args.candidate)
    except (OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
