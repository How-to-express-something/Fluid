"""Archive two committed builds and run an interleaved, matched pressure-graph experiment.

Uses only the Python standard library. Raw CSV files and binaries stay in --raw-dir.
No build, clock locking, power setting changes or profiler instrumentation is performed.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone

from compare_benchmarks import MATCH


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args, cwd=None):
    result = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            check=True, text=True, encoding='utf-8', errors='replace')
    return result.stdout.strip()


def source_identity(root):
    paths = set()
    for base, patterns in [('source', ('*.h', '*.cu')), ('graphics', ('*.h', '*.cpp'))]:
        for pattern in patterns:
            paths.update((root / base).rglob(pattern))
    paths.update((root / 'graphics/tests').glob('*.cu'))
    paths.update([root / 'CMakeLists.txt', root / 'graphics/CMakeLists.txt'])
    rows = [(p.relative_to(root).as_posix(), sha(p)) for p in paths]
    rows.sort()
    text = ''.join(f'{name}:{digest}\n' for name, digest in rows)
    return hashlib.sha256(text.encode()).hexdigest(), dict(rows)


def gpu_state():
    try:
        return command(['nvidia-smi', '--query-gpu=name,driver_version,pstate,temperature.gpu,'
                        'power.draw,clocks.current.graphics,clocks.current.memory,utilization.gpu',
                        '--format=csv'])
    except (OSError, subprocess.CalledProcessError) as error:
        return 'Unavailable: ' + str(error)


def quantile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = int(index)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for side in ('a', 'b'):
        parser.add_argument(f'--{side}-root', required=True, type=Path)
        parser.add_argument(f'--{side}-exe', required=True, type=Path)
    parser.add_argument('--raw-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--target-seconds', type=float, default=3)
    parser.add_argument('--blocks', type=int, default=2, help='Alternating ABBA / BAAB blocks')
    parser.add_argument('--grids', type=int, nargs='+', default=[256, 512, 1024])
    args = parser.parse_args()
    if not 1 <= args.blocks <= 20 or not 0 < args.target_seconds <= 60:
        parser.error('Invalid block count or target duration')
    if any(g < 2 or g > 2048 for g in args.grids):
        parser.error('Grid must be in [2, 2048]')
    raw, output = args.raw_dir.resolve(), args.output_dir.resolve()
    if raw.exists() or output.exists():
        parser.error('Choose new raw and output directories; existing evidence is never overwritten')
    raw.mkdir(parents=True)
    output.mkdir(parents=True)
    manifest = {'schema_version': 1, 'started_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'running', 'platform': platform.platform(), 'cpu': platform.processor(),
                'raw_directory': str(raw), 'target_seconds': args.target_seconds, 'blocks': args.blocks,
                'warmup_steps': 240, 'pilot_frames': 600, 'frames_cap': 60000,
                'seed': 123, 'grids': args.grids, 'variants': {}, 'trials': [], 'pilot_trials': [],
                'power_policy_changed': False, 'clocks_locked': False}
    try:
        manifest['power_scheme'] = command(['powercfg', '/getactivescheme'])
        manifest['cpu'] = command(['powershell', '-NoProfile', '-Command',
                                  '(Get-CimInstance Win32_Processor).Name'])
        manifest['battery'] = command(['powershell', '-NoProfile', '-Command',
            'Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining | ConvertTo-Json'])
    except (OSError, subprocess.CalledProcessError):
        pass
    for side, label in [('a', 'direct'), ('b', 'graph')]:
        root = getattr(args, side + '_root').resolve()
        exe = getattr(args, side + '_exe').resolve()
        identity, files = source_identity(root)
        command(['git', 'diff', '--exit-code', 'HEAD', '--', *files], cwd=root)
        commit = command(['git', 'rev-parse', 'HEAD'], cwd=root)
        dest = raw / ('Fluid_' + side.upper() + '.exe')
        shutil.copy2(exe, dest)
        archive = raw / ('source_' + side.upper() + '.zip')
        command(['git', 'archive', '--format=zip', '--output=' + str(archive), commit], cwd=root)
        manifest['variants'][side.upper()] = {'backend': label, 'commit': commit,
            'root': str(root), 'executable': str(dest), 'executable_sha256': sha(dest),
            'source_snapshot_id': identity, 'source_files': files, 'source_archive_sha256': sha(archive)}
        cache = exe.parent.parent / 'CMakeCache.txt'
        keys = ('CMAKE_CUDA_ARCHITECTURES:', 'CMAKE_CUDA_COMPILER:', 'CMAKE_CXX_COMPILER:',
                'CMAKE_GENERATOR:', 'CMAKE_GENERATOR_TOOLSET:', 'FLUID_BENCHMARK_VARIANT:')
        if cache.is_file():
            manifest['variants'][side.upper()]['build_cache'] = [line for line in
                cache.read_text(encoding='utf-8').splitlines() if line.startswith(keys)]
    write_json(output / 'manifest.json', manifest)

    def run(side, mode, grid, frames, name, pilot=False):
        version = manifest['variants'][side]
        prefix = raw / name
        cmd = [version['executable'], '--benchmark', mode, '--grid', str(grid), str(grid),
               '--window', '1024', '720', '--frames', str(frames), '--warmup', '240',
               '--repeats', '1', '--inflight', '4', '--jacobi', '60', '--diffuse', '20',
               '--viscosity', '0', '--seed', '123', '--label', side + '_' + version['backend'],
               '--output', str(prefix)]
        print(f'{name}: {side} {mode} {grid}x{grid}, {frames} steps', flush=True)
        before = gpu_state()
        started = datetime.now(timezone.utc).isoformat()
        begin = time.monotonic()
        with (raw / (name + '.log')).open('wb') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        meta = json.loads(Path(str(prefix) + '_metadata.json').read_text(encoding='utf-8'))
        if meta['pressure_backend'] != version['backend'] or meta['source_snapshot_id'] != version['source_snapshot_id']:
            raise ValueError('Build identity mismatch: ' + name)
        trial = {'name': name, 'variant': side, 'mode': mode, 'grid': grid, 'frames': frames,
                 'started_utc': started, 'process_wall_seconds': time.monotonic() - begin,
                 'gpu_before': before, 'gpu_after': gpu_state(), 'command': cmd,
                 'report_sha256': {ext: sha(str(prefix) + ext) for ext in
                                  ('_metadata.json', '_samples.csv', '_summary.csv')}}
        manifest['pilot_trials' if pilot else 'trials'].append(trial)
        write_json(output / 'manifest.json', manifest)
        return meta

    try:
        for grid in args.grids:
            for mode in ('solver', 'pipeline'):
                pilots = [run(side, mode, grid, 600, f'pilot_{mode}_{grid}_{side}', True) for side in ('A', 'B')]
                fastest = min(m['results'][0]['wall_ms'] / m['frames'] for m in pilots)
                frames = min(60000, max(1200, math.ceil(args.target_seconds * 1000 / fastest)))
                for block in range(args.blocks):
                    order = 'ABBA' if block % 2 == 0 else 'BAAB'
                    for index, side in enumerate(order):
                        run(side, mode, grid, frames, f'{mode}_{grid}_block{block}_{index}_{side}')
        manifest['status'] = 'complete'
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output / 'manifest.json', manifest)
        summarize(raw, output, manifest)
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['error'] = str(error)
        write_json(output / 'manifest.json', manifest)
        raise


def summarize(raw, output, manifest):
    """Each trial is one independent process and one timed run; never pool P95 values."""
    groups = {}
    for trial in manifest['trials']:
        prefix = raw / trial['name']
        meta = json.loads(Path(str(prefix) + '_metadata.json').read_text())
        with Path(str(prefix) + '_summary.csv').open(newline='') as file:
            rows = {r['metric']: r for r in csv.DictReader(file)}
        key = (trial['mode'], trial['grid'])
        groups.setdefault(key, []).append((trial, meta, rows))
    results = []
    for (mode, grid), trials in groups.items():
        reference = trials[0][1]
        for trial, meta, rows in trials:
            mismatch = [key for key in MATCH if reference.get(key) != meta.get(key)]
            if mismatch:
                raise ValueError('Unmatched workload: ' + ', '.join(mismatch))
            for field in ('density_mass', 'density_checksum', 'density_min', 'density_max'):
                if not math.isclose(reference['results'][0][field], meta['results'][0][field], rel_tol=1e-5, abs_tol=1e-5):
                    raise ValueError('Density signature mismatch: ' + trial['name'])
            if int(rows['step_cpu']['count']) != meta['frames']:
                raise ValueError('Incomplete sample count')
        item = {'mode': mode, 'grid': grid, 'frames': reference['frames'], 'quality_signature_match': True,
                'source_identity_match': True, 'statistics': {}}
        for metric in ('completed_run_wall', 'step_cpu', 'step_gpu_interval', 'frame_cpu',
                       'publish_cpu', 'draw_gpu', 'swap_cpu', 'recycle_wait_cpu'):
            if metric not in trials[0][2]:
                continue
            entry = {}
            for side in ('A', 'B'):
                selected = [rows[metric] for trial, meta, rows in trials if trial['variant'] == side]
                means = [float(r['mean']) for r in selected]
                entry[side] = {'trials': len(means), 'median_mean_ms': statistics.median(means),
                               'min_mean_ms': min(means), 'max_mean_ms': max(means),
                               'median_p95_ms': statistics.median(float(r['p95']) for r in selected),
                               'trial_means_ms': means}
            entry['speedup'] = entry['A']['median_mean_ms'] / entry['B']['median_mean_ms']
            item['statistics'][metric] = entry
        for side in ('A', 'B'):
            durations = [meta['results'][0]['wall_ms'] / 1000 for trial, meta, rows in trials if trial['variant'] == side]
            item[side + '_timed_seconds_range'] = [min(durations), max(durations)]
        # Pair temporally adjacent unlike variants within each ABBA/BAAB block.
        ratios = []
        for i in range(0, len(trials), 2):
            pair = {t[0]['variant']: t[1]['results'][0]['wall_ms'] for t in trials[i:i + 2]}
            ratios.append(pair['A'] / pair['B'])
        item['paired_wall_speedups'] = ratios
        item['paired_wall_speedup_range'] = [min(ratios), max(ratios)]
        results.append(item)
    write_json(output / 'results.json', results)
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['mode', 'grid', 'frames', 'metric', 'A_median_mean_ms', 'B_median_mean_ms',
                         'A_min_mean_ms', 'A_max_mean_ms', 'B_min_mean_ms', 'B_max_mean_ms',
                         'A_median_p95_ms', 'B_median_p95_ms', 'speedup'])
        for result in results:
            for metric, entry in result['statistics'].items():
                writer.writerow([result['mode'], result['grid'], result['frames'], metric,
                    entry['A']['median_mean_ms'], entry['B']['median_mean_ms'],
                    entry['A']['min_mean_ms'], entry['A']['max_mean_ms'],
                    entry['B']['min_mean_ms'], entry['B']['max_mean_ms'],
                    entry['A']['median_p95_ms'], entry['B']['median_p95_ms'], entry['speedup']])
    # Compact per-process summaries/metadata are versionable; full samples stay in the raw archive.
    compact = output / 'reports'
    compact.mkdir(exist_ok=True)
    for trial in manifest['trials']:
        for ext in ('_metadata.json', '_summary.csv'):
            shutil.copy2(raw / (trial['name'] + ext), compact / (trial['name'] + ext))
        samples = raw / (trial['name'] + '_samples.csv')
        if samples.is_file():
            with samples.open('rb') as source, (compact / (samples.name + '.gz')).open('wb') as target:
                with gzip.GzipFile(filename='', mode='wb', fileobj=target, mtime=0) as compressed:
                    shutil.copyfileobj(source, compressed)
    print('Completed matched experiment: ' + str(output), flush=True)


if __name__ == '__main__':
    main()
