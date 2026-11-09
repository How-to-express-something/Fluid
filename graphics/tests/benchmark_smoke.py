"""Verify benchmark report contracts, never assert hardware speed thresholds."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--exe', required=True)
args = parser.parse_args()
exe = str(Path(args.exe).resolve())

def run(options, expected=0):
    result = subprocess.run([exe, *options], capture_output=True, text=True, errors='replace', timeout=45)
    if result.returncode != expected:
        raise AssertionError(f'exit={result.returncode}, expected={expected}\n{result.stdout}\n{result.stderr}')
    return result

with tempfile.TemporaryDirectory(prefix='fluid_benchmark_') as directory:
    root = Path(directory)
    common = ['--grid', '31', '19', '--window', '320', '240', '--frames', '9', '--warmup', '2',
              '--repeats', '2', '--inflight', '4', '--jacobi', '7', '--viscosity', '0.2', '--diffuse', '3', '--seed', '123']
    signatures = []
    for name, mode, extra in [('solver', 'solver', []), ('pipeline', 'pipeline', []), ('gray', 'pipeline', ['--gray'])]:
        prefix = root / name
        options = ['--benchmark', mode, *common, *extra, '--label', 'smoke "quoted"', '--output', str(prefix)]
        run(options)
        metadata = json.loads(Path(str(prefix) + '_metadata.json').read_text(encoding='utf-8'))
        with Path(str(prefix) + '_samples.csv').open(newline='') as file:
            samples = list(csv.DictReader(file))
        with Path(str(prefix) + '_summary.csv').open(newline='') as file:
            summary = list(csv.DictReader(file))
        assert len(samples) == 18 and len(metadata['results']) == 2
        assert metadata['mode'] == mode and metadata['label'] == 'smoke "quoted"'
        assert metadata['grid_width'] == 31 and metadata['grid_height'] == 19
        assert metadata['solver_buffer_bytes'] == 31 * 19 * 44
        assert metadata['density_texture_payload_bytes'] == (31 * 19 * 4 if mode == 'pipeline' else 0)
        assert metadata['color'] == (name != 'gray')
        for repeat in range(2):
            block = [s for s in samples if int(s['run']) == repeat]
            assert [int(s['sample']) for s in block] == list(range(9))
            assert [int(s['simulation_step']) for s in block] == list(range(3, 12))
        pipeline_columns = ['publish_cpu_ms', 'draw_submit_cpu_ms', 'draw_gpu_ms', 'swap_cpu_ms']
        for row in samples:
            for column, value in row.items():
                if column in ['run', 'sample', 'simulation_step']:
                    continue
                if mode == 'solver' and column in pipeline_columns:
                    assert value == ''
                else:
                    assert math.isfinite(float(value)) and float(value) >= 0
        for row in summary:
            assert int(row['count']) > 0
            for column in ['mean', 'p50', 'p95', 'max']:
                assert math.isfinite(float(row[column])) and float(row[column]) >= 0
            assert float(row['p50']) <= float(row['p95']) <= float(row['max'])
        first, second = metadata['results']
        assert math.isclose(first['density_checksum'], second['density_checksum'], rel_tol=1e-7)
        signatures.append(first['density_checksum'])
        # Reports should not silently overwrite a baseline on a second run.
        before = Path(str(prefix) + '_samples.csv').read_bytes()
        run(options, expected=1)
        assert Path(str(prefix) + '_samples.csv').read_bytes() == before
    assert all(math.isclose(s, signatures[0], rel_tol=1e-7) for s in signatures)
    comparison = Path(__file__).resolve().parents[2] / 'scripts' / 'compare_benchmarks.py'
    same = subprocess.run([sys.executable, str(comparison), '--baseline', str(root / 'solver'),
                           '--candidate', str(root / 'solver')], capture_output=True, text=True)
    assert same.returncode == 0, same.stderr
    unmatched = subprocess.run([sys.executable, str(comparison), '--baseline', str(root / 'solver'),
                                '--candidate', str(root / 'pipeline')], capture_output=True, text=True)
    assert unmatched.returncode == 2 and 'Unmatched workload' in unmatched.stderr
    run(['--benchmark', 'solver', '--frames', '0'], expected=2)
    # Exercise pool draining with more slots than measured samples and no warm-up.
    run(['--benchmark', 'solver', '--grid', '2', '3', '--frames', '1', '--warmup', '0',
         '--repeats', '1', '--inflight', '8', '--output', str(root / 'one')])
print('PASS: real solver/pipeline reports, sample counts, reproducibility, gray parity, overwrite guard, short runs')
