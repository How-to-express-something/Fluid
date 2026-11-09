"""Known-value checks for the retest report contract; no GPU or timing thresholds."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retest_pressure_graph import summarize


class ReportContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / 'raw'
        self.out = self.root / 'out'
        self.raw.mkdir()
        self.out.mkdir()
        self.manifest = {'trials': []}
        for index, (side, wall) in enumerate([('A', 40), ('B', 20), ('B', 22), ('A', 44)]):
            name = str(index)
            trial = {'name': name, 'variant': side, 'mode': 'solver', 'grid': 256}
            self.manifest['trials'].append(trial)
            meta = {'mode': 'solver', 'frames': 10, 'repeats': 1, 'pressure_backend': side,
                    'source_snapshot_id': side, 'results': [{'wall_ms': wall, 'density_mass': 5,
                    'density_checksum': 15, 'density_min': 0, 'density_max': 3}]}
            (self.raw / (name + '_metadata.json')).write_text(json.dumps(meta), encoding='utf-8')
            with (self.raw / (name + '_summary.csv')).open('w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['metric', 'count', 'mean', 'p95'])
                writer.writerow(['completed_run_wall', 1, wall, wall])
                writer.writerow(['step_cpu', 10, wall / 10, wall / 5])

    def tearDown(self):
        self.temp.cleanup()

    def test_known_medians_ranges_and_pairing(self):
        summarize(self.raw, self.out, self.manifest)
        result = json.loads((self.out / 'results.json').read_text(encoding='utf-8'))[0]
        wall = result['statistics']['completed_run_wall']
        self.assertEqual(wall['A']['median_mean_ms'], 42)
        self.assertEqual(wall['B']['median_mean_ms'], 21)
        self.assertEqual(wall['speedup'], 2)
        self.assertEqual(result['paired_wall_speedups'], [2, 2])
        self.assertEqual(result['A_timed_seconds_range'], [.04, .044])

    def change_metadata(self, field, value):
        path = self.raw / '1_metadata.json'
        meta = json.loads(path.read_text(encoding='utf-8'))
        if field == 'mass':
            meta['results'][0]['density_mass'] = value
        else:
            meta[field] = value
        path.write_text(json.dumps(meta), encoding='utf-8')

    def test_rejects_workload_mismatch(self):
        self.change_metadata('frames', 11)
        with self.assertRaisesRegex(ValueError, 'Unmatched workload'):
            summarize(self.raw, self.out, self.manifest)

    def test_rejects_changed_density(self):
        self.change_metadata('mass', 100)
        with self.assertRaisesRegex(ValueError, 'Density signature mismatch'):
            summarize(self.raw, self.out, self.manifest)


if __name__ == '__main__':
    unittest.main()
