"""Recompute the paper's recorded results without network or accelerator access."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def tsv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Compare recomputed tables and summaries with reported values')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    entries = manifest['records'] + manifest['package_files']
    for entry in entries:
        path = ROOT / entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise SystemExit('Hash mismatch: ' + entry['path'])
    print(f'Verified {len(entries)} bundled files.', flush=True)
    logs = ROOT / 'results/reproduction_logs'
    logs.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, MPLBACKEND='Agg', PYTHONHASHSEED='0')
    environment.pop('PYTHONPATH', None)
    scripts = [
        'logs/hpcasia_evidence.py',
        'logs/hpcasia_learning_analysis.py',
        'logs/hpcasia_input_analysis.py',
        'logs/hpcasia_supplementary_analysis.py',
        'logs/hpcasia_worker_repeat_audit.py',
        'logs/hpcasia_reshuffle_metrics.py',
        'paper/build_hpcasia_tables.py',
    ]
    for name in scripts:
        result = subprocess.run([sys.executable, str(ROOT / 'analysis/scripts' / name)],
                                cwd=ROOT.parents[1], env=environment,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (logs / (Path(name).stem + '.log')).write_text(result.stdout)
        if result.returncode:
            print(result.stdout, file=sys.stderr)
            raise SystemExit(f'Failed: {name}')
        print(f'Recomputed {Path(name).stem}', flush=True)
    if args.check:
        for expected in (ROOT / 'reported/tables').glob('*.tex'):
            if expected.read_bytes() != (ROOT / 'tables' / expected.name).read_bytes():
                raise SystemExit('Table mismatch: ' + expected.name)
        for name in ('learning/summary.tsv', 'throughput/summary.tsv',
                     'worker_repeat_review/summary.tsv'):
            expected = ROOT / 'reported' / name
            actual = ROOT / 'results' / expected.relative_to(ROOT / 'reported')
            if tsv(expected) != tsv(actual):
                raise SystemExit('Summary mismatch: ' + str(expected.relative_to(ROOT)))
        for expected in (ROOT / 'reported').glob('*/*.json'):
            actual = ROOT / 'results' / expected.relative_to(ROOT / 'reported')
            if json.loads(expected.read_text()) != json.loads(actual.read_text()):
                raise SystemExit('Metric mismatch: ' + str(expected.relative_to(ROOT)))
        print('Reported metrics match: 12 learning jobs, 48 throughput jobs, 24 matched worker jobs.')


if __name__ == '__main__':
    main()
