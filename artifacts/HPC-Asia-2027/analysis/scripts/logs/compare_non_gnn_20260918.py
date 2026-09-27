"""Recompute the saved CS-3/H100 training-window comparison (standard library).

Run from any directory: python analyze/scripts/logs/compare_non_gnn_20260918.py
CS-3 uses log timestamps at completion of steps 20 and 200; H100 uses
the summary timer for steps 21--200. Startup and warmup are excluded.
"""
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[3] / 'records'
ROOT = ARTIFACTS / 'model_dirs/non_gnn'
RAW = ARTIFACTS / 'raw_logs/hpcasia/non_gnn'
CS = ROOT / 'cerebras/20260918/campaign_csx_20260918_133713_5bbf7dfa'
GPU = ROOT / 'pegasus/20260918/campaign_gpu_20260918_172316_310e21d8'
PATTERN = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}).*Train Device=CSX, Step=(\d+), Loss=([^,]+),')


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare():
    # The shared input manifest is retained once, with the Pegasus records.
    input_manifest = next((ROOT / 'pegasus/20260918/data').rglob('manifest.json'))
    rows = []
    for run in sorted(CS.iterdir()):
        if not run.is_dir():
            continue
        gpu_run = GPU / (run.name + '_native')
        cs, gpu = read_json(run / 'launch.json'), read_json(gpu_run / 'launch.json')
        for key in ('base_config_sha256', 'sequence_length', 'effective_batch_size',
                    'precision', 'warmup_steps', 'max_optimizer_steps'):
            assert cs[key] == gpu[key], (run.name, key)
        for key in ('num_workers', 'seed'):
            assert cs['arguments'][key] == gpu['arguments'][key]
        assert cs['warmup_steps'] == 20 and cs['max_optimizer_steps'] == 200
        assert sha(run / 'params.yaml') == cs['params_sha256']
        assert sha(gpu_run / 'params.yaml') == gpu['params_sha256']
        status = read_json(run / 'client_status.json')
        assert status['state'] == 'completed' and status['exit_code'] == 0
        log = RAW / (run.name + '_cs3.log')
        gpu_metrics = RAW / (run.name + '_h100.metrics.jsonl')
        points = [PATTERN.match(line) for line in log.read_text().splitlines()]
        points = [p for p in points if p]
        assert [int(p[2]) for p in points] == list(range(1, 201))
        assert all(math.isfinite(float(p[3])) for p in points)
        assert 'Training completed successfully!' in log.read_text()
        elapsed = (datetime.strptime(points[199][1], '%Y-%m-%d %H:%M:%S,%f')
                   - datetime.strptime(points[19][1], '%Y-%m-%d %H:%M:%S,%f')).total_seconds()
        events = [json.loads(line) for line in gpu_metrics.read_text().splitlines()]
        train = [e for e in events if e['event'] == 'train']
        assert [e['step'] for e in train] == list(range(1, 201))
        assert all(math.isfinite(e['loss']) for e in train)
        assert all(e['warmup'] == (e['step'] <= 20) for e in train)
        summaries = [e for e in events if e['event'] == 'summary']
        assert len(summaries) == 1
        summary = summaries[0]
        gpu_seconds = summary['elapsed_seconds']
        endpoint_seconds = train[-1]['elapsed_seconds'] - train[19]['elapsed_seconds']
        assert abs(gpu_seconds - endpoint_seconds) < 0.001
        samples = 180 * cs['effective_batch_size']
        assert samples == train[-1]['samples'] - train[19]['samples'] == summary['samples']
        assert summary['measured_optimizer_steps'] == 180 and summary['warmup_steps'] == 20
        gpu_rate = samples / gpu_seconds
        assert math.isclose(gpu_rate, summary['samples_per_second'], rel_tol=1e-9)
        cs_rate = samples / elapsed
        rows.append(dict(profile=run.name, measured_steps=180,
                         sequence_length=cs['sequence_length'], effective_batch_size=cs['effective_batch_size'],
                         cs3_window_seconds=elapsed, h100_window_seconds=gpu_seconds,
                         cs3_samples_per_second=cs_rate, h100_samples_per_second=gpu_rate,
                         cs3_nominal_tokens_per_second=cs_rate * cs['sequence_length'],
                         h100_nominal_tokens_per_second=gpu_rate * cs['sequence_length'],
                         cs3_over_h100=cs_rate / gpu_rate,
                         cs3_log=str(log.relative_to(ARTIFACTS)), cs3_log_sha256=sha(log),
                         h100_metrics=str(gpu_metrics.relative_to(ARTIFACTS)),
                         h100_metrics_sha256=sha(gpu_metrics)))
    assert len(rows) == 4
    return {'source_paths_relative_to': 'artifacts',
            'input_manifest_sha256': sha(input_manifest),
            'window': 'Steps 21--200 after 20 warmup steps; CS-3 log timestamps, H100 summary timer',
            'scope': 'One run per profile and platform; descriptive throughput, no statistical significance test',
            'rows': rows}


if __name__ == '__main__':
    print(json.dumps(compare(), indent=2))
