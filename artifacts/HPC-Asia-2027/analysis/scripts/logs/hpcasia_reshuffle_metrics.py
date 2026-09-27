"""Recompute the supplementary reshuffle claims from the two saved jobs."""
import json
import sys
from hpcasia_evidence import ROOT, read_tsv
from hpcasia_learning_analysis import parse_raw, require

sys.path.insert(0, str(ROOT.parents[1] / 'src'))
from cerebras.modelzoo.models.gnn.tools import measure_window


def main():
    rows = read_tsv(ROOT / 'records/raw_logs/hpcasia/RUNS.tsv')
    old = [r for r in rows if r['purpose'] == 'learning' and r['platform'] == 'cs3'
           and r['dataset'] == 'products' and r['seed'] == '42']
    new = [r for r in rows if r['purpose'] == 'reshuffle']
    require(len(old) == len(new) == 1, 'Unexpected learning cohort')
    learning = {}
    for role, row in [('original', old[0]), ('reshuffle', new[0])]:
        result = json.loads((ROOT / row['result']).read_text())
        require(row['status'] == result['status'] == 'completed' and result['returncode'] == 0,
                'Incomplete learning job')
        ev, losses, contracts, _ = parse_raw((ROOT / row['log']).read_text(), 'cs3')
        require(len(ev) == 25 and len(losses) == 100 and len(contracts) == 1, 'Missing observations')
        learning[role] = {'accuracy': ev[-1]['accuracy'],
                          'losses': {r['step']: r['loss'] for r in losses}, 'contract': contracts[0]}
    a, b = learning['original'], learning['reshuffle']
    require(a['contract']['ordered_targets_and_labels_sha256'] == b['contract']['ordered_targets_and_labels_sha256'],
            'First-pass targets changed')
    require(b['contract']['target_order'] == 'reshuffle_each_epoch', 'Missing reshuffle contract')
    first = min(s for s in a['losses'] if a['losses'][s] != b['losses'][s])
    old = [r for r in rows if r['purpose'] == 'throughput' and r['platform'] == 'cs3'
           and r['dataset'] == 'products' and r['seed'] == '42' and '/throughput_r' in r['run']]
    new = [r for r in rows if r['purpose'] == 'reshuffle_throughput']
    require(len(old) == 3 and len(new) == 1, 'Unexpected throughput cohort')
    rates = {}
    for role, cohort in [('original', old), ('reshuffle', new)]:
        rates[role] = []
        for row in cohort:
            saved = json.loads((ROOT / row['result']).read_text())
            measured = measure_window.summarize(ROOT / row['log'], 40, 1640)
            require(saved['returncode'] == 0 and measured == saved['measurement'], 'Saved measurement mismatch')
            rates[role].append(measured['throughput'])
    output = {'first_different_logged_loss_step': first,
              'final_accuracy': {k: v['accuracy'] for k, v in learning.items()},
              'tail_losses': {k: {str(s): v['losses'][s] for s in (490, 980)} for k, v in learning.items()},
              'baseline_throughput_min': min(rates['original']),
              'baseline_throughput_max': max(rates['original']),
              'reshuffle_throughput': rates['reshuffle'][0]}
    target = ROOT / 'results/reshuffle/metrics.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
