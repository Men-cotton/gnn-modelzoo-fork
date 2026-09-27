# HPC Asia 2027 GraphSAGE artifact

This artifact accompanies the September 2026 measurements of GraphSAGE through Cerebras Model Zoo on physical CS-3 and PyG on H100. The `HPC-Asia-2027` branch contains the implementation, recorded inputs, and offline analysis. It is a separate publication branch, following the `IA3-2026` artifact convention. No sibling repository, private analysis checkout, Grafana account, or accelerator is needed to recompute the reported values.

## Recompute the reported results

From this repository's root, with Python 3.12:

```sh
python -m venv .venv-artifact
. .venv-artifact/bin/activate
python -m pip install -r artifacts/HPC-Asia-2027/requirements.txt
python artifacts/HPC-Asia-2027/scripts/reproduce_metrics.py --check
```

The command verifies bundled file hashes, parses the saved training logs, and compares four generated manuscript tables and three unrounded summary TSVs with `reported/`. It also checks the two supplementary reshuffle jobs against their reported accuracy, tail losses, and throughput. It writes intermediate results to `results/` and generated LaTeX to `tables/`. A successful check ends with `Reported metrics match`. Package installation needs network access unless dependencies are already available; recomputation is offline.

## Records and cohorts

`records/raw_logs/hpcasia/RUNS.tsv` selects 253 active records by purpose. The primary comparisons comprise 12 learning jobs and 48 throughput jobs. Learning retains all 300 validation observations. Worker sensitivity uses 24 completed jobs, six per worker count (4, 8, 12, 16). Saved Grafana responses cover these same 24 jobs plus 15 input controls and three diagnostic jobs. These responses and their reported resource summary are retained as recorded observations; the reproduction command does not acquire or recompute Grafana metrics. Incomplete attempts and superseded configurations remain in the 70-row `records/raw_logs/invalid/hpcasia/RUNS.tsv` exclusion index. The two target-reshuffle runs have separate purposes and are not part of the primary learning or throughput comparison.

`FILES.tsv` maps retained raw files to their model-directory copies. Model directories contain resolved configurations and available results, launch metadata, and SDK performance records. SDK executor entries are not independent training runs. The one-run-per-job indices, timestamp boundaries, actual-target counts, nominal slots, and source fields remain available for inspection.

`MANIFEST.json` records the hashes of original and sanitized records and the hashes of the bundled analysis, historical sources, and reported values. Operational paths and dashboard URLs have been sanitized, and file references and hashes have been updated accordingly. This manifest checks internal consistency; it is not independent authentication of the measurements. No browser cookies, authentication headers, compiled executables, datasets, or checkpoints are required or bundled.

## Implementation and fresh execution

The branch's root implementation starts from revision `390c31772d97f2105b98df77429e27354cef9b69`. `sources/archives/<revision>.tar.gz` contains the complete tracked checkout for the nine GraphSAGE source revisions recorded in the run environment metadata. Extract the relevant archive into an empty directory when reproducing a historical implementation. Small source copies under `sources/<revision>/` support the worker-cohort source-identity check without Git history; shallow clones work.

The five primary CS-3 learning jobs with recorded revisions predate per-pass target reshuffling. The sixth retry has no separate revision record; this package does not infer a revision for it. The later reshuffle runs remain supplementary. Source revisions and configurations should be selected per run rather than substituting the branch-root implementation for every historical job.

Fresh execution additionally requires compatible OGB datasets, the recorded software environment, and either H100 access or allocation on a compatible CS-3 appliance. Refer to the selected source snapshot's GNN README and the bundled per-run configurations and launch records. Dataset and operational paths in sanitized records must be supplied for the new environment. Offline reproduction checks saved-record metrics; it does not launch training or reproduce compiler-graph checks, because compiler outputs are omitted.
