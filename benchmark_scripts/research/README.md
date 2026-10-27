# GNN research execution

This directory owns experimental GNN execution, data preparation and numerical
reference runs. Invoke Python entrypoints with `python -m` from the fork root;
use `uv run --no-sync -- python -m MODULE` with `PYTHONPATH=$PWD:$PWD/src`
in the client and both paths in the input Worker.
`benchmark_scripts/research/run.sh MODULE ...` sets this up automatically.
The script supports `launcher` for the existing tmux launcher; all driver
subprocesses also use this uv project.
Use the existing `cerebras.modelzoo.tools.benchmark_launcher` for tmux.

- [History and exact feature means](history_embedding/DEPLOY.md): full real OGB
  graphs, SDK CPU/CSX runs, CUDA comparisons, checkpoint evaluation.
- [Shared tree features](feature_sharing/device/DEPLOY.md): real-data paired
  dense/dictionary probes importing the fork's native GraphSAGE.
- `logs`: existing local HPC Asia reproduction runners and their samplers.
  Historical baselines are explicit `--baseline` paths; factorial also accepts
  `--sampling-baseline`. These are local reproduction tools, not new CSX jobs.

Wafer-GNN-manager retains raw results, plots and scripts that analyze them.
It must not contain a second executable GNN implementation or runtime archive.
