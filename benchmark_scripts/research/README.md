# GNN research experiments

The executable models and input factories live in
[`cerebras.modelzoo.models.gnn.research`](../../src/cerebras/modelzoo/models/gnn/research).
They use the same `src` import path, installed environment and shared data
location as the other GNN jobs. There is no separate research shell dispatcher
or site configuration file.

- [Shared tree features](feature_sharing/device/DEPLOY.md): native GraphSAGE
  with dense inputs or a dictionary of shared features.
- [History and exact feature means](history_embedding/DEPLOY.md): recursive,
  exact raw-feature means and immutable epoch histories.
- `logs/` and `feature_sharing/probe_shared_tree_features.py`: local historical
  reproduction tools. They remain local probes; their `--baseline` arguments
  refer to saved results.

On ALCF, use the existing shared checkout, `.venv`, SDK authentication and
processed OGB datasets. The `csx-shared-tree`, `csx-history-pilot` and
`csx-history` entries in `benchmark_scripts/jobs.json` use the existing
`benchmark_launcher jobs` path. It fixes one committed source per run and
launches the driver in tmux. Child processes inherit that Python environment;
Worker imports use the pinned `src`. Inputs, outputs and compilation files
must be on the shared filesystem, as for the existing GNN campaigns.

`jobs --dry-run` previews without launching. Uncommitted edits are excluded
from a source run. Local development checks can call modules directly:

```bash
PYTHONPATH=src uv run --no-sync -- python -m cerebras.modelzoo.models.gnn.research.history_embedding.pilot --help
```

Wafer-GNN-manager retains saved-result analysis, plots and raw results.
