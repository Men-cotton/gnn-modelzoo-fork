# Cerebras WIO and WSE visualizers

Two small utilities live here:
- `wio_summary_visualizer.py` reads wafer I/O (WIO) summary reports (`analyze/raw_report/wio_report_*.txt`) and renders per-flow stacked bars, edge placement dots, and a fabric outline.
- `wio_log_visualizer_cli.py` is the CLI/driver that parses full Cerebras WSE logs and renders connectivity, periphery floorplan, shard-domain heatmaps, buffer layouts, and (optionally) the WIO summary plot above.

## WIO report quick start

Run from the repo root:

```bash
UV_CACHE_DIR=.uv_cache uv run python analyze/wio_summary_visualizer.py \
  --input analyze/raw_report/wio_report_1.txt \
  --show-summary \
  --output analyze/output/wio_report_1.png
```

To render all figures (connectivity, floorplan, domains, buffers, and the WIO summary) from the same report:

```bash
UV_CACHE_DIR=.uv_cache uv run python analyze/wio_log_visualizer_cli.py \
  --input analyze/raw_report/wio_report_1.txt \
  --output analyze/output \
  --format png
```

Outputs land in `analyze/output/` by default; use `--format svg` for vector output or pass multiple `--input` files to batch render (then `--output` must be a directory).

Select specific plots with flags such as `--connectivity`, `--floorplan`, `--domain-heatmap`, `--buffer-layout`, or `--all` (default). The summary plot uses the same input path as the WIO report.

## Requirements

Both scripts depend on matplotlib (installed via `uv pip install matplotlib==3.9.2`). Rendering uses the Agg backend, so no GUI is required. Set `UV_CACHE_DIR=.uv_cache` with `uv run` if your default UV cache is not writable.

## Tests

Run the regression suite from repo root:

```bash
UV_CACHE_DIR=.uv_cache uv run python -m unittest discover -s analyze/tests -p 'test_*.py'
```

Individual files:
- `analyze/tests/test_wio_summary_visualizer.py`
- `analyze/tests/test_wio_log_visualizer_cli.py`
- `analyze/tests/test_wio_log_parser.py`
