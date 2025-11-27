# WIO Report Visualizer

This utility parses the wafer I/O (WIO) reports in this directory and renders quick visual summaries of utilization and placement along the fabric edges.

## What it does

- Reads `analyze/wio_report_*.txt` files and extracts fabric geometry, flow-level WIO counts, and per-WIO placement by edge/domain.
- Generates PNG or SVG images that show per-flow stacked bars and placement dots, plus a text summary of total/left/right WIO usage.
- Prints a textual summary when requested.

## Quick start

Run from the repo root:

```bash
UV_CACHE_DIR=.uv_cache uv run python analyze/wio_visualizer.py \
  --input analyze/wio_report_1.txt \
  --show-summary \
  --output analyze/output/wio_report_1.png
```

Outputs land in `analyze/output/` by default; the directory is created as needed. Use `--format svg` for vector output or pass multiple `--input` files to batch render (then `--output` must be a directory).

## Requirements

The visualizer depends on matplotlib (installed when you run `uv pip install matplotlib==3.9.2`). Rendering is headless (Agg backend) so no GUI is needed. Use `UV_CACHE_DIR=.uv_cache` with `uv run` if your default UV cache is not writable.

## Tests

Run the small regression suite:

```bash
UV_CACHE_DIR=.uv_cache uv run python -m unittest discover -s analyze/tests -p 'test_wio_visualizer.py'
```

Tests cover parsing correctness and verify that rendering produces an image file.
