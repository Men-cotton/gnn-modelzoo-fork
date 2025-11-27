# Cerebras WSE Log Visualization ExecPlan

This ExecPlan is a living document. Keep the `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` sections current as work proceeds. This document must be maintained in accordance with agent/PLANS.md.

## Purpose / Big Picture

Enable anyone to parse Cerebras wafer-scale engine (WSE) log text and render high-resolution Matplotlib figures that reveal how wafer I/O (WIO) ports connect to core rows, how periphery modules consume area, how shards map to FPGA domains, and where memory buffers sit relative to the compute core. After implementing this plan, a user can run a single Python entrypoint that ingests the bundled log text and produces multiple plots (connectivity density, periphery floorplan, domain heatmap, and buffer layout) plus structured data ready for further analysis.

## Progress

- [x] (2025-11-27 14:05Z) Drafted initial ExecPlan based on agent/GOALS.md and repository inspection.
- [x] (2025-11-27 14:14Z) Added CerebrasLogParser, WSEVisualizer, bundled sample log module, CLI entrypoint, and initial parser/CLI tests.
- [x] (2025-11-27 14:20Z) Parser/visualizer/CLI tests pass via `UV_CACHE_DIR=.uv_cache uv run python -m unittest discover -s analyze/tests -p 'test_*.py'`; connectivity/floorplan/domain/buffer plots generate headlessly.
- [ ] (pending) Refine visuals if gaps emerge during broader validation and update Outcomes/Retrospective accordingly.

## Surprises & Discoveries

- Observation: Buffer headings include lowercase tokens (e.g., "pre-demux") and right-edge module listings follow immediately after left-edge rows; regex and block termination needed to capture all buffers and right modules.
  Evidence: Parser tests initially failed to find ACT sink buffers and right-edge PAD until regex broadened and module parsing stopped at the next edge header; fixed in cerebras_log_parser.py and tests now pass.

## Decision Log

- Decision: Remove bundled SAMPLE_WSE_LOG in favor of requiring an explicit --input file (use analyze/wio_report_1.txt in tests); simplifies dependency surface and keeps outputs tied to concrete log files.
  Rationale: Avoid stale embedded fixtures and keep CLI/test behavior aligned with real inputs.
  Date/Author: 2025-11-27 / assistant

## Outcomes & Retrospective

Early outcome: Parser, visualizer, and CLI generate all four plots headlessly from the bundled sample log; unit and smoke tests pass under `uv run`. Remaining to capture: any visual refinement or additional validation results once broader usage feedback is available.

## Context and Orientation

The repository already contains a WIO summary visualizer (analyze/wio_visualizer.py) with tests under analyze/tests and sample WIO summary reports in analyze/wio_report_1.txt through analyze/wio_report_3.txt. That summary script parses high-level WIO summaries and renders placement/bar plots. The new work targets the richer Cerebras WSE log format (as seen in analyze/wio_report_1.txt) that includes fabric geometry, compute core origin/size, buffer dimensions with y_offset values, detailed connectivity listings per flow/domain/edge, floorplan module ranges, and shard-to-domain mappings. The deliverable is a new parser class and visualization suite (distinct from the existing summary utility) that can run headlessly via Matplotlib’s Agg backend and ships with the log text embedded as a string to keep the workflow reproducible. The Python toolchain uses uv for isolated runs (see analyze/README.md); figures should be written to analyze/output/, which already exists and is safe for generated artifacts.

## Plan of Work

Begin by creating a dedicated module (analyze/cerebras_log_parser.py) that defines a CerebrasLogParser class responsible for ingesting raw log text, normalizing line endings, and extracting structured records. Implement targeted parsing helpers for geometry/core metadata, floorplan modules, connectivity rows, domain/shard mappings, and buffer block definitions. Use regex to capture flow type, edge, domain, and row lists for connectivity and expand comma-separated ranges into individual pairings. Return Pandas DataFrames (or well-structured lists of dicts when lighter) with clear columns such as flow_type, edge, domain, wio_y, and core_row_abs so downstream plotting is straightforward.

Create a companion visualization module (analyze/wse_visualizer.py) with a WSEVisualizer class that accepts parsed data and renders four plots: (1) a bipartite connectivity LineCollection view with WIO dots at x=-50/750 and core rows across the core span, colored by flow type with alpha blending to reveal density and fan-out markers; (2) a periphery floorplan strip chart using rectangles and a broken axis to show left/right periphery zones while skipping the core bulk, explicitly labeling wide modules such as CSG_BUF and PAD; (3) a shard-to-domain heatmap (shards as rows, domains as columns) using imshow/pcolormesh with in-cell annotations; and (4) a buffer layout schematic that draws the compute core rectangle plus edge-attached buffer blocks positioned by their column widths, row counts, and y_offset distances. Reuse a consistent flow-type color palette across plots.

Add an executable Python entrypoint (analyze/wse_log_viz.py) that loads a provided log file via --input (no bundled sample), invokes CerebrasLogParser to produce structured data, and drives WSEVisualizer to emit PNG outputs into analyze/output/. Provide CLI flags for choosing output directory/format and toggling which plots to generate. Keep rendering headless (Agg) and set dpi=300 per GOALS.

Extend the test suite in analyze/tests with focused unit tests for the parser (geometry fields, connectivity expansion, floorplan widths, buffer offsets, and domain mapping) using the bundled log string as a fixture. Add a smoke test for the CLI that writes to a temporary directory and asserts files exist. Keep tests deterministic by seeding Matplotlib if needed and avoiding reliance on system fonts.

Document how to run the tool and tests in the Concrete Steps and Validation sections, and update living sections (Progress, Decisions, Surprises) as implementation proceeds. Ensure new code is additive and does not regress the existing WIO visualizer.

## Concrete Steps

From the repository root, ensure dependencies are installed (uv handles Matplotlib):
    UV_CACHE_DIR=.uv_cache uv pip install -r requirements.txt

Run the forthcoming CLI to generate all plots from the bundled log string:
    UV_CACHE_DIR=.uv_cache uv run python analyze/wse_log_viz.py --output analyze/output --format png --all
Expect PNG files (connectivity, floorplan, domain_heatmap, buffer_layout) in analyze/output/.

Execute the test suite (including new parser/render smoke tests):
    UV_CACHE_DIR=.uv_cache uv run python -m unittest discover -s analyze/tests -p 'test_*.py'
All tests should pass; new parser tests should fail before implementation and pass after.

## Validation and Acceptance

Acceptance requires that running the CLI produces four dpi=300 images in analyze/output/ using only the bundled log text. The connectivity plot must show left/right WIO dots at the reported y positions with colored lines to each target core row and visually highlight fan-out cases. The floorplan strip must show only periphery regions with correctly scaled module widths and labels for CSG_BUF and PAD. The domain heatmap must show shards on rows, domains on columns, and in-cell annotations of counts. The buffer layout schematic must place ACT/EVT/WGT/GRD blocks around the core with y_offset respected. Parser unit tests must cover geometry, connectivity expansion, floorplan parsing, buffer offsets, and domain mapping; the smoke test must confirm images are generated when run headlessly.

## Idempotence and Recovery

The CLI should be safe to rerun: it overwrites outputs in analyze/output/ and recreates the directory if missing. Parsing functions should tolerate repeated calls with the same input string. If rendering fails midway, rerun after cleaning only the affected output files; no persistent state beyond images is expected. Tests are read-only aside from temporary output directories they create and delete.

## Artifacts and Notes

Store generated images in analyze/output/ for inspection. Keep the bundled log string colocated with the CLI to avoid external dependencies. If intermediate CSV/JSON exports aid debugging, write them to analyze/output/ with clear names and document their purpose here during execution.

## Interfaces and Dependencies

Implement CerebrasLogParser in analyze/cerebras_log_parser.py with methods such as parse_geometry(), parse_floorplan(), parse_connectivity(), parse_domain_mapping(), and parse_buffers(), each returning DataFrames or lists of dicts with explicit columns (e.g., flow_type, edge, domain, wio_y, core_row_abs). Implement WSEVisualizer in analyze/wse_visualizer.py with methods draw_connectivity(df), draw_floorplan(df), draw_domain_heatmap(df), and draw_buffer_layout(buffers, geometry) returning Matplotlib Figure objects. Implement a CLI in analyze/wse_log_viz.py that wires parser outputs to visualizer methods, saves figures with dpi=300 using the Agg backend, and accepts flags --output, --format (png|svg), and plot selection switches.

Change note (2025-11-27): Initial ExecPlan added to guide Cerebras WSE log visualization per agent/GOALS.md.
Change note (2025-11-27 14:14Z): Recorded parser/visualizer/CLI implementation progress and decision to bundle sample log for offline, deterministic runs.
Change note (2025-11-27 14:20Z): Documented parsing adjustments for buffer naming and edge boundaries; tests now passing headlessly via uv.
