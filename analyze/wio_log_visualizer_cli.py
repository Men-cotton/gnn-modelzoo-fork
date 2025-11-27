"""
CLI entrypoint for Cerebras WSE log visualization.

Requires a log file path via --input. Renders connectivity, floorplan, domain heatmap,
buffer layout, and summary figures into analyze/output/ (or a user-provided directory).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position

# Allow running as a script (python analyze/wio_log_visualizer_cli.py) by adding repo root.
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from analyze.wio_summary_visualizer import parse_wio_report, render_wio_report
from analyze.wio_log_parser import CerebrasLogParser
from analyze.wio_log_visualizer import WSEVisualizer


def render_figures(
    raw_text: str,
    output_dir: Path,
    fmt: str,
    plots: Iterable[str],
    summary_report: Optional[Path] = None,
) -> List[Path]:
    parser = CerebrasLogParser(raw_text)
    parsed = parser.parse()
    visualizer = WSEVisualizer()

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    selected = list(plots)
    if not selected:
        selected = ["connectivity", "floorplan", "domain_heatmap", "buffer_layout"]

    if "connectivity" in selected:
        fig = visualizer.draw_connectivity(parsed)
        path = output_dir / f"connectivity.{fmt}"
        fig.savefig(path, dpi=300, format=fmt, bbox_inches="tight")
        plt.close(fig)
        generated.append(path)

    if "floorplan" in selected:
        fig = visualizer.draw_floorplan(parsed)
        path = output_dir / f"floorplan.{fmt}"
        fig.savefig(path, dpi=300, format=fmt, bbox_inches="tight")
        plt.close(fig)
        generated.append(path)

    if "domain_heatmap" in selected:
        fig = visualizer.draw_domain_heatmap(parsed)
        path = output_dir / f"domain_heatmap.{fmt}"
        fig.savefig(path, dpi=300, format=fmt, bbox_inches="tight")
        plt.close(fig)
        generated.append(path)

    if "buffer_layout" in selected:
        fig = visualizer.draw_buffer_layout(parsed)
        path = output_dir / f"buffer_layout.{fmt}"
        fig.savefig(path, dpi=300, format=fmt, bbox_inches="tight")
        plt.close(fig)
        generated.append(path)

    if "summary" in selected and summary_report and summary_report.exists():
        report = parse_wio_report(summary_report)
        path = output_dir / f"summary.{fmt}"
        render_wio_report(report, path, fmt=fmt)
        generated.append(path)

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Cerebras WSE log visualizations (connectivity, floorplan, domains, buffers)."
    )
    parser.add_argument(
        "--input",
        "-i",
        default=None,
        help="Path to a log file (also used for the summary plot).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="analyze/output",
        help="Directory for generated figures (default: analyze/output).",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["png", "svg"],
        default="png",
        help="Image format.",
    )
    parser.add_argument(
        "--connectivity",
        action="store_true",
        help="Render only the connectivity plot.",
    )
    parser.add_argument(
        "--floorplan",
        action="store_true",
        help="Render only the floorplan plot.",
    )
    parser.add_argument(
        "--domain-heatmap",
        dest="domain_heatmap",
        action="store_true",
        help="Render only the domain heatmap.",
    )
    parser.add_argument(
        "--buffer-layout",
        dest="buffer_layout",
        action="store_true",
        help="Render only the buffer layout plot.",
    )
    parser.add_argument(
        "--all",
        dest="render_all",
        action="store_true",
        help="Render all plots (default if none specified).",
    )

    args = parser.parse_args()

    plot_flags = []
    if args.connectivity:
        plot_flags.append("connectivity")
    if args.floorplan:
        plot_flags.append("floorplan")
    if args.domain_heatmap:
        plot_flags.append("domain_heatmap")
    if args.buffer_layout:
        plot_flags.append("buffer_layout")
    if args.render_all or not plot_flags:
        plot_flags = [
            "connectivity",
            "floorplan",
            "domain_heatmap",
            "buffer_layout",
            "summary",
        ]

    raw_text: str
    input_path: Path | None = None
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise SystemExit(f"Input log not found: {input_path}")
        raw_text = input_path.read_text()
    else:
        raise SystemExit("Error: --input is required to render plots.")

    output_dir = Path(args.output)
    summary_path = input_path
    generated = render_figures(
        raw_text, output_dir, args.format, plot_flags, summary_report=summary_path
    )
    for path in generated:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
