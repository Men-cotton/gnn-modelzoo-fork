"""
Visualize Cerebras wafer I/O (WIO) reports produced under analyze/wio_report_*.txt.

Parses the textual report, summarizes WIO utilization, and renders a PNG/SVG
showing per-flow counts and placement along the fabric edges.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
except ImportError:  # pragma: no cover - handled at runtime
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]


FLOW_COLORS: Dict[str, str] = {
    "ACT": "#2ca02c",
    "WGT": "#ff7f0e",
    "GRD": "#d62728",
    "EVT": "#9467bd",
    "CMD": "#7f7f7f",
    "DBG": "#17becf",
}


@dataclass
class WioPlacement:
    edge: str  # "left" or "right"
    domain: int
    x: int
    y_start: int
    y_end: int
    shard: str
    port_group: str
    flow: str
    init_txrx: str
    channel: int | None


@dataclass
class WioReport:
    fabric_columns: int
    fabric_rows: int
    compute_core_origin: Tuple[int, int]
    compute_core_size: Tuple[int, int]
    lanes: int
    lane_width: int
    total_wios: int
    total_capacity: int
    left_wios: int
    right_wios: int
    placements: List[WioPlacement]
    flows: Dict[str, int]


def _parse_int(value: str) -> int | None:
    value = value.strip()
    return int(value) if value.isdigit() else None


def parse_wio_report(path: Path) -> WioReport:
    lines = path.read_text().splitlines()

    fabric_columns = fabric_rows = None
    compute_core_origin = compute_core_size = None
    lanes = lane_width = None
    flows: Dict[str, int] = {}
    total_wios = total_capacity = left_wios = right_wios = None
    placements: List[WioPlacement] = []

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()

        if stripped.startswith("Fabric size"):
            # Next line contains the size.
            size_line = lines[idx + 1]
            match = re.search(r"Size:\s*(\d+)\s+columns x\s+(\d+)\s+rows", size_line)
            if match:
                fabric_columns = int(match.group(1))
                fabric_rows = int(match.group(2))
            idx += 2
            continue

        if stripped.startswith("Compute core"):
            origin_line = lines[idx + 1]
            size_line = lines[idx + 2]
            lanes_line = lines[idx + 3]
            lane_width_line = lines[idx + 4]

            origin_match = re.search(r"Origin\s*:\s*\((\d+),\s*(\d+)\)", origin_line)
            size_match = re.search(r"Size\s*:\s*(\d+)\s+columns x\s+(\d+)\s+rows", size_line)
            lanes_match = re.search(r"Lanes\s*:\s*(\d+)", lanes_line)
            lane_width_match = re.search(r"Lane width:\s*(\d+)\s+columns", lane_width_line)

            if origin_match:
                compute_core_origin = (int(origin_match.group(1)), int(origin_match.group(2)))
            if size_match:
                compute_core_size = (int(size_match.group(1)), int(size_match.group(2)))
            if lanes_match:
                lanes = int(lanes_match.group(1))
            if lane_width_match:
                lane_width = int(lane_width_match.group(1))
            idx += 5
            continue

        if stripped.startswith("WIO utilization by flow"):
            idx += 1
            while idx < len(lines):
                flow_line = lines[idx].strip()
                if not flow_line or flow_line.startswith("WIO utilization"):
                    break
                match = re.search(r"Flow\s+(\w+):\s+(\d+)\s+WIOs", flow_line)
                if match:
                    flow_name = match.group(1).upper()
                    flows[flow_name] = int(match.group(2))
                idx += 1
            continue

        if stripped.startswith("WIO utilization:"):
            match = re.search(r"WIO utilization:\s*(\d+)\s*/\s*(\d+)", stripped)
            if match:
                total_wios = int(match.group(1))
                total_capacity = int(match.group(2))
            # Parse left/right totals in the following few lines.
            for j in range(idx + 1, min(len(lines), idx + 8)):
                if left_wios is None:
                    left_match = re.search(r"Left edge:\s*(\d+)\s*/\s*(\d+)", lines[j])
                    if left_match:
                        left_wios = int(left_match.group(1))
                if right_wios is None:
                    right_match = re.search(r"Right edge:\s*(\d+)\s*/\s*(\d+)", lines[j])
                    if right_match:
                        right_wios = int(right_match.group(1))
                if left_wios is not None and right_wios is not None:
                    break
            idx += 3
            continue

        if stripped.startswith("WIO placement and configuration"):
            placements.extend(_parse_placements(lines[idx + 1 :]))
            break

        idx += 1

    if None in (
        fabric_columns,
        fabric_rows,
        compute_core_origin,
        compute_core_size,
        lanes,
        lane_width,
        total_wios,
        total_capacity,
        left_wios,
        right_wios,
    ):
        raise ValueError(f"Incomplete report data parsed from {path}")

    return WioReport(
        fabric_columns=fabric_columns,
        fabric_rows=fabric_rows,
        compute_core_origin=compute_core_origin,  # type: ignore[arg-type]
        compute_core_size=compute_core_size,  # type: ignore[arg-type]
        lanes=lanes,  # type: ignore[arg-type]
        lane_width=lane_width,  # type: ignore[arg-type]
        total_wios=total_wios,  # type: ignore[arg-type]
        total_capacity=total_capacity,  # type: ignore[arg-type]
        left_wios=left_wios,  # type: ignore[arg-type]
        right_wios=right_wios,  # type: ignore[arg-type]
        placements=placements,
        flows=flows,
    )


def _parse_placements(lines: Iterable[str]) -> List[WioPlacement]:
    placements: List[WioPlacement] = []
    current_domains: Tuple[int, int] | None = None
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("Floorplan detail") or stripped.startswith("Floorplan:"):
            break
        if "Domain 3" in raw_line and "Domain 0" in raw_line:
            current_domains = (3, 0)
            continue
        if "Domain 2" in raw_line and "Domain 1" in raw_line:
            current_domains = (2, 1)
            continue
        if not raw_line or raw_line.startswith("-"):
            continue
        if "|" not in raw_line or current_domains is None:
            continue
        left_text, right_text = raw_line.split("|", 1)
        placements.extend(
            _parse_placement_segment(left_text, "left", current_domains[0])
        )
        placements.extend(
            _parse_placement_segment(right_text, "right", current_domains[1])
        )
    return placements


def _parse_placement_segment(text: str, edge: str, domain: int) -> List[WioPlacement]:
    text = text.strip()
    if not text or text.startswith("-"):
        return []
    tokens = re.split(r"\s{2,}", text)
    # Expect at least x, y, shard, port_group, core, init_txrx, channel
    if len(tokens) < 4:
        return []

    if tokens[0].startswith("-") or tokens[2].startswith("-"):
        return []

    x = _parse_int(tokens[0])
    y_token = tokens[1]
    shard = tokens[2]
    port_group = tokens[3] if len(tokens) > 3 else ""
    init_txrx = tokens[-2] if len(tokens) >= 2 else ""
    channel = _parse_int(tokens[-1])

    if x is None:
        return []

    y_start, y_end = _parse_y_span(y_token)
    flow = shard.split("-")[0].upper() if "-" in shard else shard.upper()

    placement = WioPlacement(
        edge=edge,
        domain=domain,
        x=x,
        y_start=y_start,
        y_end=y_end,
        shard=shard,
        port_group=port_group,
        flow=flow,
        init_txrx=init_txrx,
        channel=channel,
    )
    return [placement]


def _parse_y_span(token: str) -> Tuple[int, int]:
    parts = [p.strip() for p in token.replace(" ", "").split(",") if p.strip()]
    if not parts:
        return (0, 0)
    numbers = [int(p) for p in parts if p.isdigit()]
    if not numbers:
        return (0, 0)
    return (min(numbers), max(numbers))


def summarize_flows(report: WioReport) -> Dict[str, Any]:
    edge_flow_counts: Dict[str, Counter[str]] = {"left": Counter(), "right": Counter()}
    for placement in report.placements:
        edge_flow_counts[placement.edge][placement.flow] += 1
    return {
        "flows_total": report.flows,
        "flows_by_edge": {
            edge: dict(counter) for edge, counter in edge_flow_counts.items()
        },
        "total_wios": report.total_wios,
        "total_capacity": report.total_capacity,
        "left_wios": report.left_wios,
        "right_wios": report.right_wios,
    }


def render_wio_report(report: WioReport, output_path: Path, fmt: str = "png") -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for rendering. Install dependencies from requirements.txt."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Flow stacked bars by edge.
    flow_order = sorted(report.flows.keys())
    edge_flow_counts = defaultdict(Counter)  # type: ignore[var-annotated]
    for placement in report.placements:
        edge_flow_counts[placement.edge][placement.flow] += 1

    fig, (ax_bar, ax_scatter) = plt.subplots(
        nrows=2, figsize=(11, 10), gridspec_kw={"height_ratios": [1, 2.5]}
    )

    edges = ["left", "right"]
    y_positions = [0, 1]
    bottoms = [0, 0]
    for flow in flow_order:
        values = [edge_flow_counts[edge].get(flow, 0) for edge in edges]
        color = FLOW_COLORS.get(flow, "#1f77b4")
        ax_bar.barh(
            y_positions,
            values,
            left=bottoms,
            color=color,
            label=flow,
            edgecolor="black",
        )
        bottoms = [bottoms[i] + values[i] for i in range(len(values))]
    ax_bar.set_yticks(y_positions, labels=[f"{edge.capitalize()} edge" for edge in edges])
    ax_bar.set_xlabel("WIO count")
    ax_bar.set_title("WIO utilization by edge and flow")
    ax_bar.legend(loc="lower right", ncol=len(flow_order))

    # Scatter of placements by Y with domain offsets on X.
    domain_offsets = {
        ("left", 3): -1.5,
        ("left", 2): -1.0,
        ("right", 0): 1.0,
        ("right", 1): 1.5,
    }
    for placement in report.placements:
        x_pos = domain_offsets.get((placement.edge, placement.domain), 0 if placement.edge == "left" else 2)
        y_pos = (placement.y_start + placement.y_end) / 2.0
        height = placement.y_end - placement.y_start + 1
        color = FLOW_COLORS.get(placement.flow, "#1f77b4")
        ax_scatter.scatter(
            x_pos,
            y_pos,
            s=max(20, height * 0.5),
            color=color,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
        )

    ax_scatter.set_ylabel("Fabric row (Y)")
    ax_scatter.set_title("WIO placement by edge, domain, and flow")
    ax_scatter.set_ylim(0, report.fabric_rows)
    ax_scatter.set_xlim(-2, 2)
    ax_scatter.set_xticks([-1.5, -1.0, 1.0, 1.5])
    ax_scatter.set_xticklabels(["Domain 3", "Domain 2", "Domain 0", "Domain 1"])
    ax_scatter.grid(axis="y", linestyle="--", alpha=0.4)

    # Summary text box.
    summary = (
        f"Total: {report.total_wios}/{report.total_capacity} | "
        f"Left: {report.left_wios} | Right: {report.right_wios}"
    )
    ax_bar.text(
        0.01,
        1.05,
        summary,
        transform=ax_bar.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#888"),
    )

    fig.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize WIO report files.")
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        default=None,
        help="One or more report files (default: analyze/wio_report_*.txt).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file (single input) or directory (multiple inputs). Defaults to analyze/output/.",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["png", "svg"],
        default="png",
        help="Image format for output files.",
    )
    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Print parsed utilization summary to stdout.",
    )

    args = parser.parse_args()

    if plt is None:
        raise SystemExit(
            "matplotlib is required. Install project dependencies (e.g., `uv pip install -r requirements.txt`)."
        )

    input_paths: List[Path]
    if args.input is None:
        input_paths = sorted(Path("analyze").glob("wio_report_*.txt"))
    else:
        input_paths = [Path(p) for p in args.input]

    if not input_paths:
        raise SystemExit("No input reports found.")

    output_arg = Path(args.output) if args.output else Path("analyze/output")
    multiple_inputs = len(input_paths) > 1

    for input_path in input_paths:
        if not input_path.exists():
            raise SystemExit(f"Input report not found: {input_path}")
        report = parse_wio_report(input_path)
        summary = summarize_flows(report)
        if args.show_summary:
            print(f"== Summary for {input_path.name} ==")
            print(
                f"Total {summary['total_wios']}/{summary['total_capacity']} "
                f"(Left {summary['left_wios']}, Right {summary['right_wios']})"
            )
            print("Flows (declared):", summary["flows_total"])
            print("Flows by edge:", summary["flows_by_edge"])

        if multiple_inputs or output_arg.is_dir():
            output_dir = output_arg
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"{input_path.stem}.{args.format}"
        else:
            output_arg.parent.mkdir(parents=True, exist_ok=True)
            out_path = output_arg
        render_wio_report(report, out_path, fmt=args.format)


if __name__ == "__main__":
    main()
