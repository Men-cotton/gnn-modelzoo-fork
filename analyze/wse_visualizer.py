"""
Matplotlib visualizations for Cerebras WSE logs.

Renders connectivity density, periphery floorplan, shard-domain heatmaps,
and buffer layouts from the parsed log data.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
import pandas as pd  # noqa: E402  pylint: disable=wrong-import-position
from matplotlib.collections import LineCollection  # noqa: E402  pylint: disable=wrong-import-position
from matplotlib.patches import Rectangle  # noqa: E402  pylint: disable=wrong-import-position

from analyze.cerebras_log_parser import ParsedLog

FLOW_COLORS: Dict[str, str] = {
    "ACT": "#2ca02c",
    "WGT": "#ff7f0e",
    "GRD": "#d62728",
    "EVT": "#9467bd",
}


class WSEVisualizer:
    def __init__(self, colors: Dict[str, str] | None = None) -> None:
        self.colors = colors or FLOW_COLORS
        self.default_color = "#1f77b4"

    def draw_connectivity(self, parsed: ParsedLog):
        df = parsed.connectivity
        geometry = parsed.geometry
        fig, ax = plt.subplots(figsize=(10, 12))
        if df.empty:
            ax.text(0.5, 0.5, "No connectivity data", ha="center", va="center")
            return fig

        left_x = -50
        right_x = geometry["fabric_columns"] + 50
        core_start_x = geometry["core_origin_x"]
        core_end_x = geometry["core_origin_x"] + geometry["core_width"]

        segments: List[List[tuple[float, float]]] = []
        colors: List[str] = []
        for row in df.itertuples():
            x0 = left_x if row.edge == "LEFT" else right_x
            x1 = core_start_x if row.edge == "LEFT" else core_end_x
            segments.append([(x0, row.wio_y), (x1, row.core_row_abs)])
            colors.append(self.colors.get(row.flow_type, self.default_color))

        line_collection = LineCollection(
            segments, colors=colors, alpha=0.3, linewidths=0.7
        )
        ax.add_collection(line_collection)

        wio_points = (
            df.groupby(["edge", "wio_y", "flow_type"])
            .size()
            .reset_index(name="fan_out")
        )
        for _, point in wio_points.iterrows():
            x = left_x if point.edge == "LEFT" else right_x
            size = 26 if point.fan_out > 1 else 16
            ax.scatter(
                x,
                point.wio_y,
                s=size,
                color=self.colors.get(point.flow_type, self.default_color),
                edgecolors="black",
                linewidths=0.4,
                alpha=0.9,
                zorder=3,
            )

        ax.add_patch(
            Rectangle(
                (core_start_x, geometry["core_origin_y"]),
                geometry["core_width"],
                geometry["core_height"],
                facecolor="none",
                edgecolor="#2c7fb8",
                linestyle="--",
                linewidth=1.4,
                label="Compute core",
            )
        )
        ax.set_xlim(left_x - 20, right_x + 20)
        ax.set_ylim(0, geometry["fabric_rows"])
        ax.set_xlabel("Logical X (Left WIO -> Core -> Right WIO)")
        ax.set_ylabel("Fabric row (Y)")
        ax.set_title("WIO to core connectivity density")
        legend_handles = [
            plt.Line2D(
                [0], [0], color=self.colors.get(flow, self.default_color), lw=3, label=flow
            )
            for flow in sorted(df.flow_type.unique())
        ]
        ax.legend(handles=legend_handles, title="Flow", loc="upper right")
        ax.grid(alpha=0.25, linestyle="--")
        return fig

    def draw_floorplan(self, parsed: ParsedLog):
        df = parsed.floorplan
        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, sharey=True, figsize=(12, 6), gridspec_kw={"wspace": 0.08}
        )
        if df.empty:
            ax_left.text(0.5, 0.5, "No floorplan data", ha="center", va="center")
            ax_right.axis("off")
            return fig

        for ax, edge, color in [(ax_left, "LEFT", "#a6cee3"), (ax_right, "RIGHT", "#b2df8a")]:
            subset = df[df["edge"] == edge].reset_index(drop=True)
            for idx, row in subset.iterrows():
                y = idx
                width = row.end_col - row.start_col + 1
                ax.add_patch(
                    Rectangle(
                        (row.start_col, y - 0.4),
                        width,
                        0.8,
                        facecolor=color,
                        edgecolor="#333",
                        linewidth=0.6,
                        alpha=0.8,
                    )
                )
                label = row.module_name
                if label in {"CSG_BUF", "PAD"}:
                    ax.text(
                        row.start_col + width / 2,
                        y,
                        f"{label}\n{width} cols",
                        ha="center",
                        va="center",
                        fontsize=8,
                        weight="bold",
                    )
                else:
                    ax.text(
                        row.start_col + width / 2,
                        y,
                        label,
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
            ax.set_title(f"{edge.title()} periphery")
            if edge == "LEFT":
                ax.set_xlim(-2, 32)
                ax.set_xlabel("Column (left edge focus)")
            else:
                max_col = df.end_col.max() + 2
                ax.set_xlim(max_col - 32, max_col + 2)
                ax.set_xlabel("Column (right edge focus)")
            ax.set_yticks(range(len(subset)))
            ax.set_yticklabels([row.module_name for _, row in subset.iterrows()], fontsize=7)
            ax.grid(axis="x", alpha=0.3, linestyle="--")
        fig.suptitle("Periphery floorplan modules (broken x-axis view)")
        return fig

    def draw_domain_heatmap(self, parsed: ParsedLog):
        df = parsed.domain_mapping
        shard_types = sorted(df.shard_type.unique()) if not df.empty else []
        cols_per_row = 1 if len(shard_types) <= 2 else 2
        rows = max(1, (len(shard_types) + cols_per_row - 1) // cols_per_row)
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(6 * cols_per_row, 4 * rows))
        if not isinstance(axes, Iterable):
            axes = [axes]
        axes_list: List = list(axes) if isinstance(axes, list) else list(axes.flat)
        if df.empty:
            axes_list[0].text(0.5, 0.5, "No domain mapping data", ha="center", va="center")
            return fig

        domains_order = [3, 2, 0, 1]
        for idx, shard_type in enumerate(shard_types):
            ax = axes_list[idx]
            subset = df[df["shard_type"] == shard_type]
            max_shard = subset.shard_id.max()
            matrix = pd.DataFrame(0, index=range(max_shard + 1), columns=domains_order)
            for _, row in subset.iterrows():
                matrix.at[row.shard_id, row.domain] = 1
            im = ax.imshow(matrix.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
            for i in range(matrix.shape[0]):
                for j, domain in enumerate(domains_order):
                    value = matrix.iloc[i, j]
                    ax.text(
                        j,
                        i,
                        f"{value}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                    )
            ax.set_xticks(range(len(domains_order)), labels=[str(d) for d in domains_order])
            ax.set_yticks(range(matrix.shape[0]), labels=[str(i) for i in range(matrix.shape[0])])
            ax.set_xlabel("FPGA domain (left:2/3, right:0/1)")
            ax.set_ylabel("Shard id")
            ax.set_title(f"{shard_type} shard-domain mapping")
            ax.axvline(1.5, color="#666", linestyle="--", linewidth=0.8, alpha=0.5)
        for j in range(len(shard_types), len(axes_list)):
            axes_list[j].axis("off")
        fig.suptitle("Shard to FPGA domain mapping")
        fig.colorbar(im, ax=axes_list, shrink=0.6, label="Mapped")
        return fig

    def draw_buffer_layout(self, parsed: ParsedLog):
        df = parsed.buffers
        fig, ax = plt.subplots(figsize=(10, 8))
        if df.empty:
            ax.text(0.5, 0.5, "No buffer data", ha="center", va="center")
            return fig
        ax.axis("off")

        def placement_label(loc: str) -> str:
            if loc == "left":
                return "Left edge (west)"
            if loc == "right":
                return "Right edge (east)"
            if loc == "below-core":
                return "Below core (south)"
            if loc == "above-core":
                return "Above core (north)"
            return loc

        cleaned_kind = df["buffer_kind"].str.replace(
            r"\s*buffers", "", flags=re.IGNORECASE, regex=True
        ).str.strip()
        table_df = (
            df.sort_values(["location", "sequence"])
            .assign(
                buffer=cleaned_kind,
                columns=df["columns"].fillna(0).astype(int),
                rows=df["rows"].fillna(0).astype(int),
                buffers=df["buffers"].fillna(0).astype(int),
                y_offset=df["y_offset"].fillna(0).astype(int),
                placement=df["location"].apply(placement_label),
            )[
                [
                    "buffer",
                    "flow_type",
                    "placement",
                    "columns",
                    "rows",
                    "buffers",
                    "y_offset",
                ]
            ]
        )
        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        ax.set_title("Buffer placement summary (edges, no schematic)")
        return fig
