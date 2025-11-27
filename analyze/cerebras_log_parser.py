"""
Parser for detailed Cerebras WSE log text.

Transforms the raw log into structured Pandas DataFrames for geometry,
floorplan modules, WIO-to-core connectivity, shard-to-domain mappings,
and buffer layouts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class ParsedLog:
    geometry: Dict[str, int]
    floorplan: pd.DataFrame
    connectivity: pd.DataFrame
    domain_mapping: pd.DataFrame
    buffers: pd.DataFrame


class CerebrasLogParser:
    """Parse Cerebras wafer-scale engine logs into structured data."""

    def __init__(self, raw_text: str) -> None:
        self.raw_text = raw_text
        self.lines = raw_text.splitlines()

    def parse(self) -> ParsedLog:
        geometry = self.parse_geometry()
        floorplan = self.parse_floorplan()
        connectivity = self.parse_connectivity()
        domain_mapping = self.parse_domain_mapping()
        buffers = self.parse_buffers()
        return ParsedLog(
            geometry=geometry,
            floorplan=floorplan,
            connectivity=connectivity,
            domain_mapping=domain_mapping,
            buffers=buffers,
        )

    def parse_geometry(self) -> Dict[str, int]:
        fabric_columns = fabric_rows = None
        core_origin_x = core_origin_y = None
        core_width = core_height = None
        lanes = lane_width = None
        buffer_columns = buffer_rows_below_core = None

        for idx, line in enumerate(self.lines):
            stripped = line.strip()
            if stripped.startswith("Fabric size"):
                if idx + 1 < len(self.lines):
                    size_line = self.lines[idx + 1]
                    match = re.search(
                        r"Size:\s*(\d+)\s+columns x\s+(\d+)\s+rows", size_line
                    )
                    if match:
                        fabric_columns = int(match.group(1))
                        fabric_rows = int(match.group(2))
            if stripped.startswith("Compute core"):
                if idx + 4 < len(self.lines):
                    origin_line = self.lines[idx + 1]
                    size_line = self.lines[idx + 2]
                    lanes_line = self.lines[idx + 3]
                    lane_width_line = self.lines[idx + 4]
                    origin_match = re.search(
                        r"Origin\s*:\s*\((\d+),\s*(\d+)\)", origin_line
                    )
                    size_match = re.search(
                        r"Size\s*:\s*(\d+)\s+columns x\s+(\d+)\s+rows", size_line
                    )
                    lanes_match = re.search(r"Lanes\s*:\s*(\d+)", lanes_line)
                    lane_width_match = re.search(
                        r"Lane width:\s*(\d+)\s+columns", lane_width_line
                    )
                    if origin_match:
                        core_origin_x = int(origin_match.group(1))
                        core_origin_y = int(origin_match.group(2))
                    if size_match:
                        core_width = int(size_match.group(1))
                        core_height = int(size_match.group(2))
                    if lanes_match:
                        lanes = int(lanes_match.group(1))
                    if lane_width_match:
                        lane_width = int(lane_width_match.group(1))
            if (
                fabric_columns is not None
                and fabric_rows is not None
                and core_origin_x is not None
                and core_origin_y is not None
                and core_width is not None
                and core_height is not None
                and lanes is not None
                and lane_width is not None
                and buffer_columns is not None
                and buffer_rows_below_core is not None
            ):
                break
            if stripped.startswith("Buffer columns"):
                match = re.search(r"Buffer columns:\s*(\d+)", stripped)
                if match:
                    buffer_columns = int(match.group(1))
            if stripped.startswith("Buffer rows (below core)"):
                match = re.search(r"Buffer rows \(below core\):\s*(\d+)", stripped)
                if match:
                    buffer_rows_below_core = int(match.group(1))

        missing = [
            name
            for name, value in {
                "fabric_columns": fabric_columns,
                "fabric_rows": fabric_rows,
                "core_origin_x": core_origin_x,
                "core_origin_y": core_origin_y,
                "core_width": core_width,
                "core_height": core_height,
                "lanes": lanes,
                "lane_width": lane_width,
                "buffer_columns": buffer_columns,
                "buffer_rows_below_core": buffer_rows_below_core,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(f"Missing geometry fields: {', '.join(missing)}")

        return {
            "fabric_columns": int(fabric_columns),
            "fabric_rows": int(fabric_rows),
            "core_origin_x": int(core_origin_x),
            "core_origin_y": int(core_origin_y),
            "core_width": int(core_width),
            "core_height": int(core_height),
            "lanes": int(lanes),
            "lane_width": int(lane_width),
            "buffer_columns": int(buffer_columns),
            "buffer_rows_below_core": int(buffer_rows_below_core),
        }

    def parse_floorplan(self) -> pd.DataFrame:
        records: List[Dict[str, object]] = []
        lines = self.lines
        for idx, line in enumerate(lines):
            if "LEFT edge modules" in line:
                left_start = idx + 2  # Skip header and underline
                left_records, last_idx = self._parse_module_block(left_start, "LEFT")
                records.extend(left_records)
                right_header = None
                for probe_idx in range(last_idx, len(lines)):
                    if "RIGHT edge modules" in lines[probe_idx]:
                        right_header = probe_idx
                        break
                if right_header:
                    right_start = right_header + 2
                    right_records, _ = self._parse_module_block(
                        right_start, "RIGHT"
                    )
                    records.extend(right_records)
                break
        return pd.DataFrame(records)

    def _parse_module_block(
        self, start_idx: int, edge: str
    ) -> tuple[List[Dict[str, object]], int]:
        records: List[Dict[str, object]] = []
        idx = start_idx
        while idx < len(self.lines):
            line = self.lines[idx].strip()
            if not line or line.startswith("=") or line.startswith("Core 0"):
                break
            if "edge modules" in line and line.upper().startswith("RIGHT") != (edge == "RIGHT"):
                break
            if "edge modules" in line and line.upper().startswith("LEFT") != (edge == "LEFT"):
                break
            match = re.match(
                r"(?P<name>[A-Za-z0-9_]+)\s+(?P<range>\d+(?:-\d+)?)\s+(?P<width>\d+)",
                line,
            )
            if match:
                col_range = match.group("range")
                if "-" in col_range:
                    start, end = col_range.split("-")
                    start_col = int(start)
                    end_col = int(end)
                else:
                    start_col = end_col = int(col_range)
                records.append(
                    {
                        "module_name": match.group("name"),
                        "start_col": start_col,
                        "end_col": end_col,
                        "width": int(match.group("width")),
                        "edge": edge,
                    }
                )
            idx += 1
        return records, idx

    def parse_connectivity(self) -> pd.DataFrame:
        records: List[Dict[str, object]] = []
        current_meta: Optional[Dict[str, object]] = None
        for line in self.lines:
            stripped = line.strip()
            header_match = re.match(
                r"Core\s+0,\s*(?P<flow>[A-Z]+)\s+port group\s+(?P<port>\d+),\s*domain\s+(?P<domain>\d+),\s*(?P<edge>LEFT|RIGHT)\s+edge",
                stripped,
                flags=re.IGNORECASE,
            )
            if header_match:
                current_meta = {
                    "flow_type": header_match.group("flow").upper(),
                    "port_group": int(header_match.group("port")),
                    "domain": int(header_match.group("domain")),
                    "edge": header_match.group("edge").upper(),
                }
                continue
            if stripped.startswith("WIO[") and current_meta:
                wio_match = re.match(
                    r"WIO\[(?P<wio_idx>\d+)] at y=(?P<y>\d+)\s+connected to\s+\d+\s+core rows.*\[(?P<rows>[0-9,\s]+)\]",
                    stripped,
                )
                if not wio_match:
                    continue
                wio_idx = int(wio_match.group("wio_idx"))
                wio_y = int(wio_match.group("y"))
                row_values = [
                    int(item.strip())
                    for item in wio_match.group("rows").split(",")
                    if item.strip().isdigit()
                ]
                for core_row in row_values:
                    record = {
                        **current_meta,
                        "wio_index": wio_idx,
                        "wio_y": wio_y,
                        "core_row_abs": core_row,
                    }
                    records.append(record)
        return pd.DataFrame(records)

    def parse_domain_mapping(self) -> pd.DataFrame:
        records: List[Dict[str, object]] = []
        shard_type: Optional[str] = None
        start_idx = None
        for idx, line in enumerate(self.lines):
            if "Connectivity between shards and FPGA domains" in line:
                start_idx = idx + 1
                break
        if start_idx is None:
            return pd.DataFrame(records)

        for line in self.lines[start_idx:]:
            stripped = line.strip()
            if not stripped:
                continue
            type_match = re.match(r"Shard type:\s*(\w+)", stripped)
            if type_match:
                shard_type = type_match.group(1).upper()
                continue
            shard_match = re.match(
                r"Shard\s+(?P<id>\d+)\s+<->\s+Domain\s+(?P<domain>\d+)", stripped
            )
            if shard_match and shard_type:
                records.append(
                    {
                        "shard_type": shard_type,
                        "shard_id": int(shard_match.group("id")),
                        "domain": int(shard_match.group("domain")),
                    }
                )
        return pd.DataFrame(records)

    def parse_buffers(self) -> pd.DataFrame:
        records: List[Dict[str, object]] = []
        in_section = False
        current_kind: Optional[str] = None
        line_index = 0
        sequence = 0
        while line_index < len(self.lines):
            line = self.lines[line_index]
            stripped = line.strip()
            if stripped.startswith("I/O kernels and buffers"):
                in_section = True
                line_index += 1
                continue
            if in_section:
                if stripped.startswith("Connectivity between WIO"):
                    break
                kind_match = re.match(r"([A-Za-z0-9 /-]+buffers):\s*(.*)", stripped)
                if kind_match:
                    current_kind = kind_match.group(1).strip()
                    details = kind_match.group(2)
                    sequence += 1
                    # Inline-only buffer descriptors (e.g., below-core)
                    if details and ("below-core" in current_kind or "above-core" in current_kind):
                        cols, rows, count, y_offset = self._parse_buffer_dims(details)
                        flow_type = current_kind.split()[0].upper()
                        location = (
                            "below-core"
                            if "below-core" in current_kind
                            else "above-core"
                        )
                        records.append(
                            {
                                "buffer_kind": current_kind,
                                "flow_type": flow_type,
                                "location": location,
                                "columns": cols,
                                "rows": rows,
                                "buffers": count,
                                "y_offset": y_offset,
                                "sequence": sequence,
                            }
                        )
                    line_index += 1
                    continue
                edge_match = re.match(
                    r"(Left edge|Right edge|below core|above core):\s*(.*)",
                    stripped,
                    flags=re.IGNORECASE,
                )
                if edge_match and current_kind:
                    location_raw = edge_match.group(1).lower()
                    if "left" in location_raw:
                        location = "left"
                    elif "right" in location_raw:
                        location = "right"
                    elif "below" in location_raw:
                        location = "below-core"
                    else:
                        location = "above-core"
                    details = edge_match.group(2)
                    cols, rows, count, y_offset = self._parse_buffer_dims(details)
                    flow_type = current_kind.split()[0].upper()
                    records.append(
                        {
                            "buffer_kind": current_kind,
                            "flow_type": flow_type,
                            "location": location,
                            "columns": cols,
                            "rows": rows,
                            "buffers": count,
                            "y_offset": y_offset,
                            "sequence": sequence,
                        }
                    )
            line_index += 1
        return pd.DataFrame(records)

    def _parse_buffer_dims(
        self, text: str
    ) -> tuple[int, int, Optional[int], Optional[int]]:
        cols = rows = 0
        count: Optional[int] = None
        y_offset: Optional[int] = None
        size_match = re.search(r"(\d+)\s+columns\s*x\s*(\d+)\s+rows", text)
        if size_match:
            cols = int(size_match.group(1))
            rows = int(size_match.group(2))
        count_match = re.search(r"x\s*(\d+)\s+buffers", text)
        if count_match:
            count = int(count_match.group(1))
        offset_match = re.search(r"y_offset\s+is\s+(\d+)", text)
        if offset_match:
            y_offset = int(offset_match.group(1))
        return cols, rows, count, y_offset
