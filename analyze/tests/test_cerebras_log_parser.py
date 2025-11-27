from pathlib import Path
from unittest import TestCase

import pandas as pd

from analyze.cerebras_log_parser import CerebrasLogParser


class CerebrasLogParserTests(TestCase):
    def setUp(self) -> None:
        fixture_path = Path(__file__).resolve().parent.parent / "wio_report_1.txt"
        self.parser = CerebrasLogParser(fixture_path.read_text())
        self.parsed = self.parser.parse()

    def test_geometry_fields(self) -> None:
        geom = self.parsed.geometry
        self.assertEqual(geom["fabric_columns"], 762)
        self.assertEqual(geom["fabric_rows"], 1172)
        self.assertEqual(geom["core_origin_x"], 29)
        self.assertEqual(geom["core_origin_y"], 1)
        self.assertEqual(geom["core_width"], 704)
        self.assertEqual(geom["core_height"], 800)
        self.assertEqual(geom["lanes"], 32)
        self.assertEqual(geom["lane_width"], 22)
        self.assertEqual(geom["buffer_columns"], 14)
        self.assertEqual(geom["buffer_rows_below_core"], 1)

    def test_connectivity_expands_rows(self) -> None:
        df = self.parsed.connectivity
        self.assertFalse(df.empty)
        act_rows = df[
            (df["edge"] == "LEFT")
            & (df["flow_type"] == "ACT")
            & (df["wio_y"] == 36)
        ].sort_values("core_row_abs")
        self.assertListEqual(
            act_rows["core_row_abs"].tolist(),
            [37, 39, 41, 43, 45, 47, 49],
        )
        fan_out = (
            df.groupby(["edge", "wio_y", "flow_type"])
            .size()
            .reset_index(name="targets")
        )
        self.assertGreater(fan_out[fan_out["targets"] > 1].shape[0], 0)

    def test_floorplan_modules(self) -> None:
        floorplan = self.parsed.floorplan
        self.assertFalse(floorplan.empty)
        left_csg = floorplan[
            (floorplan["module_name"] == "CSG_BUF") & (floorplan["edge"] == "LEFT")
        ]
        self.assertEqual(int(left_csg.iloc[0]["start_col"]), 8)
        self.assertEqual(int(left_csg.iloc[0]["end_col"]), 24)
        self.assertEqual(int(left_csg.iloc[0]["width"]), 17)
        pad = floorplan[(floorplan["module_name"] == "PAD") & (floorplan["edge"] == "RIGHT")]
        self.assertFalse(pad.empty)

    def test_domain_mapping_counts(self) -> None:
        mapping = self.parsed.domain_mapping
        self.assertFalse(mapping.empty)
        counts = mapping.groupby("shard_type")["shard_id"].nunique().to_dict()
        self.assertEqual(counts.get("WGT"), 21)
        self.assertEqual(counts.get("ACT"), 21)
        first_wgt = mapping[(mapping["shard_type"] == "WGT") & (mapping["shard_id"] == 0)]
        self.assertEqual(int(first_wgt.iloc[0]["domain"]), 3)

    def test_buffer_offsets(self) -> None:
        buffers = self.parsed.buffers
        self.assertFalse(buffers.empty)
        act_sink_left = buffers[
            (buffers["buffer_kind"].str.contains("ACT sink mem buffers"))
            & (buffers["location"] == "left")
        ]
        self.assertEqual(int(act_sink_left.iloc[0]["columns"]), 5)
        self.assertEqual(int(act_sink_left.iloc[0]["rows"]), 1)
        self.assertEqual(int(act_sink_left.iloc[0]["buffers"]), 50)
        self.assertEqual(int(act_sink_left.iloc[0]["y_offset"]), 17)

        wgt_left = buffers[
            (buffers["buffer_kind"].str.startswith("WGT buffers"))
            & (buffers["location"] == "left")
        ]
        self.assertEqual(int(wgt_left.iloc[0]["columns"]), 5)
        self.assertEqual(int(wgt_left.iloc[0]["rows"]), 35)
