import tempfile
from pathlib import Path
from unittest import TestCase

from analyze import wio_visualizer


class WioVisualizerTests(TestCase):
    def setUp(self) -> None:
        self.sample_path = Path(__file__).resolve().parent.parent / "wio_report_1.txt"
        if not self.sample_path.exists():
            self.skipTest("Sample report not available for parsing tests.")

    def test_parse_wio_report_sample1(self) -> None:
        report = wio_visualizer.parse_wio_report(self.sample_path)
        self.assertEqual(report.total_wios, 123)
        self.assertEqual(report.total_capacity, 124)
        self.assertEqual(report.left_wios, 62)
        self.assertEqual(report.right_wios, 61)
        self.assertEqual(report.buffer_columns, 14)
        self.assertEqual(report.buffer_rows_below_core, 1)
        self.assertIn("ACT", report.flows)
        self.assertEqual(report.flows["ACT"], 101)
        self.assertGreater(len(report.placements), 0)

    def test_render_creates_image(self) -> None:
        if wio_visualizer.plt is None:
            self.skipTest("matplotlib not available in environment")
        report = wio_visualizer.parse_wio_report(self.sample_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "wio.png"
            wio_visualizer.render_wio_report(report, out_path, fmt="png")
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)
