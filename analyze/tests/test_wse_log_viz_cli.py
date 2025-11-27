import tempfile
from pathlib import Path
from unittest import TestCase

from analyze import wse_log_viz
from analyze.sample_wse_log import SAMPLE_WSE_LOG


class WSELogVizCLITests(TestCase):
    def test_render_figures_produces_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            generated = wse_log_viz.render_figures(
                SAMPLE_WSE_LOG,
                out_dir,
                fmt="png",
                plots=["connectivity", "floorplan", "summary"],
                summary_report=Path("analyze/wio_report_1.txt"),
            )
            expected_files = {
                out_dir / "connectivity.png",
                out_dir / "floorplan.png",
                out_dir / "summary.png",
            }
            self.assertTrue(expected_files.issubset(set(generated)))
            for path in expected_files:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)
