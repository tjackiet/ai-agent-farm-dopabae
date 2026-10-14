"""学習の測定の台本。見せる順番と、測る細胞群が決まっていること。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
try:
    import probe_learning as learning  # noqa: E402
except ImportError:  # pragma: no cover - numpy などが無い環境では飛ばす
    learning = None
sys.path.pop(0)


@unittest.skipIf(learning is None, "依存が無い")
class SequenceTest(unittest.TestCase):
    def test_warms_up_with_blank_then_cycles_charts(self):
        pictures = {learning.probe.BLANK: (), "上昇": (), "下落": ()}
        order = learning.sequence(pictures, 6)
        self.assertEqual(order[: learning.probe.WARMUP], [learning.probe.BLANK] * learning.probe.WARMUP)
        self.assertEqual(order[learning.probe.WARMUP:], ["上昇", "下落", "上昇", "下落"])

    def test_records_the_learning_cells(self):
        self.assertEqual(set(learning.RECORD) >= {"MBON07", "MBON11", "PAM11", "PPL101"}, True)


if __name__ == "__main__":
    unittest.main()
