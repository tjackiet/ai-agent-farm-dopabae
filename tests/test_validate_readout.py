"""確かめ直しの測定。別の画像の組・順序・検定・型の選びかたが、決まった値を返すこと。

網もシミュレーションも使わない。検定は numpy が無ければ飛ばす（requirements-connectome.txt）。
"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

from tests import helpers

try:
    import numpy as np
except ImportError:  # pragma: no cover - 依存が無い環境では飛ばす
    np = None

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import probe_readout as probe  # noqa: E402
import validate_readout as validate  # noqa: E402

sys.path.pop(0)


class ImageSetTest(unittest.TestCase):
    def setUp(self):
        self.cfg = helpers.load_config()

    def test_differs_from_the_discovery_set(self):
        """選んだ画像の組と同じ画像を使わない。"""
        self.assertFalse(set(validate.CHARTS) & set(probe.CHARTS))
        self.assertFalse(set(validate.CHARTS.values()) & set(probe.CHARTS.values()))
        found = {v[0].sha256 for k, v in probe.images(self.cfg).items() if k != probe.BLANK}
        mine = {v[0].sha256 for k, v in validate.images(self.cfg).items() if k != validate.BLANK}
        self.assertFalse(found & mine)

    def test_pairs_are_mirrored_moves(self):
        """1 組の 2 枚は、足の大きさが同じで向きだけが逆。"""
        for up, down in validate.PAIRS:
            self.assertEqual(validate.CHARTS[down], tuple(-m for m in validate.CHARTS[up]), (up, down))
        self.assertEqual({n for pair in validate.PAIRS for n in pair}, set(validate.CHARTS))

    def test_all_charts_end_at_the_same_close(self):
        for name, moves in validate.CHARTS.items():
            candles = validate.chart_candles(moves, 0)
            self.assertEqual(len(candles), self.cfg.vision_lookback_candles, name)
            self.assertEqual(candles[-1].close, validate.END_CLOSE, name)

    def test_text_rows_are_identical(self):
        """文字（ペア名・気配の数字）の行は 6 種で同じ。文字の差が画像の差に混ざらない。"""
        width = self.cfg.vision_width
        rows = self.cfg.vision_margin_px + self.cfg.vision_text_rows_px
        tops = {img[0].pixels[: rows * width * 3] for k, img in validate.images(self.cfg).items()
                if k != validate.BLANK}
        self.assertEqual(len(tops), 1)

    def test_images_are_distinct_and_deterministic(self):
        first = validate.images(self.cfg)
        second = validate.images(self.cfg)
        self.assertEqual({k: [f.sha256 for f in v] for k, v in first.items()},
                         {k: [f.sha256 for f in v] for k, v in second.items()})
        self.assertEqual(len({v[0].sha256 for v in first.values()}), len(first))

    def test_motion_keeps_the_text_still(self):
        """動きの枚でも、文字の行は 6 種・全枚で同じ。"""
        cfg = dataclasses.replace(self.cfg, vision_motion="grow_last", vision_motion_frames=10)
        width = cfg.vision_width
        rows = cfg.vision_margin_px + cfg.vision_text_rows_px
        tops = {f.pixels[: rows * width * 3] for k, frames in validate.images(cfg).items()
                if k != validate.BLANK for f in frames}
        self.assertEqual(len(tops), 1)


class SequenceTest(unittest.TestCase):
    def test_each_cycle_shows_every_chart_once(self):
        seq = validate.sequence(4)
        self.assertEqual(seq[: validate.WARMUP], [validate.BLANK] * validate.WARMUP)
        body = seq[validate.WARMUP:]
        n = len(validate.CHARTS)
        for k in range(4):
            self.assertEqual(sorted(body[k * n:(k + 1) * n]), sorted(validate.CHARTS))

    def test_order_changes_between_cycles_and_is_seeded(self):
        """順序が固定だと、直前に見た画像の差が画像の差に混ざる。"""
        seq = validate.sequence(8)
        n = len(validate.CHARTS)
        cycles = {tuple(seq[validate.WARMUP + k * n: validate.WARMUP + (k + 1) * n]) for k in range(8)}
        self.assertGreater(len(cycles), 1)
        self.assertEqual(seq, validate.sequence(8))
        self.assertNotEqual(seq, validate.sequence(8, seed=1))


class HolmTest(unittest.TestCase):
    def test_stops_at_the_first_failure(self):
        # 0.01 ≦ 0.05/3、0.02 ≦ 0.05/2、0.04 ≦ 0.05/1
        self.assertEqual(validate.holm([0.04, 0.01, 0.02]), [True, True, True])
        # 0.03 > 0.05/2 なので、それより大きい 0.04 も棄却しない
        self.assertEqual(validate.holm([0.04, 0.01, 0.03]), [False, True, False])
        self.assertEqual(validate.holm([0.5, 0.6]), [False, False])


class PreregisteredTest(unittest.TestCase):
    def test_selects_by_discovery_p_only(self):
        discovery = {"screen": {"p_le_0.01": [
            {"type": "A", "superclass": "x", "p": 0.0005},
            {"type": "B", "superclass": "x", "p": 0.001},
            {"type": "C", "superclass": "x", "p": 0.005},
        ]}}
        self.assertEqual([t["type"] for t in validate.preregistered(discovery, 0.001)], ["A", "B"])

    def test_refuses_an_old_discovery_file(self):
        """p ≦ 0.01 の一覧が無い JSON（上位 15 型だけ）からは選ばない。"""
        with self.assertRaises(SystemExit):
            validate.preregistered({"screen": {"top": []}}, 0.001)


@unittest.skipIf(np is None, "numpy が無い")
class TestsOnPlantedData(unittest.TestCase):
    def setUp(self):
        # 6 種 × 8 巡。札は 0..5、組は (0, 1), (2, 3), (4, 5)。
        self.labels = np.array(list(range(6)) * 8)
        self.pairs = [(0, 1), (2, 3), (4, 5)]
        rng = np.random.default_rng(1)
        noise = rng.normal(0, 1, size=(3, len(self.labels)))
        up = np.isin(self.labels, [0, 2, 4])
        self.values = np.vstack([
            noise[0] + np.where(up, 3.0, 0.0),        # 上げで高い
            noise[1] + (self.labels == 5) * 4.0,      # 1 枚だけ高い（上げ下げとは関係ない）
            noise[2],                                  # 差なし
        ])

    def test_image_test(self):
        p = validate.image_test(self.values, self.labels, 6, 2000, 7)
        self.assertLess(p[0], 0.01)
        self.assertLess(p[1], 0.01)
        self.assertGreater(p[2], 0.05)

    def test_direction_test(self):
        diff, p = validate.direction_test(self.values, self.labels, self.pairs, 2000, 7)
        self.assertGreater(diff[0], 2.0)
        self.assertLess(p[0], 0.01)
        self.assertGreater(p[2], 0.05)

    def test_direction_test_is_deterministic(self):
        a = validate.direction_test(self.values, self.labels, self.pairs, 500, 7)
        b = validate.direction_test(self.values, self.labels, self.pairs, 500, 7)
        self.assertTrue((a[0] == b[0]).all() and (a[1] == b[1]).all())


if __name__ == "__main__":
    unittest.main()
