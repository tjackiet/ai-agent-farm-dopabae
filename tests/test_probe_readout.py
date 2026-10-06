"""読み出しの候補の測定。検定と合成の画像が、決まった値を返すこと。

網もシミュレーションも使わない（numpy も要らない）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from tests import helpers

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import probe_readout as probe  # noqa: E402

sys.path.pop(0)


class FStatisticTest(unittest.TestCase):
    def test_clear_difference_gives_large_f(self):
        labels = ["a", "a", "a", "b", "b", "b"]
        values = [1.0, 1.1, 0.9, 5.0, 5.1, 4.9]
        self.assertGreater(probe.f_statistic(labels, values), 100)

    def test_no_within_spread_gives_none(self):
        """群内に揺れが無ければ F は決まらない（0 除算にしない）。"""
        self.assertIsNone(probe.f_statistic(["a", "a", "b", "b"], [1.0, 1.0, 2.0, 2.0]))


class PermutationTest(unittest.TestCase):
    def test_separated_groups_have_small_p(self):
        labels = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
        values = [1.0, 1.2, 0.8, 1.1, 5.0, 5.2, 4.8, 5.1, 9.0, 9.2, 8.8, 9.1]
        _, p = probe.image_permutation_test(labels, values)
        self.assertLess(p, 0.01)

    def test_mixed_groups_have_large_p(self):
        labels = ["a", "b", "c"] * 4
        values = [1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.1]
        _, p = probe.image_permutation_test(labels, values)
        self.assertGreater(p, 0.05)

    def test_same_input_same_p(self):
        labels = ["a", "b"] * 5
        values = [float(i % 3) for i in range(10)]
        self.assertEqual(probe.image_permutation_test(labels, values), probe.image_permutation_test(labels, values))

    def test_constant_values_give_none(self):
        self.assertEqual(probe.image_permutation_test(["a", "b"] * 3, [0.0] * 6), (None, None))


class ImageTest(unittest.TestCase):
    def setUp(self):
        self.cfg = helpers.load_config()

    def test_charts_match_lookback(self):
        for name, moves in probe.CHARTS.items():
            self.assertEqual(len(moves), self.cfg.vision_lookback_candles, name)

    def test_images_are_distinct_and_deterministic(self):
        first = probe.images(self.cfg)
        second = probe.images(self.cfg)
        self.assertEqual({k: v.sha256 for k, v in first.items()}, {k: v.sha256 for k, v in second.items()})
        self.assertEqual(len({v.sha256 for v in first.values()}), len(first))

    def test_blank_has_only_background(self):
        blank = probe.blank_image(self.cfg)
        self.assertEqual(set(blank.pixels), set(self.cfg.vision_palette["background"]))


if __name__ == "__main__":
    unittest.main()
