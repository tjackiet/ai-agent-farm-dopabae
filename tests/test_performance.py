"""実績の集計。観測できなかった回を埋めないこと。"""

from __future__ import annotations

import unittest
from decimal import Decimal

from dopabae import performance
from tests import helpers

TZ = helpers.TZ


def record(minute: int, price: float | None, equity: float | None, amount: float = 0.0) -> dict:
    rec: dict = {
        "run_id": f"2026-09-15T09:{minute:02d}:00+09:00",
        "price": price,
        "position": {"amount": amount, "avg_cost": None},
    }
    rec["account"] = None if equity is None else {"cash_jpy": equity, "equity_jpy": equity, "drawdown_pct": 0.0}
    return rec


class PointsTest(unittest.TestCase):
    def test_skips_records_without_price_or_equity(self):
        records = [record(0, 100.0, 1000.0), record(15, None, 1000.0), record(30, 100.0, None)]
        self.assertEqual(len(performance.points_from(records, TZ)), 1)

    def test_does_not_fill_gaps(self):
        """観測できなかった回を前の値で埋めない（CLAUDE.md）。"""
        points = performance.points_from([record(0, 100.0, 1000.0), record(15, None, None), record(30, 110.0, 1100.0)], TZ)
        self.assertEqual([float(p.price) for p in points], [100.0, 110.0])

    def test_sorted_by_time(self):
        points = performance.points_from([record(30, 100.0, 1000.0), record(0, 90.0, 900.0)], TZ)
        self.assertLess(points[0].at, points[1].at)

    def test_bad_run_id_is_skipped(self):
        self.assertEqual(performance.points_from([{"run_id": "nonsense", "price": 1.0, "account": {"equity_jpy": 1.0}}], TZ), [])


class MeasureTest(unittest.TestCase):
    def points(self, pairs):
        return performance.points_from([record(i * 15, p, e) for i, (p, e) in enumerate(pairs)], TZ)

    def test_empty_gives_all_none(self):
        result = performance.measure([])
        self.assertEqual(result.observations, 0)
        self.assertIsNone(result.total_return_pct)
        self.assertIsNone(result.buy_and_hold_pct)

    def test_total_return_and_drawdown(self):
        # 1000 → 1200 → 900 → 1100
        result = performance.measure(self.points([(100.0, 1000.0), (120.0, 1200.0), (90.0, 900.0), (110.0, 1100.0)]))
        self.assertAlmostEqual(result.total_return_pct, 10.0, places=2)
        self.assertAlmostEqual(result.max_drawdown_pct, 25.0, places=2)  # 1200 → 900

    def test_buy_and_hold_and_excess(self):
        # 価格は +10%、資産も +10% なので超過はゼロ。
        result = performance.measure(self.points([(100.0, 1000.0), (110.0, 1100.0)]))
        self.assertAlmostEqual(result.buy_and_hold_pct, 10.0, places=2)
        self.assertAlmostEqual(result.excess_vs_buy_and_hold_pct, 0.0, places=2)

    def test_flat_equity_in_a_rising_market_is_negative_excess(self):
        """上げ相場ではどんな買い手も上手に見える。超過で見る。"""
        result = performance.measure(self.points([(100.0, 1000.0), (200.0, 1000.0)]))
        self.assertAlmostEqual(result.total_return_pct, 0.0, places=2)
        self.assertAlmostEqual(result.excess_vs_buy_and_hold_pct, -100.0, places=2)

    def test_single_point_has_no_buy_and_hold(self):
        self.assertIsNone(performance.measure(self.points([(100.0, 1000.0)])).buy_and_hold_pct)


class BuildTest(unittest.TestCase):
    def test_document_names_the_direction_source(self):
        cfg = helpers.load_config()
        document = performance.build(cfg, helpers.at(helpers.NOW_ISO), [record(0, 100.0, 1000.0)])
        self.assertEqual(document["direction_source"], cfg.direction_source)
        self.assertEqual(document["agent_id"], cfg.agent_id)
        self.assertEqual(document["observations"], 1)
