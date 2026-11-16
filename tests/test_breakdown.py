"""成績の内訳。観測値だけで分け、分けられない区間は分けないこと。"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import unittest
from pathlib import Path

from dopabae import breakdown, evaluate
from dopabae.direction import APPROACH, AVOID, NONE
from tests import helpers

TZ = helpers.TZ


def record(
    minute: int,
    value: str | None,
    price: float | None = 100.0,
    equity: float | None = 1000.0,
    cash: float | None = 1000.0,
) -> dict:
    return {
        "run_id": f"2026-09-15T{9 + minute // 60:02d}:{minute % 60:02d}:00+09:00",
        "config": {"version": "0.1", "fingerprint": "aaaa1111"},
        "price": price,
        "position": {"amount": 0.0, "avg_cost": None},
        "account": None if equity is None else {"cash_jpy": cash, "equity_jpy": equity, "drawdown_pct": 0.0},
        "direction": None if value is None else {"source": "fly", "value": value, "reason": "", "error": None, "seed": None},
        "cage": "still",
    }


def snaps(records):
    return breakdown.snapshots(records, TZ)


class SnapshotTest(unittest.TestCase):
    def test_exposure_and_btc_amount_come_from_equity_and_cash(self):
        (s,) = snaps([record(0, NONE, price=100.0, equity=1000.0, cash=600.0)])
        self.assertAlmostEqual(s.exposure, 0.4)
        self.assertAlmostEqual(s.btc_amount, 4.0)

    def test_missing_account_is_kept_but_not_complete(self):
        (s,) = snaps([record(0, NONE, equity=None)])
        self.assertFalse(s.complete)
        self.assertIsNone(s.equity_jpy)

    def test_unreadable_run_id_is_dropped(self):
        self.assertEqual(snaps([{**record(0, NONE), "run_id": "?"}]), [])


class HoldingsTest(unittest.TestCase):
    def test_holding_through_a_drop_is_market_pnl(self):
        """1 BTC を持ったまま 100 → 90。損益はすべて持っていたことによる。"""
        report = breakdown.holdings(snaps([
            record(0, NONE, price=100.0, equity=1000.0, cash=900.0),
            record(15, NONE, price=90.0, equity=990.0, cash=900.0),
        ]))
        self.assertEqual(report["pnl"]["market_jpy"], -10.0)
        self.assertEqual(report["pnl"]["trading_jpy"], 0.0)
        self.assertEqual(report["pnl"]["total_jpy"], -10.0)

    def test_parts_add_up_and_gaps_are_not_split(self):
        """観測の欠けた回をまたぐ区間は分けない。3 つの和は総資産の差に一致する。"""
        report = breakdown.holdings(snaps([
            record(0, NONE, price=100.0, equity=1000.0, cash=900.0),
            record(15, NONE, price=95.0, equity=994.0, cash=900.0),
            record(30, NONE, price=None, equity=None),
            record(45, NONE, price=80.0, equity=960.0, cash=960.0),
        ]))
        pnl = report["pnl"]
        self.assertEqual(pnl["attributed_intervals"], 1)
        self.assertEqual(pnl["unattributed_intervals"], 1)
        self.assertEqual(pnl["market_jpy"], -5.0)
        self.assertEqual(pnl["trading_jpy"], -1.0)
        self.assertEqual(pnl["unattributed_jpy"], -34.0)
        self.assertAlmostEqual(pnl["market_jpy"] + pnl["trading_jpy"] + pnl["unattributed_jpy"], pnl["total_jpy"])

    def test_level_and_timing_add_up(self):
        """軽くしてから下がった腕は timing が正になる。"""
        report = breakdown.holdings(snaps([
            record(0, NONE, price=100.0, equity=1000.0, cash=500.0),   # 重さ 0.5 → +10%
            record(15, NONE, price=110.0, equity=1050.0, cash=500.0),
            record(30, NONE, price=110.0, equity=1050.0, cash=1050.0),  # 重さ 0 → −20%
            record(45, NONE, price=88.0, equity=1050.0, cash=1050.0),
        ]))
        timing = report["timing"]
        weighted = 0.5 * 10.0 + 0.0 * 0.0 + 0.0 * -20.0
        self.assertAlmostEqual(timing["level_pct"] + timing["timing_pct"], weighted, places=3)
        self.assertGreater(timing["timing_pct"], 0)
        self.assertAlmostEqual(timing["market_sum_pct"], -10.0, places=3)

    def test_start_and_end_show_what_was_held(self):
        report = breakdown.holdings(snaps([
            record(0, NONE, price=100.0, equity=1000.0, cash=800.0),
            record(15, NONE, price=100.0, equity=1000.0, cash=1000.0),
        ]))
        self.assertEqual(report["start"]["exposure"], 0.2)
        self.assertEqual(report["start"]["btc_amount"], 2.0)
        self.assertEqual(report["end"]["exposure"], 0.0)
        self.assertEqual(report["exposure"]["nearly_flat_ratio"], 0.5)

    def test_daily_series_groups_by_local_date(self):
        records = [record(0, NONE, cash=500.0), record(15, NONE, cash=1000.0)]
        records.append({**record(0, NONE), "run_id": "2026-09-16T00:05:00+09:00"})
        daily = breakdown.holdings(snaps(records))["daily"]
        self.assertEqual([d["date"] for d in daily], ["2026-09-15", "2026-09-16"])
        self.assertEqual(daily[0]["mean_exposure"], 0.25)

    def test_no_observation_is_safe(self):
        report = breakdown.holdings(snaps([record(0, NONE, equity=None)]))
        self.assertEqual(report["observations"], 0)
        self.assertIsNone(report["pnl"])


class BackwardReturnTest(unittest.TestCase):
    def obs(self, rows):
        return evaluate.observations([record(m, v, price=p) for m, v, p in rows], TZ)

    def test_return_before_the_answer(self):
        by = breakdown.backward_returns(self.obs([(0, NONE, 100.0), (60, AVOID, 90.0)]), 60, 8)
        self.assertAlmostEqual(by[AVOID][0], -10.0, places=6)
        self.assertEqual(by[NONE], [])

    def test_gap_is_skipped_not_filled(self):
        by = breakdown.backward_returns(self.obs([(0, NONE, 100.0), (180, AVOID, 90.0)]), 60, 8)
        self.assertEqual(by[AVOID], [])

    def test_pairs_keep_time_order(self):
        pairs = breakdown.backward_pairs(
            self.obs([(0, NONE, 100.0), (15, APPROACH, 110.0), (30, AVOID, 99.0)]), 15, 8
        )
        self.assertEqual([d for d, _ in pairs], [APPROACH, AVOID])
        self.assertAlmostEqual(pairs[1][1], -10.0, places=6)

    def test_reaction_check_carries_a_warning_when_small(self):
        result = breakdown.reaction_check([(APPROACH, 1.0), (AVOID, -1.0)], 120)
        self.assertEqual(result["lookback_minutes"], 120)
        self.assertFalse(result["enough_samples"])
        self.assertIn("偶然と区別できない", result["sample_warning"])


class ShiftTest(unittest.TestCase):
    def test_answers_that_follow_the_values_are_detected(self):
        """手前が下げのときだけ AVOID と答える並びは、ずらすと崩れる。"""
        values = [((i * 37) % 101 - 50) / 10 for i in range(400)]
        pairs = [(AVOID if v < -3 else APPROACH, v) for v in values]
        difference, p, shifts = breakdown.shift_test(pairs)
        self.assertGreater(difference, 0)
        self.assertLess(p, 0.01)
        self.assertEqual(shifts, 400 - 2 * 20 + 1)

    def test_unrelated_runs_are_not_significant(self):
        """同じ答えが長く続いても、値動きと無関係なら小さな p 値にならない。"""
        labels = ([APPROACH] * 30 + [AVOID] * 10) * 10
        values = [((i * 53) % 97 - 48) / 10 for i in range(len(labels))]
        _, p, _ = breakdown.shift_test(list(zip(labels, values)))
        self.assertGreater(p, 0.05)

    def test_deterministic_and_never_zero(self):
        pairs = [(AVOID if i % 5 == 0 else APPROACH, float(i % 5 == 0)) for i in range(100)]
        first = breakdown.shift_test(pairs)
        self.assertEqual(first, breakdown.shift_test(pairs))
        self.assertGreater(first[1], 0.0)

    def test_missing_direction_gives_none(self):
        self.assertEqual(breakdown.shift_test([(APPROACH, 1.0), (NONE, 2.0)]), (None, None, 0))


class PersistenceTest(unittest.TestCase):
    def test_stay_ratio_and_independent_expectation(self):
        values = [APPROACH, APPROACH, AVOID, AVOID, NONE]
        result = breakdown.persistence(snaps([record(i * 15, v) for i, v in enumerate(values)]))
        self.assertEqual(result["pairs"], 4)
        self.assertEqual(result["same_as_previous_ratio"], 0.5)
        self.assertEqual(result["independent_expectation"], round(0.4**2 + 0.4**2 + 0.2**2, 4))
        self.assertEqual(result["stay"][AVOID], {"after": 2, "stayed_ratio": 0.5, "frequency": 0.4})

    def test_a_round_without_direction_breaks_the_chain(self):
        result = breakdown.persistence(snaps([record(0, AVOID), record(15, None), record(30, AVOID)]))
        self.assertEqual(result["pairs"], 0)
        self.assertIsNone(result["same_as_previous_ratio"])


class BreakdownTest(unittest.TestCase):
    def arm(self, label, records):
        cfg = helpers.load_config()
        return evaluate.Arm(label=label, source=cfg.direction_source, fingerprints=("aaaa1111",), records=records, config=cfg)

    def test_default_lookbacks_come_from_the_image_window(self):
        cfg = helpers.load_config()
        minutes = breakdown.CANDLE_MINUTES[cfg.vision_candle_type]
        self.assertEqual(
            breakdown.image_lookbacks(cfg), sorted({minutes, minutes * cfg.vision_lookback_candles})
        )

    def test_unknown_candle_type_is_an_error(self):
        cfg = dataclasses.replace(helpers.load_config(), vision_candle_type="2min")
        with self.assertRaises(ValueError):
            breakdown.image_lookbacks(cfg)

    def test_verdict_is_never_written_and_single_arm_is_noted(self):
        report = breakdown.breakdown([self.arm("fly", [record(0, APPROACH)])], None, 8)
        self.assertIsNone(report["verdict"])
        self.assertTrue(any("比較" in note for note in report["notes"]))

    def test_unattributed_intervals_are_noted(self):
        records = [record(0, NONE), record(15, NONE, equity=None), record(30, NONE)]
        report = breakdown.breakdown([self.arm("fly", records), self.arm("random", records)], [60], 8)
        self.assertTrue(any("unattributed_jpy" in note for note in report["notes"]))
        self.assertEqual([c["lookback_minutes"] for c in report["arms"][0]["reaction"]], [60])


class CliTest(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "dopabae.breakdown", *args],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent, timeout=120,
        )

    def test_runs_and_prints_json_without_touching_the_exchange(self):
        proc = self.run_cli("--lookback-minutes", "60")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertIsNone(report["verdict"])
        self.assertEqual(report["arms"][0]["reaction"][0]["lookback_minutes"], 60)

    def test_bad_since_is_an_error_not_a_crash(self):
        proc = self.run_cli("--since", "2026-10-20")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("error", json.loads(proc.stdout))

    def test_non_positive_lookback_is_an_error(self):
        proc = self.run_cli("--lookback-minutes", "0")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("error", json.loads(proc.stdout))


if __name__ == "__main__":
    unittest.main()
