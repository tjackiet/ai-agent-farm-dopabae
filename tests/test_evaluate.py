"""評価。ランダムと区別できるかを、標本の少なさごと正直に出すこと。"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dopabae import evaluate
from dopabae.direction import APPROACH, AVOID, NONE
from tests import helpers

TZ = helpers.TZ


def record(minute: int, value: str | None, price: float = 100.0, cage: str = "bought", fingerprint: str = "aaaa1111") -> dict:
    return {
        "run_id": f"2026-09-15T{9 + minute // 60:02d}:{minute % 60:02d}:00+09:00",
        "config": {"version": "0.1", "fingerprint": fingerprint},
        "price": price,
        "position": {"amount": 0.0, "avg_cost": None},
        "account": {"cash_jpy": 1000.0, "equity_jpy": 1000.0, "drawdown_pct": 0.0},
        "direction": None if value is None else {"source": "fly", "value": value, "reason": "", "error": None, "seed": None},
        "cage": cage,
    }


class DirectionStatsTest(unittest.TestCase):
    def test_counts_and_frequencies(self):
        stats = evaluate.direction_stats([record(0, APPROACH), record(15, AVOID), record(30, APPROACH), record(45, NONE)])
        self.assertEqual(stats.counts[APPROACH], 2)
        self.assertEqual(stats.frequencies[APPROACH], 0.5)
        self.assertEqual(stats.decided, 4)

    def test_unavailable_and_not_consulted_are_separate(self):
        failed = record(0, None)
        failed["direction"] = {"source": "fly", "value": None, "error": "見えなかった"}
        stats = evaluate.direction_stats([failed, record(15, None)])
        self.assertEqual((stats.unavailable, stats.not_consulted), (1, 1))
        self.assertEqual(stats.decided, 0)

    def test_one_sided_bias_is_flagged(self):
        """いつも同じ方向なら、相場の洞察ではなくネットワークの癖。"""
        stats = evaluate.direction_stats([record(i * 15, APPROACH) for i in range(19)] + [record(300, AVOID)])
        self.assertTrue(stats.as_dict()["nearly_one_sided"])
        self.assertEqual(stats.as_dict()["bias_direction"], APPROACH)

    def test_balanced_is_not_flagged(self):
        stats = evaluate.direction_stats([record(i * 15, APPROACH if i % 2 else AVOID) for i in range(20)])
        self.assertFalse(stats.as_dict()["nearly_one_sided"])

    def test_empty_is_safe(self):
        stats = evaluate.direction_stats([])
        self.assertIsNone(stats.as_dict()["bias_direction"])
        self.assertFalse(stats.as_dict()["nearly_one_sided"])


class ForwardReturnTest(unittest.TestCase):
    def obs(self, rows):
        return evaluate.observations([record(m, v, price=p) for m, v, p in rows], TZ)

    def test_return_is_measured_at_the_horizon(self):
        by = evaluate.forward_returns(self.obs([(0, APPROACH, 100.0), (60, NONE, 110.0)]), 60, 8)
        self.assertAlmostEqual(by[APPROACH][0], 10.0, places=6)

    def test_gap_is_skipped_not_filled(self):
        """観測が欠けた区間は飛ばす。前の値で埋めない。"""
        by = evaluate.forward_returns(self.obs([(0, APPROACH, 100.0), (180, NONE, 110.0)]), 60, 8)
        self.assertEqual(by[APPROACH], [])

    def test_tolerance_accepts_a_small_drift(self):
        by = evaluate.forward_returns(self.obs([(0, APPROACH, 100.0), (65, NONE, 110.0)]), 60, 8)
        self.assertEqual(len(by[APPROACH]), 1)

    def test_last_observations_have_no_forward_return(self):
        by = evaluate.forward_returns(self.obs([(0, APPROACH, 100.0)]), 60, 8)
        self.assertEqual(sum(len(v) for v in by.values()), 0)


class PermutationTest(unittest.TestCase):
    def test_reproducible(self):
        a, b = [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]
        self.assertEqual(evaluate.permutation_test(a, b), evaluate.permutation_test(a, b))

    def test_identical_groups_are_not_significant(self):
        values = [float(i % 7) for i in range(60)]
        _, p = evaluate.permutation_test(values, list(values), trials=2000)
        self.assertGreater(p, 0.2)

    def test_clearly_separated_groups_are_significant(self):
        _, p = evaluate.permutation_test([10.0] * 30, [-10.0] * 30, trials=2000)
        self.assertLess(p, 0.01)

    def test_empty_group_gives_none(self):
        self.assertEqual(evaluate.permutation_test([], [1.0]), (None, None))

    def test_p_value_is_never_zero(self):
        _, p = evaluate.permutation_test([5.0] * 20, [-5.0] * 20, trials=100)
        self.assertGreater(p, 0.0)


class SignalCheckTest(unittest.TestCase):
    def test_small_sample_carries_a_warning(self):
        result = evaluate.signal_check({APPROACH: [1.0, 2.0], AVOID: [0.0], NONE: []})
        self.assertFalse(result["enough_samples"])
        self.assertIn("偶然と区別できない", result["sample_warning"])

    def test_enough_samples_clears_the_warning(self):
        n = evaluate.MIN_SAMPLES_PER_DIRECTION
        result = evaluate.signal_check({APPROACH: [1.0] * n, AVOID: [0.0] * n, NONE: []})
        self.assertTrue(result["enough_samples"])
        self.assertIsNone(result["sample_warning"])


class CompareTest(unittest.TestCase):
    def arm(self, label, records, source="fly"):
        cfg = dataclasses.replace(helpers.load_config(), direction_source=source)
        fps = sorted({r["config"]["fingerprint"] for r in records})
        return evaluate.Arm(label=label, source=source, fingerprints=tuple(fps), records=records, config=cfg)

    def test_single_arm_is_not_a_comparison(self):
        report = evaluate.compare([self.arm("fly", [record(0, APPROACH)])], 60, 8)
        self.assertTrue(any("比較" in note for note in report["notes"]))

    def test_mixed_config_is_reported(self):
        records = [record(0, APPROACH), record(15, AVOID, fingerprint="bbbb2222")]
        report = evaluate.compare([self.arm("fly", records), self.arm("random", records, "random")], 60, 8)
        self.assertTrue(any("混ざっている" in note for note in report["notes"]))

    def test_matched_weights_come_from_observed_frequencies(self):
        records = [record(0, APPROACH), record(15, APPROACH), record(30, AVOID), record(45, NONE)]
        report = evaluate.compare([self.arm("fly", records)], 60, 8)
        self.assertEqual(report["arms"][0]["matched_random_weights"][APPROACH], 0.5)

    def test_verdict_is_never_written_by_the_tool(self):
        """結論は人間が書く。設計メモ 4 章の 1〜4 が揃うまで「学んだ」と言わない。"""
        report = evaluate.compare([self.arm("fly", [record(0, APPROACH)])], 60, 8)
        self.assertIsNone(report["verdict"])

    def test_cage_breakdown_is_counted(self):
        records = [record(0, APPROACH, cage="bought"), record(15, APPROACH, cage="wall"), record(30, APPROACH, cage="wall")]
        report = evaluate.compare([self.arm("fly", records)], 60, 8)
        self.assertEqual(report["arms"][0]["cage"], {"bought": 1, "wall": 2})


class CliTest(unittest.TestCase):
    def test_runs_and_prints_json_without_touching_the_exchange(self):
        proc = subprocess.run(
            [sys.executable, "-m", "dopabae.evaluate", "--horizon-minutes", "60"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent, timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertIsNone(report["verdict"])
        self.assertEqual(report["horizon_minutes"], 60)


class MakeArmTest(unittest.TestCase):
    def test_changes_only_the_direction_and_the_paths(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        try:
            import make_arm
        finally:
            sys.path.pop(0)
        import yaml

        raw = yaml.safe_load((Path(__file__).resolve().parent.parent / "agent.yaml").read_text(encoding="utf-8"))
        arm = make_arm.build(raw, "random", "random", 7, {"APPROACH": 0.5})
        # 檻とリスクは正本と同じ。腕の差が方向の差だけになるように。
        self.assertEqual(arm["strategy"], raw["strategy"])
        self.assertEqual(arm["risk"], raw["risk"])
        self.assertEqual(arm["direction"]["source"], "random")
        self.assertEqual(arm["direction"]["random_seed"], 7)
        # 置き場所は腕ごとに分かれる。分けないと口座とログを奪い合う。
        self.assertNotEqual(arm["cli"]["state_path"], raw["cli"]["state_path"])
        self.assertNotEqual(arm["memory"]["decisions"]["path"], raw["memory"]["decisions"]["path"])
        self.assertIn("var/arms/random", arm["agent"]["status_output"])
        # 元の辞書を壊していない。
        self.assertEqual(raw["direction"]["source"], "always_approach")
