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


class WindowTest(unittest.TestCase):
    def test_since_and_until_keep_only_the_window(self):
        records = [record(0, APPROACH), record(15, AVOID), record(30, NONE), record(45, APPROACH)]
        kept = evaluate.within(records, "Asia/Tokyo", "2026-09-15T09:15:00+09:00", "2026-09-15T09:30:00+09:00")
        self.assertEqual([r["run_id"] for r in kept], [records[1]["run_id"], records[2]["run_id"]])
        self.assertEqual(evaluate.within(records, "Asia/Tokyo", None, None), records)

    def test_unreadable_run_id_is_dropped_only_when_filtering(self):
        records = [record(0, APPROACH), {**record(15, AVOID), "run_id": "?"}]
        self.assertEqual(len(evaluate.within(records, "Asia/Tokyo", "2026-09-15T00:00:00+09:00", None)), 1)

    def test_bad_since_is_an_error_not_a_crash(self):
        proc = subprocess.run(
            [sys.executable, "-m", "dopabae.evaluate", "--since", "2026-10-20"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent, timeout=120,
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("error", json.loads(proc.stdout))


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

    def _make_arm(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        try:
            import make_arm
        finally:
            sys.path.pop(0)
        import yaml

        raw = yaml.safe_load((Path(__file__).resolve().parent.parent / "agent.yaml").read_text(encoding="utf-8"))
        return make_arm, raw

    def test_fly_arm_can_change_only_the_half_saturation(self):
        make_arm, raw = self._make_arm()
        arm = make_arm.build(raw, "fly-hs0.2", "fly", None, None, half_saturation=0.2)
        self.assertEqual(arm["fly"]["simulation"]["photoreceptor_input"]["half_saturation"], 0.2)
        # 檻とリスクは正本と同じ。
        self.assertEqual(arm["strategy"], raw["strategy"])
        self.assertEqual(arm["risk"], raw["risk"])
        # 半飽和のほかの網の値は変えない。
        base = make_arm.build(raw, "fly-hs0.2", "fly", None, None)
        base["fly"]["simulation"]["photoreceptor_input"]["half_saturation"] = 0.2
        self.assertEqual(arm, base)
        # 元の辞書を壊していない。
        self.assertEqual(raw["fly"]["simulation"]["photoreceptor_input"]["half_saturation"], 0.02)

    def test_fly_arms_do_not_share_the_network_state(self):
        """網の状態を腕ごとに分ける。分けないとハエ版の腕どうしが上書きし合う。"""
        make_arm, raw = self._make_arm()
        a = make_arm.build(raw, "fly", "fly", None, None)
        b = make_arm.build(raw, "fly-hs0.2", "fly", None, None, half_saturation=0.2)
        paths = {raw["fly"]["simulation"]["checkpoint_path"],
                 a["fly"]["simulation"]["checkpoint_path"],
                 b["fly"]["simulation"]["checkpoint_path"]}
        self.assertEqual(len(paths), 3)

    def test_arm_with_half_saturation_loads_as_config(self):
        make_arm, raw = self._make_arm()
        import yaml

        from dopabae import config

        arm = make_arm.build(raw, "fly-hs0.2", "fly", None, None, half_saturation=0.2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arm.yaml"
            path.write_text(yaml.safe_dump(arm, allow_unicode=True), encoding="utf-8")
            cfg = config.load(path)
        self.assertEqual(cfg.simulation.input_half_saturation, 0.2)

    def test_fly_arm_can_change_the_motion(self):
        make_arm, raw = self._make_arm()
        arm = make_arm.build(raw, "fly-grow", "fly", None, None, motion="grow_last")
        self.assertEqual(arm["fly"]["vision"]["motion"]["mode"], "grow_last")
        self.assertEqual(arm["strategy"], raw["strategy"])
        self.assertEqual(arm["risk"], raw["risk"])
        self.assertEqual(raw["fly"]["vision"]["motion"]["mode"], "none")
        script = Path(__file__).resolve().parent.parent / "scripts" / "make_arm.py"
        proc = subprocess.run([sys.executable, str(script), "x", "--source", "random", "--motion", "grow_last"],
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 2)

    def test_fly_arm_can_change_the_palette_and_flip_y(self):
        make_arm, raw = self._make_arm()
        gray = [118, 118, 118]
        arm = make_arm.build(raw, "fly-mono", "fly", None, None,
                             palette={"up": gray, "down": gray}, flip_y=False)
        self.assertEqual(arm["fly"]["vision"]["palette"]["up"], gray)
        self.assertEqual(arm["fly"]["vision"]["palette"]["down"], gray)
        self.assertEqual(arm["fly"]["vision"]["palette"]["wick"], raw["fly"]["vision"]["palette"]["wick"])
        self.assertFalse(arm["fly"]["retina"]["flip_y"])
        self.assertEqual(arm["strategy"], raw["strategy"])
        self.assertEqual(arm["risk"], raw["risk"])
        self.assertNotEqual(raw["fly"]["vision"]["palette"]["up"], gray)
        self.assertTrue(raw["fly"]["retina"]["flip_y"])

    def test_palette_is_parsed_strictly(self):
        make_arm, _ = self._make_arm()
        self.assertEqual(make_arm.parse_palette(["up=1,2,3"]), {"up": [1, 2, 3]})
        for bad in ("up=1,2", "up=1,2,256", "sky=1,2,3", "up", "up=a,b,c"):
            with self.assertRaises(ValueError, msg=bad):
                make_arm.parse_palette([bad])
        script = Path(__file__).resolve().parent.parent / "scripts" / "make_arm.py"
        for extra in (["--source", "random", "--flip-y", "false"],
                      ["--source", "cage", "--palette", "up=1,2,3"],
                      ["--source", "fly", "--palette", "up=1,2"]):
            proc = subprocess.run([sys.executable, str(script), "x", *extra],
                                  capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 2, extra)

    def test_fly_arm_can_drop_the_gate(self):
        """関門なしの腕（2026-10-14 に人間が決めた fly-hs1-nogate）。差だけで方向を出す。"""
        make_arm, raw = self._make_arm()
        arm = make_arm.build(raw, "fly-hs1-nogate", "fly", None, None, half_saturation=1.0, no_gate=True)
        self.assertIsNone(arm["fly"]["readout"]["gate_neurons"])
        self.assertEqual(raw["fly"]["readout"]["gate_neurons"], ["DNpe017"])
        base = make_arm.build(raw, "fly-hs1-nogate", "fly", None, None, half_saturation=1.0)
        base["fly"]["readout"]["gate_neurons"] = None
        self.assertEqual(arm, base)
        self.assertIsNone(self._load(arm).readout_gate)

    def test_fly_arm_can_turn_on_the_mbon_term(self):
        make_arm, raw = self._make_arm()
        baselines = make_arm.parse_mbon_baselines(["MBON11=17.7", "MBON07=1.14"])
        arm = make_arm.build(raw, "fly-b2", "fly", None, None, mbon_baselines=baselines)
        cfg = self._load(arm)
        self.assertEqual((cfg.readout_mbon.approach_baseline_hz, cfg.readout_mbon.avoid_baseline_hz), (17.7, 1.14))
        self.assertEqual(cfg.readout_mbon.weight, 0.5)
        self.assertFalse(raw["fly"]["readout"]["mbon"]["enabled"])
        with self.assertRaises(ValueError):
            make_arm.build(raw, "x", "fly", None, None, mbon_baselines={"MBON11": 17.7})
        for bad in ("MBON11", "MBON11=0", "MBON11=x", "=1"):
            with self.assertRaises(ValueError, msg=bad):
                make_arm.parse_mbon_baselines([bad])
        script = Path(__file__).resolve().parent.parent / "scripts" / "make_arm.py"
        for extra in (["--source", "random", "--no-gate"],
                      ["--source", "always_approach", "--mbon-baseline", "MBON11=17.7", "--mbon-baseline", "MBON07=1.1"],
                      ["--source", "fly", "--mbon-baseline", "MBON11=17.7"]):
            proc = subprocess.run([sys.executable, str(script), "x", *extra],
                                  capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 2, extra)

    def _load(self, arm):
        import yaml

        from dopabae import config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arm.yaml"
            path.write_text(yaml.safe_dump(arm, allow_unicode=True), encoding="utf-8")
            return config.load(path)

    def test_half_saturation_is_only_for_fly_arms(self):
        """対照群は画像を見ないので、半飽和を変えても腕の差にならない。"""
        script = Path(__file__).resolve().parent.parent / "scripts" / "make_arm.py"
        for extra in (["--source", "random", "--half-saturation", "0.2"],
                      ["--source", "fly", "--half-saturation", "0"]):
            proc = subprocess.run([sys.executable, str(script), "x", *extra],
                                  capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 2, extra)
