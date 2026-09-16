"""1周の流れ。外には出ない。観測に失敗した回は status.yaml を更新しない。"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from dopabae import cli
from dopabae.direction import AVOID, Direction
from dopabae.run import run_once
from tests import helpers
from tests.helpers import FakeCli, default_responses, load_config

NOW = helpers.at(helpers.NOW_ISO)

POSITION_PNL = {
    "perPair": {
        "btc_jpy": {
            "pair": "btc_jpy",
            "position": 0.0068,
            "avgCost": 14550000,
            "currentPrice": 14700000,
            "realizedPnl": 0,
            "unrealizedPnl": 1020,
            "totalPnl": 1020,
        }
    },
    "total": {"realizedPnl": 0, "unrealizedPnl": 1020, "totalPnl": 1020},
}
BUY_HISTORY = [
    {"id": "t1", "pair": "btc_jpy", "side": "buy", "type": "limit", "amount": 0.0068,
     "fillPrice": 14550000, "feeQuote": 0, "filledAt": "2026-09-14T00:00:00.000Z"}
]
ASSETS_WITH_POSITION = [
    {"asset": "jpy", "total": 901060, "locked": 0, "available": 901060},
    {"asset": "btc", "total": 0.0068, "locked": 0, "available": 0.0068},
]


class CycleTest(unittest.TestCase):
    def setUp(self):
        # dry_run を外して、発注コマンドが FakeCli に届くことを見る。
        self.config = dataclasses.replace(load_config(), dry_run=False)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cycle(self, fake: FakeCli, config=None, resolver=None, **kw):
        cfg = config if config is not None else self.config
        client = cli.Client(cfg, runner=fake)
        return run_once(cfg, client, NOW, repo_root=self.root, resolver=resolver, **kw)

    def journal(self) -> list[dict]:
        path = self.root / "var" / "memory" / "decisions" / "2026-09-15.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_always_approach_buys_and_records_direction_and_cage(self):
        fake = FakeCli(default_responses())
        cycle = self.run_cycle(fake)
        self.assertEqual(cycle.action, "BUY")
        self.assertEqual(cycle.cage, "bought")
        self.assertEqual(cycle.direction["value"], "APPROACH")
        self.assertTrue(any("paper create-order" in c and "--side=buy" in c for c in fake.calls))
        record = self.journal()[-1]
        self.assertEqual(record["direction"]["source"], "always_approach")
        self.assertEqual(record["cage"], "bought")
        self.assertIsNotNone(record["config"]["fingerprint"])
        self.assertTrue(cycle.status_written)
        status = yaml.safe_load((self.root / "var" / "status.yaml").read_text(encoding="utf-8"))
        self.assertEqual(status["state"], "IDLE")
        self.assertEqual(status["last_direction"]["value"], "APPROACH")
        self.assertEqual(status["mood"], "ドパ切れ")

    def test_dry_run_builds_orders_without_calling_create_order(self):
        fake = FakeCli(default_responses())
        cycle = self.run_cycle(fake, force_dry_run=True)
        self.assertEqual(cycle.action, "BUY")
        self.assertFalse(cycle.orders[0]["executed"])
        self.assertFalse(any("create-order" in c for c in fake.calls))

    def test_avoid_sells_position(self):
        fake = FakeCli(default_responses(assets=ASSETS_WITH_POSITION, pnl=POSITION_PNL, history=BUY_HISTORY))
        cycle = self.run_cycle(fake, resolver=lambda cfg, key: Direction(source="test", value=AVOID))
        self.assertEqual((cycle.action, cycle.cage), ("SELL", "sold"))
        self.assertTrue(any("--side=sell" in c for c in fake.calls))

    def test_observation_failure_holds_and_does_not_write_status(self):
        fake = FakeCli(default_responses(), errors={"ticker": "boom"})
        cycle = self.run_cycle(fake)
        self.assertEqual(cycle.action, "HOLD")
        self.assertEqual(cycle.cage, "guard")
        self.assertIsNotNone(cycle.error)
        self.assertIsNone(cycle.direction)  # 観測に失敗した回は方向を諮らない
        self.assertFalse(cycle.status_written)
        self.assertFalse((self.root / "var" / "status.yaml").exists())
        self.assertEqual(len(self.journal()), 1)  # HOLD でも1行残す
        self.assertFalse(any("create-order" in c for c in fake.calls))

    def test_direction_failure_holds(self):
        fake = FakeCli(default_responses())
        cycle = self.run_cycle(fake, resolver=lambda cfg, key: Direction(source="fly", value=None, error="見えなかった"))
        self.assertEqual((cycle.action, cycle.cage), ("HOLD", "no_direction"))
        self.assertTrue(any("方向を得られません" in w for w in cycle.warnings))
        self.assertFalse(any("create-order" in c for c in fake.calls))

    def test_never_builds_a_forbidden_command(self):
        fake = FakeCli(default_responses(assets=ASSETS_WITH_POSITION, pnl=POSITION_PNL, history=BUY_HISTORY))
        self.run_cycle(fake, resolver=lambda cfg, key: Direction(source="test", value=AVOID))
        for call in fake.calls:
            self.assertNotIn("bitbank trade", call)
            self.assertNotIn("paper reset", call)

    def test_random_source_records_seed(self):
        cfg = dataclasses.replace(self.config, direction_source="random", direction_random_seed=3)
        cycle = self.run_cycle(FakeCli(default_responses()), config=cfg)
        self.assertEqual(cycle.direction["seed"], 3)


class VisionCycleTest(CycleTest):
    def test_vision_hash_and_png_are_recorded(self):
        fake = FakeCli(default_responses())
        cycle = self.run_cycle(fake, force_dry_run=True)
        self.assertIsNotNone(cycle.vision)
        self.assertEqual(len(cycle.vision["sha256"]), 64)
        self.assertEqual(cycle.vision["candle_count"], self.config.vision_lookback_candles)
        record = self.journal()[-1]
        self.assertEqual(record["vision"]["sha256"], cycle.vision["sha256"])
        pngs = list((self.root / "var" / "memory" / "vision" / "2026-09-15").glob("*.png"))
        self.assertEqual(len(pngs), 1)
        self.assertTrue(any("candles" in c and "--type=15min" in c for c in fake.calls))

    def test_render_failure_warns_but_control_groups_keep_deciding(self):
        fake = FakeCli(default_responses(), errors={"candles": "boom"})
        cycle = self.run_cycle(fake, force_dry_run=True)
        self.assertIsNone(cycle.vision)
        self.assertTrue(any("画像を描けません" in w for w in cycle.warnings))
        self.assertEqual(cycle.action, "BUY")  # always_approach は画像を見ない
        self.assertIsNone(self.journal()[-1]["vision"])


class PerformanceCycleTest(CycleTest):
    def test_equity_is_recorded_as_an_observation_and_performance_is_written(self):
        fake = FakeCli(default_responses())
        cycle = self.run_cycle(fake, force_dry_run=True)
        record = self.journal()[-1]
        self.assertEqual(record["account"]["equity_jpy"], 1000000.0)
        self.assertEqual(record["account"]["cash_jpy"], 1000000.0)
        document = yaml.safe_load((self.root / "var" / "performance.yaml").read_text(encoding="utf-8"))
        self.assertEqual(document["observations"], 1)
        self.assertEqual(document["direction_source"], self.config.direction_source)
        self.assertTrue(cycle.status_written)

    def test_failed_observation_records_no_account(self):
        fake = FakeCli(default_responses(), errors={"ticker": "boom"})
        self.run_cycle(fake)
        self.assertIsNone(self.journal()[-1]["account"])


class NarrateCycleTest(CycleTest):
    """言語化は記録の文章を書くだけ。失敗しても発注は止まらない。"""

    def put_diary(self):
        """空欄のある日誌を置く。これが無いと書き手は呼ばれない。"""
        from dopabae import summary

        directory = self.root / "var" / "memory" / "daily"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "2026-09-14.md"
        path.write_text(f"# 2026-09-14\n\n## おれの一日\n\n{summary.UNWRITTEN}\n", encoding="utf-8")
        return path

    def test_a_failing_narrator_only_warns(self):
        self.put_diary()

        def broken(system, user):
            raise RuntimeError("boom")

        cycle = self.run_cycle(FakeCli(default_responses()), force_dry_run=True, narrator=broken)
        self.assertEqual(cycle.action, "BUY")  # 判断は通っている
        self.assertIsNone(cycle.error)
        self.assertTrue(any("日誌の本文を書けません" in w for w in cycle.warnings))

    def test_a_working_narrator_fills_the_diary(self):
        path = self.put_diary()
        cycle = self.run_cycle(
            FakeCli(default_responses()), force_dry_run=True,
            narrator=lambda s, u: "近づいた。もっかい。",
        )
        self.assertEqual(cycle.action, "BUY")
        self.assertIn("近づいた。もっかい。", path.read_text(encoding="utf-8"))

    def test_the_narrator_is_not_called_when_there_is_no_diary(self):
        """その日が終わるまで日誌は作られない。書くものが無ければ呼ばない。"""
        calls = []
        cycle = self.run_cycle(
            FakeCli(default_responses()), force_dry_run=True,
            narrator=lambda s, u: calls.append(u) or "近づいた。",
        )
        self.assertEqual(cycle.action, "BUY")
        self.assertEqual(calls, [])
