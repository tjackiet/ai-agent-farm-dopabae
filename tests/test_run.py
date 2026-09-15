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
