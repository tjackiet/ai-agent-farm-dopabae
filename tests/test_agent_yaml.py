"""agent.yaml が CLAUDE.md の不変ルールを満たしていることを確かめる。

まだ檻もシミュレーションも無い段階で CI が見るのはこれだけ。
外には出ない（bitbank も claude も呼ばない）。
"""

from __future__ import annotations

import pathlib
import unittest

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
AGENT_YAML = ROOT / "agent.yaml"

FORBIDDEN = (
    "bitbank trade create-order",
    "bitbank trade cancel-order",
    "bitbank paper reset",
)


def load() -> dict:
    with AGENT_YAML.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def walk_strings(node):
    """YAML の中の文字列値をすべて列挙する。"""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from walk_strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk_strings(v)


class AgentYamlTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load()

    def test_identity(self):
        agent = self.cfg["agent"]
        self.assertEqual(agent["id"], "dopabae")
        self.assertEqual(agent["name"], "ドパバエ")
        self.assertEqual(agent["name_en"], "Dopabae")
        self.assertIsInstance(self.cfg["version"], str)

    def test_paper_only(self):
        """実取引は行わない。CLI は paper モードで、禁止コマンドは列挙されている。"""
        cli = self.cfg["cli"]
        self.assertEqual(cli["mode"], "paper")
        for cmd in FORBIDDEN:
            self.assertIn(cmd, cli["forbidden"])

    def test_no_forbidden_command_is_used(self):
        """禁止コマンドが forbidden の列挙以外の場所に現れない。"""
        cfg = dict(self.cfg)
        cfg["cli"] = dict(cfg["cli"])
        cfg["cli"].pop("forbidden")
        for s in walk_strings(cfg):
            for cmd in FORBIDDEN:
                self.assertNotIn(cmd, s)

    def test_every_cli_command_is_public_or_paper(self):
        """観測は公開 API、口座と発注は paper。`bitbank trade` は出てこない。"""
        cli = self.cfg["cli"]
        for section in ("observe", "account", "act", "setup"):
            for cmd in cli[section].values():
                self.assertTrue(cmd.startswith("bitbank "), cmd)
                self.assertNotIn("bitbank trade", cmd)
        for section in ("account", "act", "setup"):
            for cmd in cli[section].values():
                self.assertTrue(cmd.startswith("bitbank paper "), cmd)

    def test_hold_on_failure(self):
        """判断できないときは HOLD。"""
        self.assertEqual(self.cfg["runtime"]["on_error"], "hold")
        self.assertEqual(self.cfg["direction"]["on_failure"], "hold")

    def test_fly_decides_direction_only(self):
        """ハエが決めるのは方向だけ。檻の値と読み出しの値は agent.yaml にある。"""
        self.assertIn(self.cfg["direction"]["source"], ("always_approach", "random", "fly"))
        strategy = self.cfg["strategy"]
        risk = self.cfg["risk"]
        self.assertTrue(strategy["spot_only"])
        for key in ("budget_jpy_per_order", "max_pending_buy_orders",
                    "cooldown_hours_after_fill", "max_fills_per_day"):
            self.assertIn(key, strategy["entry"])
        self.assertIn("time_stop_days", strategy["exit"])
        for key in ("max_position_ratio", "min_cash_reserve_ratio", "per_order_max_jpy"):
            self.assertIn(key, risk)
        self.assertLess(risk["drawdown"]["forced_exit_pct"], risk["drawdown"]["halt_new_buys_pct"])
        # 読み出しの閾値・対象ニューロンはここに置く（ハエが書き換えない）。未決なら null。
        readout = self.cfg["fly"]["readout"]
        for key in ("approach_neurons", "avoid_neurons",
                    "approach_threshold_hz", "avoid_threshold_hz"):
            self.assertIn(key, readout)

    def test_cage_is_within_capital(self):
        """檻の値が初期資金の範囲に収まる。"""
        capital = self.cfg["capital"]["initial_jpy"]
        entry = self.cfg["strategy"]["entry"]
        risk = self.cfg["risk"]
        self.assertLessEqual(entry["budget_jpy_per_order"], risk["per_order_max_jpy"])
        self.assertLessEqual(risk["per_order_max_jpy"], capital)
        self.assertLess(risk["max_position_ratio"] + risk["min_cash_reserve_ratio"], 1.0 + 1e-9)
        self.assertGreater(risk["guards"]["max_data_age_sec"], 0)

    def test_fly_does_not_see_its_wallet(self):
        """ハエに「自分の懐」を見せない。"""
        vision = self.cfg["fly"]["vision"]
        for item in ("balance", "pnl", "position"):
            self.assertIn(item, vision["hide"])
            self.assertNotIn(item, vision["show"])

    def test_reward_only_on_closed_position(self):
        """報酬は決済が確定した回だけ。学習は Phase 5 まで無効。"""
        reward = self.cfg["fly"]["reward"]
        self.assertEqual(reward["trigger"], "position_closed")
        self.assertFalse(reward["enabled"])

    def test_secrets_are_redacted(self):
        """API キー・シークレット・プロファイル名は記憶に残さない。"""
        redact = self.cfg["memory"]["redact"]
        for key in ("api_key", "api_secret", "profile_name"):
            self.assertIn(key, redact)


if __name__ == "__main__":
    unittest.main()
