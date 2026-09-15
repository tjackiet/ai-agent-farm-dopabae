"""agent.yaml の読み込み。欠けた値や現物でない設定を黙って通さないこと。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from dopabae import config as config_module


def write(raw: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(raw, tmp, allow_unicode=True)
    tmp.close()
    return Path(tmp.name)


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.raw = config_module.load().raw

    def test_loads_repo_config(self):
        cfg = config_module.load()
        self.assertEqual(cfg.agent_id, "dopabae")
        self.assertTrue(cfg.spot_only)
        self.assertEqual(cfg.direction_on_failure, "hold")

    def test_rejects_non_spot(self):
        raw = dict(self.raw); raw["strategy"] = dict(raw["strategy"]); raw["strategy"]["spot_only"] = False
        with self.assertRaises(config_module.ConfigError):
            config_module.load(write(raw))

    def test_rejects_non_paper_mode(self):
        raw = dict(self.raw); raw["cli"] = dict(raw["cli"]); raw["cli"]["mode"] = "live"
        with self.assertRaises(config_module.ConfigError):
            config_module.load(write(raw))

    def test_rejects_unknown_direction_source(self):
        raw = dict(self.raw); raw["direction"] = dict(raw["direction"]); raw["direction"]["source"] = "llm"
        with self.assertRaises(config_module.ConfigError):
            config_module.load(write(raw))

    def test_rejects_on_failure_other_than_hold(self):
        raw = dict(self.raw); raw["direction"] = dict(raw["direction"]); raw["direction"]["on_failure"] = "proceed"
        with self.assertRaises(config_module.ConfigError):
            config_module.load(write(raw))

    def test_missing_value_is_an_error(self):
        raw = dict(self.raw); raw["risk"] = dict(raw["risk"]); del raw["risk"]["per_order_max_jpy"]
        with self.assertRaises(config_module.ConfigError):
            config_module.load(write(raw))

    def test_fingerprint_changes_with_cage_values(self):
        a = config_module.fingerprint(self.raw)
        raw = dict(self.raw); raw["risk"] = dict(raw["risk"]); raw["risk"]["per_order_max_jpy"] = 1
        self.assertNotEqual(a, config_module.fingerprint(raw))
