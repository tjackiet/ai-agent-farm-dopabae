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


class ConnectomeConfigTest(unittest.TestCase):
    """配線図の取得元と指紋。トークン入りの URL と指紋の無いファイルを通さない。"""

    def setUp(self):
        self.raw = config_module.load().raw

    def with_connectome(self, **changes) -> dict:
        import copy

        raw = copy.deepcopy(self.raw)
        node = raw["fly"]["connectome"]
        for dotted, value in changes.items():
            target = node
            keys = dotted.split("__")
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
        return raw

    def test_loads_manifest(self):
        cfg = config_module.load()
        self.assertEqual(set(cfg.connectome_files), set(config_module.CONNECTOME_FILE_KEYS))
        for item in cfg.connectome_files.values():
            self.assertEqual(len(item.sha256), 64)
            self.assertGreater(item.size_bytes, 0)
        self.assertTrue(cfg.connectome_download_base.endswith("/"))

    def test_rejects_url_with_token(self):
        """Codex 由来の URL は API トークンを含む（docs/CONNECTOME_SURVEY.md 3.6）。"""
        raw = self.with_connectome(download_base="https://example.org/data?api_token=secret")
        with self.assertRaises(config_module.ConfigError):
            config_module.load(write(raw))

    def test_rejects_malformed_sha256(self):
        raw = self.with_connectome(files__weights__sha256="abc")
        with self.assertRaises(config_module.ConfigError):
            config_module.load(write(raw))

    def test_rejects_path_outside_download_base(self):
        for bad in ("/etc/passwd", "../other/file.feather"):
            raw = self.with_connectome(files__annotations__path=bad)
            with self.assertRaises(config_module.ConfigError, msg=bad):
                config_module.load(write(raw))

    def test_simulation_values_are_loaded(self):
        sim = config_module.load().simulation
        self.assertLess(sim.resting_mv, sim.threshold_mv)
        self.assertIn("histamine", sim.inhibitory_transmitters)
        self.assertEqual(sim.lamina_types, ("L1", "L2", "L3", "L5"))
        self.assertTrue(sim.carry_state)

    def test_rejects_equal_time_constants(self):
        """厳密解の式が 0 除算になる。"""
        import copy

        raw = copy.deepcopy(self.raw)
        raw["fly"]["simulation"]["synaptic_tau_ms"] = raw["fly"]["simulation"]["membrane_tau_ms"]
        with self.assertRaises(config_module.ConfigError):
            config_module.load(write(raw))

    def test_rejects_threshold_below_rest(self):
        import copy

        raw = copy.deepcopy(self.raw)
        raw["fly"]["simulation"]["threshold_mv"] = -60
        with self.assertRaises(config_module.ConfigError):
            config_module.load(write(raw))

    def test_rejects_zero_synapse_threshold(self):
        raw = self.with_connectome(synapse_threshold=0)
        with self.assertRaises(config_module.ConfigError):
            config_module.load(write(raw))

    def retina(self, **changes) -> dict:
        import copy

        raw = copy.deepcopy(self.raw)
        raw["fly"]["retina"].update(changes)
        return raw

    def test_by_subtype_needs_known_r8_types(self):
        """型から色を決められない R8 に、作った値を渡さない。"""
        raw = self.retina(r8_channel="by_subtype", blue_green_types=["R8p", "R8_unclear"])
        with self.assertRaises(config_module.ConfigError):
            config_module.load(write(raw))
        config_module.load(write(self.retina(r8_channel="blue_green_mean", blue_green_types=["R8p", "R8_unclear"])))

    def test_rejects_bad_eye_width(self):
        for bad in (0, 1.5, -0.2):
            with self.assertRaises(config_module.ConfigError, msg=bad):
                config_module.load(write(self.retina(eye_width=bad)))

    def test_rejects_empty_anchor_list(self):
        with self.assertRaises(config_module.ConfigError):
            config_module.load(write(self.retina(luminance_anchor_types=[])))
        self.assertIsNone(config_module.load(write(self.retina(luminance_anchor_types=None))).retina_luminance_anchor_types)

    def test_rejects_unknown_layout_and_sampling(self):
        for key, value in (("layout", "cyclops"), ("sampling", "bilinear")):
            with self.assertRaises(config_module.ConfigError, msg=key):
                config_module.load(write(self.retina(**{key: value})))

    def test_rejects_transmitter_both_inhibitory_and_modulatory(self):
        import copy

        raw = copy.deepcopy(self.raw)
        raw["fly"]["simulation"]["modulatory_transmitters"] = ["dopamine", "gaba"]
        with self.assertRaises(config_module.ConfigError):
            config_module.load(write(raw))

    def test_rejects_kc_rest_above_threshold(self):
        import copy

        raw = copy.deepcopy(self.raw)
        raw["fly"]["simulation"]["kc"]["resting_mv"] = -40
        with self.assertRaises(config_module.ConfigError):
            config_module.load(write(raw))

    def test_rejects_unknown_refractory_input(self):
        import copy

        raw = copy.deepcopy(self.raw)
        raw["fly"]["simulation"]["refractory_input"] = "maybe"
        with self.assertRaises(config_module.ConfigError):
            config_module.load(write(raw))

    def test_rejects_missing_file_entry(self):
        raw = self.with_connectome()
        del raw["fly"]["connectome"]["files"]["weights"]
        with self.assertRaises(config_module.ConfigError):
            config_module.load(write(raw))
