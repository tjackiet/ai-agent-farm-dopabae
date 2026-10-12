"""ハエ版の方向の出どころ。読み出しの規則、状態の引き継ぎ、失敗が HOLD に伝わること。

配線図データは使わない。小さな網と座標表を作って動かす。numpy が無ければ飛ばす。
"""

from __future__ import annotations

import dataclasses
import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dopabae import direction, retina
from tests import helpers
from tests.test_retina import image, lattice, write_map

try:
    import numpy as np
except ImportError:  # pragma: no cover - 依存が無い環境では飛ばす
    np = None

if np is not None:
    from dopabae import connectome, fly, lif

DNP20_L, DNP20_R, DNPE017 = 9001, 9002, 9003


def small_network(column_map: retina.ColumnMap, wire_to: int | None) -> "connectome.Network":
    """光受容細胞と、読み出しの 3 細胞だけの網。光受容細胞は wire_to と DNpe017 を興奮させる。"""
    photo = sorted(c.body_id for c in column_map.mapped)
    body_ids = np.array(photo + [DNP20_L, DNP20_R, DNPE017], dtype=np.int64)
    index = {int(b): i for i, b in enumerate(body_ids)}
    edges = []
    for b in photo:
        if wire_to is not None:
            edges.append((index[b], index[wire_to], 30))
        edges.append((index[b], index[DNPE017], 30))
    pre, post, signed = zip(*edges)
    indptr, post_sorted, signed_sorted = connectome.build_csr(len(body_ids), list(pre), list(post), list(signed))
    types = [c.cell_type for c in sorted(column_map.mapped, key=lambda c: c.body_id)] + ["DNp20", "DNp20", "DNpe017"]
    sides = [c.side for c in sorted(column_map.mapped, key=lambda c: c.body_id)] + ["L", "R", "R"]
    return connectome.Network(
        body_ids=body_ids,
        indptr=indptr,
        post=post_sorted,
        signed_count=signed_sorted,
        cell_type=np.array(types, dtype=str),
        side=np.array(sides, dtype=str),
        modulatory=np.zeros(len(body_ids), dtype=bool),
        meta={"key": f"fly-test-{wire_to}"},
    )


@unittest.skipIf(np is None, "numpy が無い")
class ReadTest(unittest.TestCase):
    """Stonkfly の Decoder と同じ規則: |右 − 左| が閾値以上で方向、関門が 0 回なら NONE。"""

    def setUp(self):
        self.dir = TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.map = retina.load_column_map(write_map(Path(self.dir.name), lattice(1)))
        self.net = small_network(self.map, None)
        self.cfg = dataclasses.replace(helpers.load_config(), direction_source="fly")

    def counts(self, left: int, right: int, gate: int):
        c = np.zeros(self.net.n_neurons, dtype=np.int32)
        c[self.net.index_of(np.array([DNP20_L, DNP20_R, DNPE017]))] = [left, right, gate]
        return c

    def test_threshold_is_inclusive(self):
        """1 細胞・0.5 秒では 1 回の差が 2 Hz。ちょうど 2 Hz で方向を出す（Stonkfly と同じ）。"""
        out = fly.read(self.cfg, self.net, self.counts(0, 1, 1), 0.5)
        self.assertEqual(out["axis_hz"], 2.0)
        self.assertEqual(out["value"], direction.APPROACH)
        self.assertEqual(fly.read(self.cfg, self.net, self.counts(1, 0, 1), 0.5)["value"], direction.AVOID)
        self.assertEqual(fly.read(self.cfg, self.net, self.counts(3, 3, 1), 0.5)["value"], direction.NONE)

    def test_gate_closed_means_none(self):
        out = fly.read(self.cfg, self.net, self.counts(0, 5, 0), 0.5)
        self.assertEqual(out["value"], direction.NONE)
        self.assertEqual(out["gate_spikes"], 0)
        self.assertIn("関門", out["rule"])

    def test_no_gate_configured(self):
        cfg = dataclasses.replace(self.cfg, readout_gate=None)
        out = fly.read(cfg, self.net, self.counts(0, 1, 0), 0.5)
        self.assertEqual(out["value"], direction.APPROACH)
        self.assertIsNone(out["gate_spikes"])

    def test_record_neurons_do_not_change_the_direction(self):
        """記録だけの細胞群は、発火率を残すだけで方向を変えない。"""
        cfg = dataclasses.replace(self.cfg, readout_record=("DNpe017",))
        with_record = fly.read(cfg, self.net, self.counts(0, 1, 3), 0.5)
        without = fly.read(dataclasses.replace(self.cfg, readout_record=None), self.net, self.counts(0, 1, 3), 0.5)
        self.assertEqual(with_record["value"], without["value"])
        self.assertEqual(with_record["record_hz"], {"DNpe017": 6.0})
        self.assertIsNone(without["record_hz"])
        missing = fly.read(dataclasses.replace(self.cfg, readout_record=("lLN1_bc",)), self.net, self.counts(0, 1, 3), 0.5)
        self.assertEqual(missing["record_hz"], {"lLN1_bc": None})
        self.assertEqual(missing["value"], direction.APPROACH)

    def test_select_by_side(self):
        self.assertEqual(list(self.net.body_ids[fly.select(self.net, ["DNp20:L"])]), [DNP20_L])
        self.assertEqual(sorted(self.net.body_ids[fly.select(self.net, ["DNp20"])]), [DNP20_L, DNP20_R])
        with self.assertRaises(fly.FlyError):
            fly.select(self.net, ["LC4"])
        with self.assertRaises(fly.FlyError):
            fly.select(self.net, ["DNpe017:L"])


@unittest.skipIf(np is None, "numpy が無い")
class DecideTest(unittest.TestCase):
    def setUp(self):
        self.dir = TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.map = retina.load_column_map(write_map(self.root, lattice(2) + lattice(2, side="L")))
        self.cfg = dataclasses.replace(helpers.load_config(), direction_source="fly")
        self.frames = (image(self.cfg),)

    def decide(self, net, cfg=None, frames=None):
        return fly.decide(cfg or self.cfg, frames or self.frames, network=net, column_map=self.map, repo_root=self.root)

    def test_right_wired_network_approaches_and_records_observations(self):
        result = self.decide(small_network(self.map, DNP20_R), cfg=dataclasses.replace(self.cfg, readout_record=None))
        self.assertTrue(result.ok)
        self.assertEqual(result.value, direction.APPROACH)
        obs = result.observation
        self.assertGreater(obs["approach_hz"], obs["avoid_hz"])
        self.assertGreaterEqual(obs["gate_spikes"], 1)
        self.assertEqual(obs["cells"]["approach"], [DNP20_R])
        self.assertEqual(obs["cells"]["avoid"], [DNP20_L])
        self.assertEqual(obs["state_from"], "rest")
        self.assertEqual(obs["frames"], 1)
        self.assertIn("approach_hz", direction.Direction(**{**result.__dict__}).as_dict()["observation"])

    def test_left_wired_network_avoids(self):
        self.assertEqual(self.decide(small_network(self.map, DNP20_L)).value, direction.AVOID)

    def test_state_is_carried_between_observations(self):
        net = small_network(self.map, DNP20_R)
        first = self.decide(net)
        second = self.decide(net)
        self.assertEqual(first.observation["state_from"], "rest")
        self.assertEqual(second.observation["state_from"], "carried")
        self.assertTrue((self.root / self.cfg.simulation.checkpoint_path).exists())
        self.assertNotEqual(first.observation["state_sha256"], second.observation["state_sha256"])

    def test_state_of_other_values_is_not_read_and_not_overwritten(self):
        """値が違う網の状態は読まない（別の脳の状態を引き継がない）。消すかどうかは人間が決める。"""
        net = small_network(self.map, DNP20_R)
        self.decide(net)
        path = self.root / self.cfg.simulation.checkpoint_path
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        other = dataclasses.replace(
            self.cfg, simulation=dataclasses.replace(self.cfg.simulation, input_half_saturation=1.0)
        )
        with self.assertRaises(lif.SimulationError):
            self.decide(net, cfg=other)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_frames_are_shown_in_order(self):
        cfg = dataclasses.replace(self.cfg, simulation=dataclasses.replace(self.cfg.simulation, carry_state=False))
        result = self.decide(small_network(self.map, DNP20_R), cfg=cfg, frames=self.frames * 5)
        self.assertEqual(result.observation["frames"], 5)
        self.assertIsNone(result.observation["state_sha256"])
        self.assertFalse((self.root / cfg.simulation.checkpoint_path).exists())

    def test_without_image_or_readout_it_fails(self):
        net = small_network(self.map, DNP20_R)
        with self.assertRaises(fly.FlyError):
            fly.decide(self.cfg, (), network=net, column_map=self.map, repo_root=self.root)
        with self.assertRaises(fly.FlyError):
            self.decide(net, cfg=dataclasses.replace(self.cfg, readout_approach=None))


if __name__ == "__main__":
    unittest.main()
