"""LIF シミュレーション。小さな網で、遅延・不応期・符号・決定性を確かめる。

配線図データは使わない。numpy が無ければ飛ばす（requirements-connectome.txt）。
"""

from __future__ import annotations

import math
import unittest

from dopabae import lif
from tests import helpers

try:
    import numpy as np
except ImportError:  # pragma: no cover - 依存が無い環境では飛ばす
    np = None

from dopabae import connectome


def params() -> lif.Params:
    return lif.Params.from_settings(helpers.load_config().simulation)


def network(n: int, edges: list[tuple[int, int, int]]) -> connectome.Network:
    """(発火元, 行き先, 符号つきシナプス数) から網を作る。"""
    pre = [e[0] for e in edges]
    post = [e[1] for e in edges]
    signed = [e[2] for e in edges]
    indptr, post_sorted, signed_sorted = connectome.build_csr(n, pre, post, signed)
    return connectome.Network(
        body_ids=np.arange(1, n + 1, dtype=np.int64) * 10,
        indptr=indptr,
        post=post_sorted,
        signed_count=signed_sorted,
        meta={},
    )


def drive(index: list[int], rate_hz: float, weight_mv: float = 68.75) -> lif.Drive:
    return lif.Drive(
        index=np.array(index, dtype=np.int64),
        rate_hz=np.full(len(index), rate_hz),
        weight_mv=weight_mv,
    )


class CoefficientTest(unittest.TestCase):
    """1 刻みの式が厳密解と一致すること（numpy は要らない）。"""

    def test_recurrence_matches_closed_form(self):
        p = params()
        a, b, c = p.coefficients()
        tm, ts = p.membrane_tau_ms, p.synaptic_tau_ms
        u, x, x0 = 0.0, 50.0, 50.0
        for k in range(1, 400):
            u, x = a * u + c * x, b * x
            t = k * p.dt_ms
            exact = x0 * ts / (ts - tm) * (math.exp(-t / ts) - math.exp(-t / tm))
            self.assertAlmostEqual(u, exact, places=9)

    def test_steps(self):
        p = params()
        self.assertEqual(p.delay_steps, 18)  # 1.8 ms / 0.1 ms
        self.assertEqual(p.refractory_steps, 22)  # 2.2 ms / 0.1 ms
        self.assertEqual(p.steps_for(500), 5000)


@unittest.skipIf(np is None, "numpy が無い")
class SimulateTest(unittest.TestCase):
    def setUp(self):
        self.p = params()

    def test_silent_without_input(self):
        """入力が無ければ静止電位のまま。何も発火しない。"""
        net = network(3, [(0, 1, 100), (1, 2, 100)])
        result = lif.simulate(net, self.p, 50)
        self.assertEqual(result.total_spikes, 0)

    def test_excitatory_edge_drives_and_inhibitory_does_not(self):
        net = network(3, [(0, 1, 100), (0, 2, -100)])
        result = lif.simulate(net, self.p, 50, drive([0], 500.0), seed=1)
        self.assertGreater(result.spike_counts[0], 0)
        self.assertGreater(result.spike_counts[1], 0)
        self.assertEqual(result.spike_counts[2], 0)

    def test_nothing_arrives_before_the_delay(self):
        """発火は遅延（18 刻み）のあとでしか届かない。重みがどれほど大きくても。"""
        net = network(2, [(0, 1, 10_000)])
        short = lif.simulate(net, self.p, self.p.delay_ms, drive([0], 10_000.0), seed=1)
        self.assertGreater(short.spike_counts[0], 0)
        self.assertEqual(short.spike_counts[1], 0)
        longer = lif.simulate(net, self.p, 10.0, drive([0], 10_000.0), seed=1)
        self.assertGreater(longer.spike_counts[1], 0)

    def test_refractory_caps_the_rate(self):
        """毎刻み入力しても、不応期より速くは発火しない。"""
        net = network(1, [])
        result = lif.simulate(net, self.p, 100.0, drive([0], 10_000.0), seed=1)
        steps = self.p.steps_for(100.0)
        self.assertLessEqual(result.spike_counts[0], steps // self.p.refractory_steps + 1)
        self.assertGreater(result.spike_counts[0], 0)

    def test_same_seed_same_result(self):
        net = network(4, [(0, 1, 30), (1, 2, 30), (2, 3, -30), (3, 0, 30)])
        a = lif.simulate(net, self.p, 100, drive([0, 2], 80.0), seed=7)
        b = lif.simulate(net, self.p, 100, drive([0, 2], 80.0), seed=7)
        self.assertTrue(np.array_equal(a.spike_counts, b.spike_counts))
        self.assertEqual(a.input_spikes, b.input_spikes)

    def test_different_seed_changes_input(self):
        net = network(1, [])
        a = lif.simulate(net, self.p, 200, drive([0], 50.0), seed=1)
        b = lif.simulate(net, self.p, 200, drive([0], 50.0), seed=2)
        self.assertNotEqual(a.input_spikes, b.input_spikes)

    def test_bad_drive_is_refused(self):
        net = network(2, [])
        for bad in (
            lif.Drive(np.array([5]), np.array([10.0]), 1.0),  # 網の外
            lif.Drive(np.array([0]), np.array([-1.0]), 1.0),  # 負の発火率
            lif.Drive(np.array([0]), np.array([20_000.0]), 1.0),  # 1 刻みに 2 発以上
            lif.Drive(np.array([0, 1]), np.array([1.0]), 1.0),  # 数が合わない
        ):
            with self.assertRaises(lif.SimulationError):
                lif.simulate(net, self.p, 10, bad)

    def test_empty_network_is_refused(self):
        with self.assertRaises(lif.SimulationError):
            lif.simulate(network(0, []), self.p, 10)

    def test_result_holds_only_observations(self):
        net = network(2, [(0, 1, 100)])
        result = lif.simulate(net, self.p, 20, drive([0], 500.0), seed=1)
        self.assertEqual(result.steps, 200)
        self.assertAlmostEqual(result.duration_ms, 20.0)
        self.assertEqual(result.active_neurons, int((result.spike_counts > 0).sum()))


if __name__ == "__main__":
    unittest.main()
