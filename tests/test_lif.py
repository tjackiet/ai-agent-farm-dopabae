"""LIF シミュレーション。小さな網で、遅延・不応期・符号・決定性を確かめる。

Stonkfly に合わせた振る舞い（光受容細胞への電流、ラミナの背景電流、KC の静止電位と適応、
調節性の細胞、不応期中の入力、状態の引き継ぎ）も確かめる。

配線図データは使わない。numpy が無ければ飛ばす（requirements-connectome.txt）。
"""

from __future__ import annotations

import dataclasses
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dopabae import lif
from tests import helpers

try:
    import numpy as np
except ImportError:  # pragma: no cover - 依存が無い環境では飛ばす
    np = None

from dopabae import connectome


def params() -> lif.Params:
    return lif.Params.from_settings(helpers.load_config().simulation)


def network(
    n: int,
    edges: list[tuple[int, int, int]],
    types: list[str] | None = None,
    modulatory: list[int] = (),
) -> connectome.Network:
    """(発火元, 行き先, 符号つきシナプス数) から網を作る。型名は既定で Mi1（特別扱いなし）。"""
    pre = [e[0] for e in edges]
    post = [e[1] for e in edges]
    signed = [e[2] for e in edges]
    indptr, post_sorted, signed_sorted = connectome.build_csr(n, pre, post, signed)
    mask = np.zeros(n, dtype=bool)
    mask[list(modulatory)] = True
    return connectome.Network(
        body_ids=np.arange(1, n + 1, dtype=np.int64) * 10,
        indptr=indptr,
        post=post_sorted,
        signed_count=signed_sorted,
        cell_type=np.array(types or ["Mi1"] * n, dtype=str),
        side=np.array(["R"] * n, dtype=str),
        modulatory=mask,
        meta={"key": f"test-{n}-{len(edges)}"},
    )


def light(index: list[int], value: float) -> lif.Light:
    return lif.Light(index=np.array(index, dtype=np.int64), value=np.full(len(index), value, dtype=np.float32))


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

    def test_adaptation_matches_closed_form(self):
        """KC の適応 w が u を下げる量も厳密解と一致する。"""
        p = params()
        a, _, _ = p.coefficients()
        d, e = p.adaptation_coefficients()
        tm, ta = p.membrane_tau_ms, p.kc_adaptation_tau_ms
        u, w, w0 = 0.0, 8.0, 8.0
        for k in range(1, 400):
            u, w = a * u - d * w, e * w
            t = k * p.dt_ms
            exact = -w0 * ta / (ta - tm) * (math.exp(-t / ta) - math.exp(-t / tm))
            self.assertAlmostEqual(u, exact, places=9)

    def test_current_steady_state(self):
        """一定電流 I だけなら u は I へ近づく（u' = a·u + (1 − a)·I）。"""
        p = params()
        a, _, _ = p.coefficients()
        u = 0.0
        for _ in range(20_000):
            u = a * u + (1 - a) * 12.0
        self.assertAlmostEqual(u, 12.0, places=6)

    def test_photoreceptor_current_saturates(self):
        p = params()
        self.assertEqual(p.photoreceptor_current(0.0), 0.0)
        self.assertAlmostEqual(p.photoreceptor_current(1.0), 30 * 1 / 1.02)
        self.assertLess(p.photoreceptor_current(1.0), p.input_max_mv)

    def test_steps(self):
        p = params()
        self.assertEqual(p.delay_steps, 18)  # 1.8 ms / 0.1 ms
        self.assertEqual(p.refractory_steps, 22)  # 2.2 ms / 0.1 ms
        self.assertEqual(p.chunk_steps, 100)  # 10 ms
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
        result = lif.simulate(net, self.p, 50, drive=drive([0], 500.0), seed=1)
        self.assertGreater(result.spike_counts[0], 0)
        self.assertGreater(result.spike_counts[1], 0)
        self.assertEqual(result.spike_counts[2], 0)

    def test_nothing_arrives_before_the_delay(self):
        """発火は遅延（18 刻み）のあとでしか届かない。重みがどれほど大きくても。"""
        net = network(2, [(0, 1, 10_000)])
        short = lif.simulate(net, self.p, self.p.delay_ms, drive=drive([0], 10_000.0), seed=1)
        self.assertGreater(short.spike_counts[0], 0)
        self.assertEqual(short.spike_counts[1], 0)
        longer = lif.simulate(net, self.p, 10.0, drive=drive([0], 10_000.0), seed=1)
        self.assertGreater(longer.spike_counts[1], 0)

    def test_refractory_caps_the_rate(self):
        """毎刻み入力しても、不応期より速くは発火しない。"""
        net = network(1, [])
        result = lif.simulate(net, self.p, 100.0, drive=drive([0], 10_000.0), seed=1)
        steps = self.p.steps_for(100.0)
        self.assertLessEqual(result.spike_counts[0], steps // self.p.refractory_steps + 1)
        self.assertGreater(result.spike_counts[0], 0)

    def test_same_seed_same_result(self):
        net = network(4, [(0, 1, 30), (1, 2, 30), (2, 3, -30), (3, 0, 30)])
        a = lif.simulate(net, self.p, 100, drive=drive([0, 2], 80.0), seed=7)
        b = lif.simulate(net, self.p, 100, drive=drive([0, 2], 80.0), seed=7)
        self.assertTrue(np.array_equal(a.spike_counts, b.spike_counts))
        self.assertEqual(a.input_spikes, b.input_spikes)

    def test_different_seed_changes_input(self):
        net = network(1, [])
        a = lif.simulate(net, self.p, 200, drive=drive([0], 50.0), seed=1)
        b = lif.simulate(net, self.p, 200, drive=drive([0], 50.0), seed=2)
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
                lif.simulate(net, self.p, 10, drive=bad)

    def test_empty_network_is_refused(self):
        with self.assertRaises(lif.SimulationError):
            lif.simulate(network(0, []), self.p, 10)

    def test_light_drives_and_darkness_does_not(self):
        net = network(2, [])
        lit = lif.simulate(net, self.p, 50, light=light([0], 1.0))
        dark = lif.simulate(net, self.p, 50, light=light([0], 0.0))
        self.assertGreater(lit.spike_counts[0], 0)
        self.assertEqual(dark.total_spikes, 0)

    def test_light_is_smoothed_per_chunk(self):
        """明るさは区切り（10 ms）ごとに時定数 10 ms で近づく。1 区切りで 1 − e^-1。"""
        net = network(1, [])
        result = lif.simulate(net, self.p, 10, light=light([0], 1.0))
        self.assertAlmostEqual(float(result.state.light[0]), 1 - math.exp(-1), places=6)

    def test_lamina_fires_without_input(self):
        """背景の電流（12 mV）は閾値までの 7 mV を超えるので、ラミナは何も見なくても発火する。"""
        net = network(2, [], types=["L1", "Mi1"])
        result = lif.simulate(net, self.p, 100)
        self.assertGreater(result.spike_counts[0], 0)
        self.assertEqual(result.spike_counts[1], 0)
        quiet = lif.simulate(net, dataclasses.replace(self.p, lamina_bias_mv=0.0), 100)
        self.assertEqual(quiet.total_spikes, 0)

    def test_kc_rests_lower(self):
        """10 mV ぶんの電流は、静止 −52 mV の細胞を発火させ、−60 mV の KC は発火させない。"""
        net = network(2, [], types=["Mi1", "KCg-m"])
        level = 0.01  # 30 × 0.01 / 0.03 = 10 mV
        result = lif.simulate(net, self.p, 100, light=light([0, 1], level))
        self.assertGreater(result.spike_counts[0], 0)
        self.assertEqual(result.spike_counts[1], 0)

    def test_kc_adaptation_slows_firing(self):
        net = network(1, [], types=["KCg-m"])
        adapted = lif.simulate(net, self.p, 200, light=light([0], 1.0))
        plain = lif.simulate(net, dataclasses.replace(self.p, kc_adaptation_mv=0.0), 200, light=light([0], 1.0))
        self.assertGreater(plain.spike_counts[0], adapted.spike_counts[0])
        self.assertGreater(adapted.spike_counts[0], 0)

    def test_modulatory_cells_do_not_move_targets(self):
        """ドーパミンなどの細胞の発火は、行き先の膜電位を動かさない（結合は残る）。"""
        fast = lif.simulate(network(2, [(0, 1, 1000)]), self.p, 50, light=light([0], 1.0))
        slow = lif.simulate(network(2, [(0, 1, 1000)], modulatory=[0]), self.p, 50, light=light([0], 1.0))
        self.assertGreater(fast.spike_counts[1], 0)
        self.assertGreater(slow.spike_counts[0], 0)
        self.assertEqual(slow.spike_counts[1], 0)

    def test_input_during_refractory_is_dropped_or_kept(self):
        net = network(2, [])
        state = lif.initial_state(net, self.p)
        state.ready_at[1] = 5  # 不応期中
        state.pending[0, 1] = 100.0  # この刻みに届く入力
        dropped = lif.simulate(net, self.p, self.p.dt_ms, state=state)
        kept = lif.simulate(net, dataclasses.replace(self.p, refractory_input="keep"), self.p.dt_ms, state=state)
        self.assertEqual(float(dropped.state.x[1]), 0.0)
        self.assertEqual(float(kept.state.x[1]), 100.0)

    def test_carrying_state_equals_one_long_run(self):
        """50 ms を 2 回続けるのと、100 ms を 1 回で回すのは同じになる。"""
        net = network(3, [(0, 1, 40), (1, 2, -20), (2, 0, 30)], types=["L1", "Mi1", "KCg-m"])
        whole = lif.simulate(net, self.p, 100, light=light([2], 0.5))
        first = lif.simulate(net, self.p, 50, light=light([2], 0.5))
        second = lif.simulate(net, self.p, 50, light=light([2], 0.5), state=first.state)
        self.assertTrue(np.array_equal(whole.spike_counts, first.spike_counts + second.spike_counts))
        self.assertEqual(whole.state.clock, second.state.clock)
        self.assertTrue(np.array_equal(whole.state.u, second.state.u))
        self.assertTrue(np.array_equal(whole.state.pending, second.state.pending))

    def test_given_state_is_not_modified(self):
        net = network(1, [])
        state = lif.initial_state(net, self.p)
        lif.simulate(net, self.p, 20, light=light([0], 1.0), state=state)
        self.assertEqual(state.clock, 0)
        self.assertEqual(float(state.u[0]), 0.0)

    def test_bad_light_is_refused(self):
        net = network(2, [])
        for bad in (
            lif.Light(np.array([5]), np.array([0.5], dtype=np.float32)),  # 網の外
            lif.Light(np.array([0, 0]), np.array([0.5, 0.5], dtype=np.float32)),  # 重複
            lif.Light(np.array([0]), np.array([np.nan], dtype=np.float32)),  # 有限でない
        ):
            with self.assertRaises(lif.SimulationError):
                lif.simulate(net, self.p, 10, light=bad)

    def test_result_holds_only_observations(self):
        net = network(2, [(0, 1, 100)])
        result = lif.simulate(net, self.p, 20, drive=drive([0], 500.0), seed=1)
        self.assertEqual(result.steps, 200)
        self.assertAlmostEqual(result.duration_ms, 20.0)
        self.assertEqual(result.active_neurons, int((result.spike_counts > 0).sum()))
        self.assertEqual(result.state.clock, 200)


@unittest.skipIf(np is None, "numpy が無い")
class StateFileTest(unittest.TestCase):
    def setUp(self):
        self.p = params()
        self.dir = TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "state.npz"

    def test_round_trip(self):
        net = network(3, [(0, 1, 40)], types=["L1", "Mi1", "KCg-m"])
        result = lif.simulate(net, self.p, 30, light=light([2], 1.0))
        digest = lif.save_state(result.state, self.path, net, self.p)
        self.assertEqual(len(digest), 64)
        loaded = lif.load_state(self.path, net, self.p)
        self.assertEqual(loaded.clock, result.state.clock)
        for name in ("u", "x", "adaptation", "ready_at", "pending", "light"):
            self.assertTrue(np.array_equal(getattr(loaded, name), getattr(result.state, name)), name)

    def test_other_network_is_refused(self):
        """別の網・別の値の状態は引き継がない。"""
        net = network(2, [])
        lif.save_state(lif.initial_state(net, self.p), self.path, net, self.p)
        with self.assertRaises(lif.SimulationError):
            lif.load_state(self.path, network(2, [(0, 1, 1)]), self.p)
        with self.assertRaises(lif.SimulationError):
            lif.load_state(self.path, net, dataclasses.replace(self.p, lamina_bias_mv=1.0))

    def test_missing_file_is_refused(self):
        with self.assertRaises(lif.SimulationError):
            lif.load_state(self.path, network(1, []), self.p)


if __name__ == "__main__":
    unittest.main()
