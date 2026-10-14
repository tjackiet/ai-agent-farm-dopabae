"""学習の規則（Phase 5）。Stonkfly の rule.py と同じ式で、発火の順序で弱めも強めもすること。

配線図データは使わない。小さな網を作って確かめる。numpy が無ければ飛ばす。
"""

from __future__ import annotations

import dataclasses
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import numpy as np
except ImportError:  # pragma: no cover - 依存が無い環境では飛ばす
    np = None

if np is not None:
    from dopabae import connectome, lif, plasticity

COMPARTMENTS = (("PAM11", "MBON07"), ("PPL101", "MBON11"))
# 細胞: KC 3 個、MBON07 2 個、MBON11 1 個、PAM11 2 個、PPL101 1 個
TYPES = ["KCg", "KCg", "KCab", "MBON07", "MBON07", "MBON11", "PAM11", "PAM11", "PPL101"]
EDGES = [
    (0, 3, 5), (1, 3, 3), (1, 4, 2), (2, 5, 4),  # KC → MBON（学習する）
    (6, 3, 6), (7, 3, 2), (6, 4, 1), (7, 4, 3),  # PAM11 → MBON07（接触）
    (8, 5, 5),  # PPL101 → MBON11
    (0, 6, 1),  # KC → PAM11（学習しない）
]


def network(edges=EDGES):
    n = len(TYPES)
    pre, post, signed = zip(*edges)
    indptr, post_sorted, signed_sorted = connectome.build_csr(n, list(pre), list(post), list(signed))
    return connectome.Network(
        body_ids=np.arange(100, 100 + n, dtype=np.int64),
        indptr=indptr,
        post=post_sorted,
        signed_count=signed_sorted,
        cell_type=np.array(TYPES, dtype=str),
        side=np.array(["R"] * n, dtype=str),
        modulatory=np.array([t in ("PAM11", "PPL101") for t in TYPES]),
        meta={"key": "plasticity-test"},
    )


def params(**changes):
    base = plasticity.RuleParams(
        eta=0.001, trace_sec=1.0, filter_sec=0.05, memory_decay_sec=1800.0,
        min_fraction=0.1, max_fraction=2.0, dan_baseline_hz=0.0,
    )
    return dataclasses.replace(base, **changes)


@unittest.skipIf(np is None, "numpy が無い")
class IdentifyTest(unittest.TestCase):
    def test_edges_and_contact_fractions(self):
        net = network()
        circuit = plasticity.identify(net, COMPARTMENTS)
        self.assertEqual(len(circuit.edges), 4)  # KC→PAM11 は入らない
        self.assertEqual(sorted(net.post[circuit.edges].tolist()), [3, 3, 4, 5])
        self.assertEqual(circuit.dan_cells.tolist(), [6, 7, 8])
        # 各結合の列の和は 1。区画の外の DAN は 0
        self.assertTrue(np.allclose(circuit.gain.sum(axis=0), 1.0))
        for column, target in enumerate(circuit.edge_mbon):
            if target == 3:
                self.assertEqual(circuit.gain[:, column].tolist(), [0.75, 0.25, 0.0])
            elif target == 4:
                self.assertEqual(circuit.gain[:, column].tolist(), [0.25, 0.75, 0.0])
            else:
                self.assertEqual(circuit.gain[:, column].tolist(), [0.0, 0.0, 1.0])

    def test_missing_contact_is_an_error(self):
        edges = [e for e in EDGES if e[0] != 8]  # PPL101 → MBON11 を外す
        with self.assertRaises(plasticity.PlasticityError):
            plasticity.identify(network(edges), COMPARTMENTS)

    def test_missing_type_is_an_error(self):
        with self.assertRaises(plasticity.PlasticityError):
            plasticity.identify(network(), (("PAM99", "MBON07"),))


@unittest.skipIf(np is None, "numpy が無い")
class RuleTest(unittest.TestCase):
    def setUp(self):
        self.circuit = plasticity.identify(network(), COMPARTMENTS)

    def run_bins(self, memory, kc_rates, dan_rates, p=None, h=0.01):
        for kc_rate, dan_rate in zip(kc_rates, dan_rates):
            plasticity.advance(memory, self.circuit, p or params(), np.asarray(kc_rate, float), np.asarray(dan_rate, float), h)
        return memory

    def test_constant_rates_do_not_learn(self):
        """発火率が一定なら drive は 0（中心化）。重みは変わらない。"""
        memory = plasticity.Memory.fresh(self.circuit)
        self.run_bins(memory, [[50, 30, 20]] * 200, [[40, 10, 5]] * 200)
        self.assertLess(abs(memory.w).max(), 1e-12)

    def test_kc_before_dopamine_weakens_and_after_strengthens(self):
        """KC が先・ドーパミンが後なら弱まり、ドーパミンが先・KC が後なら強まる。"""
        quiet = [[0, 0, 0]] * 5
        kc_first = self.run_bins(
            plasticity.Memory.fresh(self.circuit),
            [[100, 100, 100]] * 5 + quiet + quiet,
            quiet + [[200, 200, 200]] * 5 + quiet,
        )
        dan_first = self.run_bins(
            plasticity.Memory.fresh(self.circuit),
            quiet + [[100, 100, 100]] * 5 + quiet,
            [[200, 200, 200]] * 5 + quiet + quiet,
        )
        self.assertTrue((kc_first.efficacy() < 1).all())
        self.assertTrue((dan_first.efficacy() > 1).all())

    def test_only_the_paired_compartment_changes(self):
        """PAM11 だけが発火すれば、KC→MBON07 だけが変わり、KC→MBON11 は変わらない。"""
        memory = self.run_bins(
            plasticity.Memory.fresh(self.circuit),
            [[100, 100, 100]] * 5 + [[0, 0, 0]] * 5,
            [[0, 0, 0]] * 5 + [[200, 200, 0]] * 5,
        )
        to_mbon11 = self.circuit.edge_mbon == 5
        self.assertTrue((memory.w[~to_mbon11] < 0).all())
        self.assertTrue((memory.w[to_mbon11] == 0).all())

    def test_bounds(self):
        memory = self.run_bins(
            plasticity.Memory.fresh(self.circuit),
            [[1000, 1000, 1000]] * 20 + [[0, 0, 0]] * 20,
            [[0, 0, 0]] * 20 + [[1000, 1000, 1000]] * 20,
            p=params(eta=10.0),
        )
        self.assertAlmostEqual(float(memory.efficacy().min()), 0.1)
        memory = self.run_bins(
            plasticity.Memory.fresh(self.circuit),
            [[0, 0, 0]] * 20 + [[1000, 1000, 1000]] * 20,
            [[1000, 1000, 1000]] * 20 + [[0, 0, 0]] * 20,
            p=params(eta=10.0),
        )
        self.assertAlmostEqual(float(memory.efficacy().max()), 2.0)

    def test_memory_returns_to_baseline(self):
        """何も起きなければ、記憶は memory_decay_sec の時定数で元へ戻る。"""
        memory = plasticity.Memory.fresh(self.circuit)
        memory.u[:] = 0.5
        memory.w[:] = 0.5
        self.run_bins(memory, [[0, 0, 0]] * 100, [[0, 0, 0]] * 100, p=params(memory_decay_sec=1.0))
        self.assertTrue(np.allclose(memory.u, 0.5 * math.exp(-1.0)))
        self.assertTrue(np.allclose(memory.w, memory.u, atol=0.02))  # 効き目は 50 ms 遅れで追う

    def test_closed_form_matches_fine_integration(self):
        """u と w の 1 区切りの更新が、微分方程式を細かく積分した値と合う。"""
        p = params(memory_decay_sec=0.3, filter_sec=0.05, eta=1.0)
        memory = plasticity.Memory.fresh(self.circuit)
        memory.u[:] = 0.2
        memory.w[:] = -0.1
        memory.kc_trace[:] = 3.0
        memory.dan_trace[:] = 1.0
        kc_rate, dan_rate = np.array([1.0, 2.0, 0.5]), np.array([0.5, 0.0, 1.5])
        # 痕跡の中点は advance と同じに作り、drive を区切りのあいだ一定とみなして積分する
        half = math.sqrt(math.exp(-0.01))
        k_mid = memory.kc_trace * half + kc_rate * (1 - half)
        d_mid = memory.dan_trace * half + dan_rate * (1 - half)
        drive = p.eta * (kc_rate[self.circuit.edge_kc] * (self.circuit.gain.T @ d_mid)
                         - (self.circuit.gain.T @ dan_rate) * k_mid[self.circuit.edge_kc])
        u, w = memory.u.copy(), memory.w.copy()
        steps = 100000
        dt = 0.01 / steps
        for _ in range(steps):
            u, w = u + dt * (-u / p.memory_decay_sec + drive), w + dt * (u - w) / p.filter_sec
        plasticity.advance(memory, self.circuit, p, kc_rate, dan_rate, 0.01)
        self.assertTrue(np.allclose(memory.u, u, atol=1e-5))
        self.assertTrue(np.allclose(memory.w, w, atol=1e-5))

    def test_bin_longer_than_10ms_is_refused(self):
        with self.assertRaises(plasticity.PlasticityError):
            plasticity.advance(plasticity.Memory.fresh(self.circuit), self.circuit, params(), np.zeros(3), np.zeros(3), 0.02)


@unittest.skipIf(np is None, "numpy が無い")
class LearnerTest(unittest.TestCase):
    def test_simulation_writes_learned_weights(self):
        """lif.simulate に渡すと、区切りごとに重みが書き換わり、記憶が残る。渡さなければ何も変わらない。"""
        net = network()
        circuit = plasticity.identify(net, COMPARTMENTS)
        lif_params = lif.Params(
            dt_ms=0.1, membrane_tau_ms=20.0, synaptic_tau_ms=5.0, resting_mv=-52.0, reset_mv=-52.0,
            threshold_mv=-45.0, refractory_ms=2.2, delay_ms=1.8, weight_per_synapse_mv=0.275,
        )
        memory = plasticity.Memory.fresh(circuit)
        learner = plasticity.Learner(circuit, params(eta=1.0), memory, plasticity.base_weight(net, circuit, 0.275))
        kc = np.array([0, 1, 2])
        dans = np.array([6, 7, 8])
        # KC を先に 50 ms 発火させ、続けてドーパミン細胞を発火させる
        state = lif.simulate(net, lif_params, 50, stimulus=lif.Stimulus(index=kc, current_mv=30.0), learner=learner).state
        lif.simulate(net, lif_params, 50, stimulus=lif.Stimulus(index=dans, current_mv=30.0), state=state, learner=learner)
        self.assertTrue((memory.efficacy() < 1).any())
        self.assertGreater(memory.summary(params())["changed_edges"], 0)


@unittest.skipIf(np is None, "numpy が無い")
class MemoryFileTest(unittest.TestCase):
    def setUp(self):
        self.dir = TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "memory.npz"
        self.net = network()
        self.circuit = plasticity.identify(self.net, COMPARTMENTS)

    def test_round_trip(self):
        memory = plasticity.Memory.fresh(self.circuit)
        memory.w[:] = 0.25
        memory.observations = 7
        plasticity.save_memory(memory, self.path, self.net, self.circuit, params())
        loaded = plasticity.load_memory(self.path, self.net, self.circuit, params())
        self.assertEqual(loaded.observations, 7)
        self.assertTrue(np.allclose(loaded.efficacy(), 1.25))

    def test_other_rule_or_circuit_is_not_read(self):
        """規則の値か学習の場所が違えば読まない（別の記憶を引き継がない）。"""
        plasticity.save_memory(plasticity.Memory.fresh(self.circuit), self.path, self.net, self.circuit, params())
        with self.assertRaises(plasticity.PlasticityError):
            plasticity.load_memory(self.path, self.net, self.circuit, params(max_fraction=3.0))
        other = plasticity.identify(network(EDGES + [(2, 3, 1)]), COMPARTMENTS)
        with self.assertRaises(plasticity.PlasticityError):
            plasticity.load_memory(self.path, self.net, other, params())


if __name__ == "__main__":
    unittest.main()
