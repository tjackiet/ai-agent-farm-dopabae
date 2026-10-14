"""学習の場所の測定。重みを変えるのは指定した結合だけで、距離の数えかたが決まった値を返すこと。

網は小さく作る。numpy が無ければ飛ばす。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - 依存が無い環境では飛ばす
    np = None

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import probe_plasticity as plasticity  # noqa: E402

sys.path.pop(0)


def small(modulatory=()):
    """0:KC → 1:MBON07 → 2:X → 3:DNp20、0:KC → 4:MBON01、5:Y → 1:MBON07。"""
    from dopabae import connectome

    edges = [(0, 1, 10), (0, 4, 7), (1, 2, -5), (2, 3, 6), (5, 1, 3)]
    pre, post, signed = zip(*edges)
    indptr, post_sorted, signed_sorted = connectome.build_csr(6, list(pre), list(post), list(signed))
    mask = np.zeros(6, dtype=bool)
    mask[list(modulatory)] = True
    return connectome.Network(
        body_ids=np.arange(1, 7, dtype=np.int64),
        indptr=indptr,
        post=post_sorted,
        signed_count=signed_sorted,
        cell_type=np.array(["KCab", "MBON07", "X", "DNp20", "MBON01", "Y"], dtype=str),
        side=np.array(["R"] * 6, dtype=str),
        modulatory=mask,
        meta={"key": "plasticity-test"},
    )


@unittest.skipIf(np is None, "numpy が無い")
class ScaledTest(unittest.TestCase):
    def weights(self, net):
        out = {}
        for i in range(net.n_neurons):
            for e in range(net.indptr[i], net.indptr[i + 1]):
                out[(i, int(net.post[e]))] = float(net.signed_count[e])
        return out

    def test_only_kc_to_target_is_changed(self):
        net = small()
        new, changed = plasticity.scaled(net, net.type_index(prefix="KC"), net.type_index(("MBON07",)), 0.5)
        self.assertEqual(changed, 1)
        before, after = self.weights(net), self.weights(new)
        self.assertEqual(after[(0, 1)], 5.0)
        for edge in [(0, 4), (1, 2), (2, 3), (5, 1)]:
            self.assertEqual(after[edge], before[edge], edge)
        # 元の網は変えない
        self.assertEqual(before[(0, 1)], 10.0)
        self.assertNotEqual(new.meta["key"], net.meta["key"])

    def test_zero_and_all_mbon(self):
        net = small()
        new, changed = plasticity.scaled(net, net.type_index(prefix="KC"), net.type_index(prefix="MBON"), 0.0)
        self.assertEqual(changed, 2)
        after = self.weights(new)
        self.assertEqual((after[(0, 1)], after[(0, 4)]), (0.0, 0.0))


@unittest.skipIf(np is None, "numpy が無い")
class HopsTest(unittest.TestCase):
    def test_counts_steps(self):
        net = small()
        self.assertEqual(plasticity.hops(net, [1], [3]), 2)
        self.assertEqual(plasticity.hops(net, [0], [3]), 3)
        self.assertIsNone(plasticity.hops(net, [3], [0]))
        self.assertIsNone(plasticity.hops(net, [0], [3], limit=2))

    def test_modulatory_cells_do_not_relay(self):
        """調節性の物質だけを出す細胞の先はたどらない（膜電位を動かさないため）。"""
        self.assertIsNone(plasticity.hops(small(modulatory=(2,)), [1], [3]))


class ConditionsTest(unittest.TestCase):
    def test_conditions_are_described(self):
        self.assertIn("base", plasticity.CONDITIONS)
        self.assertEqual(plasticity.CONDITIONS["zero"][1], {"scale": 0.0})
        self.assertEqual(plasticity.PLASTIC_TYPES, ("MBON07", "MBON11"))

    def test_compartment_conditions_stay_within_stonkfly_bounds(self):
        """区画ごとの条件は 1 つの型だけを、Stonkfly の上下限（0.1〜2.0 倍）の端まで動かす。"""
        for name in ("m07_low", "m07_high", "m11_low", "m11_high"):
            types = plasticity.CONDITIONS[name][1]["scale_types"]
            self.assertEqual(len(types), 1)
            self.assertTrue(set(types) <= set(plasticity.PLASTIC_TYPES))
            self.assertIn(next(iter(types.values())), (0.1, 2.0))


if __name__ == "__main__":
    unittest.main()


class CompareToNullTest(unittest.TestCase):
    def report(self, conditions):
        def obs(dirs, gate=1, axis=4.0):
            return [{"direction": d, "gate_spikes": gate, "axis_hz": axis, "MBON07_hz": 1.0, "MBON11_hz": 2.0,
                     "wall_sec": 0.1 * i} for i, d in enumerate(["NONE", "NONE"] + dirs)]
        return {"conditions": {k: {"observations": obs(*v)} for k, v in conditions.items()}}

    def test_flags_only_what_leaves_the_null_span(self):
        base = ["APPROACH", "NONE", "APPROACH", "NONE"]
        a = self.report({"base": (base,), "tiny": (["NONE", "NONE", "APPROACH", "NONE"],)})
        b = self.report({"base": (base,), "zero": (["APPROACH"] * 4,), "same": (base,)})
        result = plasticity.compare_to_null([a, b])
        self.assertEqual(result["null"], ["base", "tiny"])
        self.assertEqual(result["null_span"]["APPROACH"], (1, 2))
        self.assertIn("APPROACH", result["outside"]["zero"])
        self.assertEqual(result["outside"]["same"], [])
        self.assertEqual(result["conditions"]["same"]["same_as_base_pct"], 100)

    def test_bases_must_agree(self):
        a = self.report({"base": (["APPROACH"] * 4,)})
        b = self.report({"base": (["NONE"] * 4,)})
        with self.assertRaises(SystemExit):
            plasticity.compare_to_null([a, b])
