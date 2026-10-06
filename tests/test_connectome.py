"""配線図から網を作ること。細胞の選びかた・閾値・符号・キャッシュの扱い。

配線図データは使わない。小さな feather をテスト側で作る。
numpy / pyarrow が無ければ、それを使うテストだけ飛ばす（requirements-connectome.txt）。
"""

from __future__ import annotations

import dataclasses
import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dopabae import connectome
from dopabae.config import ConnectomeFile
from tests import helpers

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
try:
    import pyarrow as pa
    import pyarrow.feather as feather
except ImportError:  # pragma: no cover
    pa = None


class SignTest(unittest.TestCase):
    INHIBITORY = ("gaba", "glutamate", "histamine")

    def test_inhibitory_transmitters(self):
        for name in self.INHIBITORY:
            self.assertEqual(connectome.sign_of(name, self.INHIBITORY), -1)

    def test_everything_else_is_excitatory(self):
        """Stonkfly と同じく、不明・欠損・ドーパミンなども興奮として扱う。"""
        for name in ("acetylcholine", "dopamine", "unclear", None, float("nan")):
            self.assertEqual(connectome.sign_of(name, self.INHIBITORY), 1)


class SpecTest(unittest.TestCase):
    def setUp(self):
        self.cfg = helpers.load_config()

    def test_spec_follows_stonkfly(self):
        spec = connectome.spec_from_config(self.cfg)
        self.assertEqual(spec.neuron_policy, "assigned_superclass")
        self.assertIn(("R8", "aMe12"), spec.excitatory_overrides)
        self.assertIn("dopamine", spec.modulatory_transmitters)

    def test_override_can_be_turned_off(self):
        sim = dataclasses.replace(self.cfg.simulation, r8_to_ame12_excitatory=False)
        spec = connectome.spec_from_config(dataclasses.replace(self.cfg, simulation=sim))
        self.assertEqual(spec.excitatory_overrides, ())
        self.assertNotEqual(spec.key(), connectome.spec_from_config(self.cfg).key())

    def test_key_changes_with_threshold_and_inputs(self):
        base = connectome.spec_from_config(self.cfg, None)
        self.assertNotEqual(base.key(), connectome.spec_from_config(self.cfg, 3).key())
        other = dataclasses.replace(base, inputs_sha256=(("weights", "0" * 64),))
        self.assertNotEqual(base.key(), other.key())
        self.assertEqual(base.key(), connectome.spec_from_config(self.cfg, None).key())

    def test_bad_threshold_is_refused(self):
        with self.assertRaises(connectome.ConnectomeError):
            connectome.spec_from_config(self.cfg, 0)


@unittest.skipIf(np is None, "numpy が無い")
class CsrTest(unittest.TestCase):
    def test_edges_are_grouped_by_source(self):
        indptr, post, signed = connectome.build_csr(3, [2, 0, 2, 1], [0, 1, 1, 2], [5, -3, 7, 1])
        self.assertEqual(indptr.tolist(), [0, 1, 2, 4])
        self.assertEqual(post.tolist(), [1, 2, 0, 1])
        self.assertEqual(signed.tolist(), [-3, 1, 5, 7])

    def network(self, types=("Mi1", "KCg-m", "L1")):
        n = len(types)
        return connectome.Network(
            body_ids=np.arange(1, n + 1, dtype=np.int64) * 10, indptr=np.zeros(n + 1, dtype=np.int64),
            post=np.zeros(0, dtype=np.int32), signed_count=np.zeros(0, dtype=np.int32),
            cell_type=np.array(types, dtype=str), side=np.array(["R"] * n, dtype=str),
            modulatory=np.zeros(n, dtype=bool), meta={},
        )

    def test_index_of_marks_missing(self):
        net = self.network()
        self.assertEqual(net.index_of([30, 10, 25, 99]).tolist(), [2, 0, -1, -1])

    def test_type_index(self):
        net = self.network(("Mi1", "KCg-m", "L1", "KCab-s", "L2"))
        self.assertEqual(net.type_index(prefix="KC").tolist(), [1, 3])
        self.assertEqual(net.type_index(("L1", "L2")).tolist(), [2, 4])
        self.assertEqual(net.type_index(("L1",), prefix="KC").tolist(), [1, 2, 3])

    def test_save_and_load_round_trip(self):
        cfg = helpers.load_config()
        spec = connectome.spec_from_config(cfg, None)
        indptr, post, signed = connectome.build_csr(2, [0], [1], [4])
        net = connectome.Network(
            np.array([1, 2]), indptr, post, signed, np.array(["R1-R6", "L1"]), np.array(["L", "R"]),
            np.array([False, True]), {"version": connectome.COMPILED_VERSION, "key": spec.key()},
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "net.npz"
            connectome.save(net, path)
            loaded = connectome.load(path, spec)
            self.assertEqual(loaded.post.tolist(), [1])
            self.assertEqual(loaded.cell_type.tolist(), ["R1-R6", "L1"])
            self.assertEqual(loaded.side.tolist(), ["L", "R"])
            self.assertEqual(loaded.modulatory.tolist(), [False, True])
            # 作りかたが違えば読まない
            with self.assertRaises(connectome.ConnectomeError):
                connectome.load(path, connectome.spec_from_config(cfg, 5))


@unittest.skipIf(np is None or pa is None, "numpy / pyarrow が無い")
class CompileTest(unittest.TestCase):
    """小さな flat-connectome から網を作る。"""

    def setUp(self):
        self.dir = TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        root = Path(self.dir.name)
        # 網に入るのは superclass があって Glia でないもの（Stonkfly と同じ）。
        # 1 Mi1・ACh / 2 Mi4・GABA / 3 superclass なし（入らない）/ 4 R1-R6・status 空・ヒスタミン /
        # 5 Tm20・伝達物質の表に無い / 6 Glia（入らない）/ 7 PAM11・ドーパミン（調節性）/
        # 8 R8p・ヒスタミン / 9 aMe12・ACh
        annotations = pa.table({
            "bodyId": pa.array([1, 2, 3, 4, 5, 6, 7, 8, 9], type=pa.int64()),
            "type": ["Mi1", "Mi4", None, "R1-R6", "Tm20", None, "PAM11", "R8p", "aMe12"],
            "status": ["Traced", "Traced", "Orphan", None, "Traced", "Glia", "Traced", "Traced", "Traced"],
            "superclass": ["ol_intrinsic", "ol_intrinsic", None, "ol_sensory", "ol_intrinsic",
                           "glia", "cb_intrinsic", "ol_sensory", "ol_intrinsic"],
            "instance": ["Mi1_R", "Mi4_R", None, "R1-R6_L", "Tm20_R", None, "PAM11_L", "R8p_R", "aMe12_R"],
        })
        weights = pa.table({
            "body_pre": pa.array([1, 2, 1, 3, 4, 5, 1, 7, 8, 8, 6], type=pa.int64()),
            "body_post": pa.array([2, 1, 3, 1, 1, 1, 5, 1, 9, 1, 1], type=pa.int64()),
            "weight": pa.array([6, 4, 9, 9, 2, 3, 1, 5, 4, 2, 7], type=pa.int64()),
        })
        transmitters = pa.table({
            "body": pa.array([1, 2, 3, 4, 7, 8, 9], type=pa.int64()),
            "consensus_nt": ["acetylcholine", "gaba", "acetylcholine", "histamine", "dopamine",
                             "histamine", "acetylcholine"],
        })
        files = {}
        for name, table in (("annotations", annotations), ("weights", weights), ("neurotransmitters", transmitters)):
            path = root / f"{name}.feather"
            feather.write_feather(table, path)
            blob = path.read_bytes()
            files[name] = ConnectomeFile(path=path.name, sha256=hashlib.sha256(blob).hexdigest(), size_bytes=len(blob))
        self.cfg = dataclasses.replace(helpers.load_config(), connectome_local_dir=str(root), connectome_files=files)

    def compile(self, threshold):
        return connectome.compile_network(self.cfg, connectome.spec_from_config(self.cfg, threshold))

    def edges(self, net) -> set[tuple[int, int, int]]:
        out = set()
        for i in range(net.n_neurons):
            for k in range(net.indptr[i], net.indptr[i + 1]):
                out.add((int(net.body_ids[i]), int(net.body_ids[net.post[k]]), int(net.signed_count[k])))
        return out

    def test_cells_have_superclass_and_are_not_glia(self):
        """status が空の R1-R6（4）は入り、superclass の無い 3 と Glia の 6 は入らない。"""
        net = self.compile(None)
        self.assertEqual(net.body_ids.tolist(), [1, 2, 4, 5, 7, 8, 9])
        self.assertEqual(net.cell_type.tolist(), ["Mi1", "Mi4", "R1-R6", "Tm20", "PAM11", "R8p", "aMe12"])
        self.assertEqual(net.side.tolist(), ["R", "R", "L", "R", "L", "R", "R"])

    def test_signs_and_edges(self):
        """GABA とヒスタミンは負、表に無い 5 は興奮。R8 → aMe12 だけ正にする。網の外への結合は落とす。"""
        net = self.compile(None)
        self.assertEqual(
            self.edges(net),
            {(1, 2, 6), (2, 1, -4), (4, 1, -2), (5, 1, 3), (1, 5, 1), (7, 1, 5), (8, 9, 4), (8, 1, -2)},
        )
        self.assertEqual(net.meta["n_synapses"], 27)
        self.assertEqual(net.meta["n_inhibitory_neurons"], 3)
        self.assertEqual(net.meta["n_overridden_edges"], 1)

    def test_modulatory_cells_are_marked_and_keep_their_edges(self):
        net = self.compile(None)
        self.assertEqual(net.modulatory.tolist(), [False, False, False, False, True, False, False])
        self.assertIn((7, 1, 5), self.edges(net))
        self.assertEqual(net.meta["n_modulatory_neurons"], 1)

    def test_threshold_drops_small_edges(self):
        net = self.compile(3)
        self.assertEqual(self.edges(net), {(1, 2, 6), (2, 1, -4), (5, 1, 3), (7, 1, 5), (8, 9, 4)})

    def test_override_off_keeps_histamine_sign(self):
        sim = dataclasses.replace(self.cfg.simulation, r8_to_ame12_excitatory=False)
        cfg = dataclasses.replace(self.cfg, simulation=sim)
        net = connectome.compile_network(cfg, connectome.spec_from_config(cfg, None))
        self.assertIn((8, 9, -4), self.edges(net))

    def test_mismatched_input_is_refused(self):
        bad = dict(self.cfg.connectome_files)
        bad["weights"] = dataclasses.replace(bad["weights"], sha256="0" * 64)
        cfg = dataclasses.replace(self.cfg, connectome_files=bad)
        with self.assertRaises(connectome.ConnectomeError):
            connectome.compile_network(cfg, connectome.spec_from_config(cfg, None))

    def test_load_or_compile_uses_cache(self):
        spec = connectome.spec_from_config(self.cfg, None)
        first, built = connectome.load_or_compile(self.cfg, spec)
        second, built_again = connectome.load_or_compile(self.cfg, spec)
        self.assertTrue(built)
        self.assertFalse(built_again)
        self.assertEqual(first.post.tolist(), second.post.tolist())


if __name__ == "__main__":
    unittest.main()
