"""光受容細胞のカラム座標を、標的のカラムから決めること。

確かめること。いちばん多くシナプスが行くカラムを選ぶこと。決められない細胞
（標的なし・同数・左右不明）に値を作らないこと。反対側の標的を数えないこと。
書いた表を `dopabae/retina.py` がそのまま読めること。

pyarrow も配線図データも使わない。格子と結合をテスト側で作る。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dopabae import retina

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import export_column_map as export  # noqa: E402

sys.path.pop(0)


def receptor(body_id: int, cell_type: str = "R1-R6", side: str | None = "R") -> export.Receptor:
    return export.Receptor(body_id=body_id, cell_type=cell_type, side=side)


# 標的。100 番台が右、200 番台が左。座標は左右で同じ値を使う（MaleCNS と同じ）。
HEX = {
    101: export.HexCell("R", 3, 4),
    102: export.HexCell("R", 3, 4),
    103: export.HexCell("R", 5, 1),
    201: export.HexCell("L", 3, 4),
    202: export.HexCell("L", 9, 9),
}


def by_id(assignments: list[export.Assignment]) -> dict[int, export.Assignment]:
    return {a.body_id: a for a in assignments}


class AssignTest(unittest.TestCase):
    def test_most_synapses_win(self):
        """同じカラムの標的は足し合わせる。101 と 102 で 6、103 で 5。"""
        out, _ = export.assign({1: receptor(1)}, HEX, [(1, 101, 4), (1, 102, 2), (1, 103, 5)])
        a = by_id(out)[1]
        self.assertEqual((a.hex1, a.hex2), (3, 4))
        self.assertEqual(a.reason, export.MAPPED)
        self.assertAlmostEqual(a.share, 6 / 11)

    def test_tie_gets_no_value(self):
        """1 位が並べば決めない。並びの順で勝ち負けを決めない。"""
        out, _ = export.assign({1: receptor(1)}, HEX, [(1, 101, 5), (1, 103, 5)])
        a = by_id(out)[1]
        self.assertEqual(a.reason, export.TIE)
        self.assertIsNone(a.hex1)
        self.assertIsNone(a.hex2)

    def test_no_target_gets_no_value(self):
        """座標を持つ標的が無ければ値を作らない。座標の無い標的（999）は数えない。"""
        out, _ = export.assign({1: receptor(1)}, HEX, [(1, 999, 50)])
        a = by_id(out)[1]
        self.assertEqual(a.reason, export.NO_TARGET)
        self.assertIsNone(a.hex1)

    def test_unknown_side_gets_no_value(self):
        out, _ = export.assign({1: receptor(1, side=None)}, HEX, [(1, 101, 5)])
        self.assertEqual(by_id(out)[1].reason, export.NO_SIDE)

    def test_other_side_is_not_counted(self):
        """左右で座標の値が同じなので、反対側を数えると別の眼のカラムに置いてしまう。"""
        out, cross = export.assign({1: receptor(1, side="R")}, HEX, [(1, 201, 50), (1, 103, 2)])
        a = by_id(out)[1]
        self.assertEqual((a.hex1, a.hex2), (5, 1))
        self.assertEqual(cross, 50)

    def test_left_eye_uses_left_targets(self):
        out, _ = export.assign({2: receptor(2, side="L")}, HEX, [(2, 202, 3), (2, 101, 9)])
        a = by_id(out)[2]
        self.assertEqual((a.hex1, a.hex2), (9, 9))

    def test_edges_from_other_cells_are_ignored(self):
        out, cross = export.assign({1: receptor(1)}, HEX, [(7, 101, 99), (1, 103, 1)])
        self.assertEqual((by_id(out)[1].hex1, by_id(out)[1].hex2), (5, 1))
        self.assertEqual(cross, 0)

    def test_output_is_sorted_and_complete(self):
        """決められなかった細胞も行は残す（数として記録に残るように）。"""
        receptors = {5: receptor(5), 1: receptor(1), 3: receptor(3)}
        out, _ = export.assign(receptors, HEX, [(1, 101, 1)])
        self.assertEqual([a.body_id for a in out], [1, 3, 5])

    def test_same_input_same_output(self):
        edges = [(1, 101, 4), (1, 103, 5), (2, 101, 1)]
        receptors = {1: receptor(1), 2: receptor(2)}
        self.assertEqual(export.assign(receptors, HEX, edges), export.assign(receptors, HEX, list(reversed(edges))))


class HelperTest(unittest.TestCase):
    def test_side_of_reads_instance_suffix(self):
        self.assertEqual(export.side_of("R1-R6_L"), "L")
        self.assertEqual(export.side_of("Mi1_R"), "R")
        self.assertIsNone(export.side_of("R7R8_unclear"))
        self.assertIsNone(export.side_of(None))

    def test_to_hex_accepts_whole_numbers(self):
        self.assertEqual(export.to_hex(12.0, 1), 12)

    def test_to_hex_rejects_fractions(self):
        """小数のカラム座標は形が想定と違う。丸めて続けない。"""
        with self.assertRaises(export.ExportError):
            export.to_hex(12.5, 1)
        with self.assertRaises(export.ExportError):
            export.to_hex("12", 1)

    def test_photoreceptor_pattern(self):
        for name in ("R1-R6", "R7p", "R8y", "R8_unclear", "R7R8_unclear"):
            self.assertTrue(export.PHOTORECEPTOR_TYPE.match(name), name)
        for name in ("Mi1", "L1", "ER1_a", "Tm20"):
            self.assertFalse(export.PHOTORECEPTOR_TYPE.match(name), name)


class TsvTest(unittest.TestCase):
    def test_retina_reads_what_export_writes(self):
        out, _ = export.assign(
            {1: receptor(1), 2: receptor(2, "R8p"), 3: receptor(3, side=None)},
            HEX,
            [(1, 101, 3), (2, 103, 2)],
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "column_map.tsv"
            path.write_bytes(export.render_tsv(out))
            column_map = retina.load_column_map(path)
        self.assertEqual(len(column_map.columns), 3)
        self.assertEqual(len(column_map.mapped), 2)
        self.assertEqual(column_map.unmapped_count, 1)
        self.assertEqual(
            [(c.body_id, c.cell_type, c.hex1, c.hex2) for c in column_map.columns],
            [(1, "R1-R6", 3, 4), (2, "R8p", 5, 1), (3, "R1-R6", None, None)],
        )

    def test_header_matches_retina(self):
        blob = export.render_tsv([])
        self.assertEqual(blob.decode("utf-8").splitlines()[0].split("\t"), list(retina.COLUMNS))


class VerifyTest(unittest.TestCase):
    def test_missing_input_is_an_error(self):
        """取得していなければ読まない。取得の手順を案内する。"""
        from dopabae.config import load

        config = load()
        with TemporaryDirectory() as tmp:
            import dataclasses

            cfg = dataclasses.replace(config, connectome_local_dir=str(Path(tmp) / "nothing"))
            with self.assertRaises(export.ExportError):
                export.verified(cfg, "annotations")


if __name__ == "__main__":
    unittest.main()
