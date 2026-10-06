"""画像から光受容細胞への入力。

確かめること。座標の無い細胞に値を作らないこと。範囲の平均で拾うので
1 画素の破線が消えないこと。型名が噛み合わなければ静かに空を返さず落ちること。
"""

from __future__ import annotations

import dataclasses
import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from dopabae import retina, vision
from tests import helpers


def candles(count: int = 8) -> tuple[vision.Candle, ...]:
    return vision.select_recent(
        vision.parse_candles(helpers.candles(count=count + 4)), count, helpers.NOW_MS
    )


def image(cfg, bid=Decimal(14699000), ask=Decimal(14701000)) -> vision.Image:
    return vision.render(cfg, candles(), bid, ask, "btc_jpy")


# MaleCNS v1.0 の光受容細胞の型名（2026-10-05、body-annotations の type 列で確かめた）。
MALECNS_PHOTORECEPTOR_TYPES = (
    "R1-R6",
    "R7R8_unclear",
    "R7_unclear",
    "R7d",
    "R7p",
    "R7y",
    "R8_unclear",
    "R8d",
    "R8p",
    "R8y",
)


def lattice(radius: int = 6, r8_every: int = 7, side: str = "R") -> list[tuple]:
    """六角格子を模した座標表の中身。実データの代わりに使う。

    型名は MaleCNS v1.0 の実データと同じものを使う（`R1-R6` / `R8p` / `R8y`）。
    """
    rows: list[tuple] = []
    body = 1000 if side == "R" else 5000
    for q in range(-radius, radius + 1):
        for p in range(-radius, radius + 1):
            body += 1
            if body % r8_every == 0:
                kind = "R8p" if body % 2 else "R8y"
            else:
                kind = "R1-R6"
            rows.append((body, kind, p, q, side))
    return rows


def write_map(directory: Path, rows, name: str = "column_map.tsv") -> Path:
    """行は (body_id, 型, hex1, hex2) か、末尾に左右を足した 5 つ組。左右の既定は R。"""
    lines = ["\t".join(retina.COLUMNS)]
    for row in rows:
        body_id, cell_type, hex1, hex2 = row[:4]
        side = row[4] if len(row) > 4 else "R"
        lines.append(
            "\t".join(
                [
                    str(body_id),
                    cell_type,
                    "null" if hex1 is None else str(hex1),
                    "null" if hex2 is None else str(hex2),
                    "null" if side is None else side,
                ]
            )
        )
    path = directory / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class LoadTest(unittest.TestCase):
    def setUp(self):
        self.dir = TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.addCleanup(self.dir.cleanup)

    def test_missing_file_raises(self):
        with self.assertRaises(retina.RetinaError):
            retina.load_column_map(self.root / "ない.tsv")

    def test_header_must_match(self):
        path = self.root / "x.tsv"
        path.write_text("id\ttype\ta\tb\n1\tR1-R6\t0\t0\n", encoding="utf-8")
        with self.assertRaises(retina.RetinaError):
            retina.load_column_map(path)

    def test_old_four_column_map_is_refused(self):
        """side の列が無い古い表は読まない（書き出し直す）。"""
        path = self.root / "old.tsv"
        path.write_text("body_id\tcell_type\thex1\thex2\n1\tR1-R6\t0\t0\n", encoding="utf-8")
        with self.assertRaises(retina.RetinaError):
            retina.load_column_map(path)

    def test_side_is_read(self):
        path = write_map(self.root, [(1, "R1-R6", 0, 0, "L"), (2, "R1-R6", 1, 0, None)])
        column_map = retina.load_column_map(path)
        self.assertEqual([c.side for c in column_map.columns], ["L", None])

    def test_bad_side_is_refused(self):
        path = write_map(self.root, [(1, "R1-R6", 0, 0, "X")])
        with self.assertRaises(retina.RetinaError):
            retina.load_column_map(path)

    def test_duplicate_body_id_raises(self):
        path = write_map(self.root, [(1, "R1-R6", 0, 0), (1, "R8p", 1, 0)])
        with self.assertRaises(retina.RetinaError):
            retina.load_column_map(path)

    def test_non_integer_hex_raises(self):
        path = write_map(self.root, [(1, "R1-R6", 0, 0)])
        path.write_text(path.read_text(encoding="utf-8").replace("R1-R6\t0\t0\t", "R1-R6\tx\t0\t"), encoding="utf-8")
        with self.assertRaises(retina.RetinaError):
            retina.load_column_map(path)

    def test_null_hex_stays_unmapped(self):
        """観測できなかった座標は埋めない。数だけ残す。"""
        path = write_map(self.root, [(1, "R1-R6", 0, 0), (2, "R1-R6", None, None), (3, "R1-R6", None, 2)])
        column_map = retina.load_column_map(path)
        self.assertEqual(len(column_map.columns), 3)
        self.assertEqual(len(column_map.mapped), 1)
        self.assertEqual(column_map.unmapped_count, 2)

    def test_sha256_is_of_the_file(self):
        import hashlib

        path = write_map(self.root, lattice(2))
        column_map = retina.load_column_map(path)
        self.assertEqual(column_map.sha256, hashlib.sha256(path.read_bytes()).hexdigest())


class ActivationTest(unittest.TestCase):
    def setUp(self):
        self.cfg = helpers.load_config()
        self.dir = TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.addCleanup(self.dir.cleanup)
        self.map = retina.load_column_map(write_map(self.root, lattice()))

    def test_unmapped_cells_get_no_value(self):
        rows = lattice(2) + [(999001, "R1-R6", None, None), (999002, "R8p", None, None)]
        column_map = retina.load_column_map(write_map(self.root, rows, "b.tsv"))
        out = retina.activations(self.cfg, image(self.cfg), column_map)
        self.assertEqual(out.unmapped_count, 2)
        self.assertNotIn(999001, out.currents)
        self.assertNotIn(999002, out.currents)

    def test_every_mapped_receptor_gets_a_value(self):
        out = retina.activations(self.cfg, image(self.cfg), self.map)
        self.assertEqual(len(out.currents), out.luminance_count + out.blue_green_count)
        self.assertEqual(len(out.currents), len(self.map.mapped))
        self.assertGreater(out.luminance_count, 0)
        self.assertGreater(out.blue_green_count, 0)

    def test_values_are_between_zero_and_one(self):
        out = retina.activations(self.cfg, image(self.cfg), self.map)
        for value in out.currents.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_same_image_same_values(self):
        a = retina.activations(self.cfg, image(self.cfg), self.map)
        b = retina.activations(self.cfg, image(self.cfg), self.map)
        self.assertEqual(a.currents, b.currents)
        self.assertEqual(a.image_sha256, b.image_sha256)

    def test_different_image_changes_values(self):
        a = retina.activations(self.cfg, image(self.cfg), self.map)
        b = retina.activations(self.cfg, image(self.cfg, Decimal(14600000), Decimal(14601000)), self.map)
        self.assertNotEqual(a.currents, b.currents)

    def test_unknown_cell_type_is_ignored(self):
        rows = lattice(2) + [(999003, "Mi1", 0, 0)]
        column_map = retina.load_column_map(write_map(self.root, rows, "c.tsv"))
        out = retina.activations(self.cfg, image(self.cfg), column_map)
        self.assertNotIn(999003, out.currents)

    def test_unused_photoreceptor_types_get_no_value(self):
        """R7 / R8d / R8_unclear は表に載っていても値を作らない（agent.yaml の型名で選ぶ）。"""
        rows = lattice(2) + [(999004, "R7p", 0, 0), (999005, "R8d", 1, 0), (999006, "R8_unclear", 0, 1)]
        column_map = retina.load_column_map(write_map(self.root, rows, "g.tsv"))
        out = retina.activations(self.cfg, image(self.cfg), column_map)
        for body_id in (999004, 999005, 999006):
            self.assertNotIn(body_id, out.currents)

    def test_repo_config_uses_malecns_type_names(self):
        """agent.yaml の型名が MaleCNS v1.0 に実在すること。噛み合わなければ入力が作れない。"""
        for name in self.cfg.retina_luminance_types + self.cfg.retina_blue_green_types:
            self.assertIn(name, MALECNS_PHOTORECEPTOR_TYPES)

    def test_no_matching_type_raises(self):
        """空の入力は「真っ暗な世界」と区別がつかない。静かに返さない。"""
        rows = [(1, "Mi1", 0, 0), (2, "Tm3", 1, 0)]
        column_map = retina.load_column_map(write_map(self.root, rows, "d.tsv"))
        with self.assertRaises(retina.RetinaError):
            retina.activations(self.cfg, image(self.cfg), column_map)

    def test_no_mapped_cell_raises(self):
        column_map = retina.load_column_map(write_map(self.root, [(1, "R1-R6", None, None)], "e.tsv"))
        with self.assertRaises(retina.RetinaError):
            retina.activations(self.cfg, image(self.cfg), column_map)

    def test_record_holds_only_observed_values(self):
        out = retina.activations(self.cfg, image(self.cfg), self.map)
        record = out.as_dict()
        self.assertEqual(record["column_map_sha256"], self.map.sha256)
        self.assertEqual(record["image_sha256"], image(self.cfg).sha256)
        self.assertEqual(record["field_px"], [out.field_w, out.field_h])
        # 発火数・電流・方向のような、まだ観測していないものを入れない。
        self.assertEqual(
            set(record),
            {
                "column_map_sha256",
                "image_sha256",
                "layout",
                "sampling",
                "field_px",
                "luminance_count",
                "blue_green_count",
                "unmapped_count",
            },
        )


class FieldTest(unittest.TestCase):
    """Phase 2 の置きかた（shared）と拾いかた（area）。"""

    def setUp(self):
        self.cfg = dataclasses.replace(helpers.load_config(), retina_layout="shared", retina_sampling="area")
        self.dir = TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.addCleanup(self.dir.cleanup)
        self.map = retina.load_column_map(write_map(self.root, lattice()))

    def test_fields_stay_inside_the_image(self):
        img = image(self.cfg)
        for field in retina.fields(self.cfg, self.map, img):
            self.assertGreaterEqual(field.x0, 0)
            self.assertGreaterEqual(field.y0, 0)
            self.assertLess(field.x1, img.width)
            self.assertLess(field.y1, img.height)
            self.assertLessEqual(field.x0, field.x1)
            self.assertLessEqual(field.y0, field.y1)

    def test_area_field_is_wider_than_one_pixel(self):
        """気配の線は 1 画素の破線。点で拾うと当たるかどうかが運で決まる。"""
        out = retina.activations(self.cfg, image(self.cfg), self.map)
        self.assertGreater(out.field_w * out.field_h, 1)

    def test_fixed_field_px_is_used(self):
        cfg = dataclasses.replace(self.cfg, retina_field_px=9)
        out = retina.activations(cfg, image(cfg), self.map)
        self.assertEqual((out.field_w, out.field_h), (9, 9))

    def test_flip_y_moves_the_lattice(self):
        img = image(self.cfg)
        up = {f.body_id: f.y0 for f in retina.fields(dataclasses.replace(self.cfg, retina_flip_y=False), self.map, img)}
        down = {f.body_id: f.y0 for f in retina.fields(dataclasses.replace(self.cfg, retina_flip_y=True), self.map, img)}
        self.assertNotEqual(up, down)

    def test_single_column_lands_in_the_middle(self):
        """広がりが 0 でも 0 除算しない。"""
        column_map = retina.load_column_map(write_map(self.root, [(1, "R1-R6", 3, 3)], "f.tsv"))
        img = image(self.cfg)
        field = retina.fields(self.cfg, column_map, img)[0]
        self.assertLessEqual(abs((field.x0 + field.x1) // 2 - img.width // 2), img.width // 4)


class StonkflyLayoutTest(unittest.TestCase):
    """Stonkfly の置きかた（split_eyes）と拾いかた（point）。agent.yaml の既定。"""

    def setUp(self):
        self.cfg = helpers.load_config()
        self.dir = TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.addCleanup(self.dir.cleanup)
        rows = lattice(4, side="L") + lattice(4, side="R")
        self.map = retina.load_column_map(write_map(self.root, rows))

    def test_defaults_follow_stonkfly(self):
        self.assertEqual(self.cfg.retina_layout, "split_eyes")
        self.assertEqual(self.cfg.retina_sampling, "point")
        self.assertEqual(self.cfg.retina_r8_channel, "by_subtype")

    def test_point_sampling_is_one_pixel(self):
        for field in retina.fields(self.cfg, self.map, image(self.cfg)):
            self.assertEqual((field.x0, field.y0), (field.x1, field.y1))

    def test_eyes_take_left_and_right_parts(self):
        """左眼は画像の左 60%、右眼は右 60% に置く。中央 20% が重なる。"""
        img = image(self.cfg)
        sides = {c.body_id: c.side for c in self.map.columns}
        xs = {"L": [], "R": []}
        for field in retina.fields(self.cfg, self.map, img):
            xs[sides[field.body_id]].append(field.x0)
        limit = self.cfg.retina_eye_width * (img.width - 1)
        self.assertLessEqual(max(xs["L"]), limit)
        self.assertGreaterEqual(min(xs["R"]), (1 - self.cfg.retina_eye_width) * (img.width - 1) - 1)
        self.assertEqual(min(xs["L"]), 0)
        self.assertEqual(max(xs["R"]), img.width - 1)

    def test_right_eye_is_mirrored(self):
        """右眼は左右を反転して置く。hex1 が大きいほど左へ寄る。"""
        img = image(self.cfg)
        by_id = {f.body_id: f.x0 for f in retina.fields(self.cfg, self.map, img)}
        right = [c for c in self.map.columns if c.side == "R" and c.hex2 == 0]
        low = min(right, key=lambda c: c.hex1)
        high = max(right, key=lambda c: c.hex1)
        self.assertGreater(by_id[low.body_id], by_id[high.body_id])

    def test_cells_without_side_get_no_value(self):
        rows = lattice(2, side="R") + [(999010, "R1-R6", 0, 0, None)]
        column_map = retina.load_column_map(write_map(self.root, rows, "s.tsv"))
        out = retina.activations(self.cfg, image(self.cfg), column_map)
        self.assertNotIn(999010, out.currents)
        self.assertEqual(out.unmapped_count, 1)
        self.assertEqual(out.as_dict()["layout"], "split_eyes")

    def test_r8_by_subtype(self):
        """R8p は青、R8y は緑（Stonkfly と同じ）。"""
        rgb = (0.1, 0.2, 0.3)
        self.assertEqual(retina.blue_green(rgb, "by_subtype", "R8p"), 0.3)
        self.assertEqual(retina.blue_green(rgb, "by_subtype", "R8y"), 0.2)
        with self.assertRaises(retina.RetinaError):
            retina.blue_green(rgb, "by_subtype", "R8_unclear")


class ChannelTest(unittest.TestCase):
    """輝度と青／緑が、値として別ものであること。"""

    def setUp(self):
        self.cfg = helpers.load_config()

    def test_luminance_uses_rec709(self):
        self.assertAlmostEqual(retina.luminance((1.0, 0.0, 0.0)), retina.LUMA[0])
        self.assertAlmostEqual(retina.luminance((0.0, 1.0, 0.0)), retina.LUMA[1])
        self.assertAlmostEqual(retina.luminance((0.0, 0.0, 1.0)), retina.LUMA[2])

    def test_blue_green_ignores_red(self):
        a = retina.blue_green((0.0, 0.4, 0.6), "blue_green_mean")
        b = retina.blue_green((1.0, 0.4, 0.6), "blue_green_mean")
        self.assertEqual(a, b)
        self.assertAlmostEqual(a, 0.5)

    def test_blue_and_green_pick_one_channel(self):
        self.assertEqual(retina.blue_green((0.1, 0.2, 0.3), "blue"), 0.3)
        self.assertEqual(retina.blue_green((0.1, 0.2, 0.3), "green"), 0.2)

    def test_unknown_channel_raises(self):
        with self.assertRaises(retina.RetinaError):
            retina.blue_green((0.1, 0.2, 0.3), "ultraviolet")

    def test_linearization_undoes_gamma(self):
        self.assertAlmostEqual(retina._LINEAR[0], 0.0)
        self.assertAlmostEqual(retina._LINEAR[255], 1.0)
        # 中間の灰色は、線形では半分よりずっと暗い。
        self.assertLess(retina._LINEAR[128], 0.25)


if __name__ == "__main__":
    unittest.main()
