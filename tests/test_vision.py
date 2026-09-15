"""ハエに見せる画像。決定的に描け、懐を見せず、ハッシュと足の範囲が残ること。"""

from __future__ import annotations

import dataclasses
import struct
import unittest
import zlib
from decimal import Decimal

from dopabae import vision
from tests import helpers


def candles(count: int = 8) -> tuple[vision.Candle, ...]:
    return vision.select_recent(vision.parse_candles(helpers.candles(count=count + 4)), count, helpers.NOW_MS)


class VisionTest(unittest.TestCase):
    def setUp(self):
        self.cfg = helpers.load_config()

    def test_size_and_light_background(self):
        image = vision.render(self.cfg, candles(), Decimal(14699000), Decimal(14701000), "btc_jpy")
        self.assertEqual((image.width, image.height), (self.cfg.vision_width, self.cfg.vision_height))
        self.assertEqual(len(image.pixels), image.width * image.height * 3)
        # 背景が画素の過半を占め、明るい。
        bg = bytes(self.cfg.vision_palette["background"])
        count = sum(1 for i in range(0, len(image.pixels), 3) if image.pixels[i : i + 3] == bg)
        self.assertGreater(count, image.width * image.height // 2)
        self.assertGreater(min(bg), 200)

    def test_same_input_same_hash(self):
        a = vision.render(self.cfg, candles(), Decimal(14699000), Decimal(14701000), "btc_jpy")
        b = vision.render(self.cfg, candles(), Decimal(14699000), Decimal(14701000), "btc_jpy")
        self.assertEqual(a.sha256, b.sha256)
        self.assertEqual(a.pixels, b.pixels)

    def test_different_bid_changes_hash(self):
        a = vision.render(self.cfg, candles(), Decimal(14699000), Decimal(14701000), "btc_jpy")
        b = vision.render(self.cfg, candles(), Decimal(14690000), Decimal(14701000), "btc_jpy")
        self.assertNotEqual(a.sha256, b.sha256)

    def test_records_candle_range_and_what_was_shown(self):
        cs = candles()
        image = vision.render(self.cfg, cs, Decimal(14699000), Decimal(14701000), "btc_jpy")
        self.assertEqual(image.candle_count, self.cfg.vision_lookback_candles)
        self.assertEqual((image.candles_from_ms, image.candles_to_ms), (cs[0].timestamp_ms, cs[-1].timestamp_ms))
        self.assertEqual(set(image.shown), {"candles", "pair", "bid", "ask"})
        for hidden in ("balance", "pnl", "position"):
            self.assertNotIn(hidden, image.shown)
        self.assertEqual(set(image.as_dict()), {"sha256", "width", "height", "candle_type", "candle_count", "candles_from_ms", "candles_to_ms", "shown"})

    def test_too_few_candles_is_an_error(self):
        rows = vision.parse_candles(helpers.candles(count=3))
        with self.assertRaises(vision.VisionError):
            vision.select_recent(rows, self.cfg.vision_lookback_candles, helpers.NOW_MS)

    def test_future_candles_are_ignored(self):
        rows = vision.parse_candles(helpers.candles(count=12, end_ms=helpers.NOW_MS + 4 * 900_000))
        recent = vision.select_recent(rows, 8, helpers.NOW_MS)
        self.assertLessEqual(recent[-1].timestamp_ms, helpers.NOW_MS)

    def test_bad_candle_row_is_an_error(self):
        with self.assertRaises(vision.VisionError):
            vision.parse_candles([{"open": "x"}])

    def test_png_is_well_formed(self):
        image = vision.render(self.cfg, candles(), Decimal(14699000), Decimal(14701000), "btc_jpy")
        png = vision.to_png(image)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        width, height = struct.unpack(">II", png[16:24])
        self.assertEqual((width, height), (image.width, image.height))
        # IDAT を復号すると、各行にフィルタバイト 1 つ＋RGB が並ぶ。
        idat_start = png.index(b"IDAT") + 4
        idat_len = struct.unpack(">I", png[idat_start - 8 : idat_start - 4])[0]
        raw = zlib.decompress(png[idat_start : idat_start + idat_len])
        self.assertEqual(len(raw), image.height * (1 + image.width * 3))

    def test_text_uses_only_known_glyphs_without_crashing(self):
        cfg = dataclasses.replace(self.cfg, vision_show=("pair",))
        image = vision.render(cfg, candles(), Decimal(1), Decimal(2), "weird pair ~!@#")
        self.assertEqual(image.shown, ("pair",))

    def test_flat_candles_still_render(self):
        flat = tuple(vision.Candle(i * 900_000, Decimal(100), Decimal(100), Decimal(100), Decimal(100)) for i in range(8))
        image = vision.render(self.cfg, flat, Decimal(100), Decimal(100), "btc_jpy")
        self.assertEqual(image.candle_count, 8)
