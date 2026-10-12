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



class GuideLineTest(unittest.TestCase):
    """気配の線。重なっても青が消えないこと（Phase 2 後半の決定）。"""

    def setUp(self):
        self.cfg = helpers.load_config()
        self.bid = self.cfg.vision_palette["bid"]
        self.ask = self.cfg.vision_palette["ask"]

    def colors(self, image: vision.Image) -> set[tuple[int, int, int]]:
        return {
            tuple(image.pixels[i : i + 3])
            for i in range(0, len(image.pixels), 3)
        }

    def test_separate_lines_keep_both_colors(self):
        image = vision.render(self.cfg, candles(), Decimal(14600000), Decimal(14800000), "btc_jpy")
        found = self.colors(image)
        self.assertIn(self.bid, found)
        self.assertIn(self.ask, found)

    def test_overlapping_lines_are_mixed(self):
        """スプレッド 1 円。同じ行に落ちても、青の寄与が残ること。"""
        image = vision.render(self.cfg, candles(), Decimal(14699999), Decimal(14700000), "btc_jpy")
        found = self.colors(image)
        mixed = vision._mix(self.bid, self.ask)
        self.assertIn(mixed, found)
        # 片方だけが生き残ってはいない。
        self.assertNotIn(self.bid, found)
        self.assertNotIn(self.ask, found)
        # 混色は両方の青を引き継いでいる。R8 が受け取る青が消えていない。
        self.assertGreater(mixed[2], min(self.bid[2], self.ask[2]))

    def test_mix_is_order_independent(self):
        self.assertEqual(vision._mix(self.bid, self.ask), vision._mix(self.ask, self.bid))

    def test_no_new_color_when_lines_are_apart(self):
        image = vision.render(self.cfg, candles(), Decimal(14600000), Decimal(14800000), "btc_jpy")
        self.assertNotIn(vision._mix(self.bid, self.ask), self.colors(image))


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


class MotionTest(unittest.TestCase):
    """1 観測のあいだに見せる動き（fly.vision.motion）。"""

    def setUp(self):
        self.cfg = helpers.load_config()
        self.grow = dataclasses.replace(self.cfg, vision_motion="grow_last", vision_motion_frames=5)
        self.candles = candles()
        self.bid, self.ask = Decimal(14_699_000), Decimal(14_701_000)

    def test_none_is_the_still_image(self):
        frames = vision.render_frames(self.cfg, self.candles, self.bid, self.ask, "btc_jpy")
        still = vision.render(self.cfg, self.candles, self.bid, self.ask, "btc_jpy")
        self.assertEqual([f.sha256 for f in frames], [still.sha256])

    def test_grow_last_ends_at_the_still_image(self):
        last = self.candles[-1]
        big = (*self.candles[:-1], dataclasses.replace(last, close=last.open + 200_000, high=last.open + 210_000))
        frames = vision.render_frames(self.grow, big, self.bid, self.ask, "btc_jpy")
        still = vision.render(self.cfg, big, self.bid, self.ask, "btc_jpy")
        self.assertEqual(len(frames), 5)
        self.assertEqual(frames[-1].sha256, still.sha256)
        # 実体が大きければ、枚ごとに画像が変わる（小さければ画素の丸めで同じ枚が続きうる）
        self.assertEqual(len({f.sha256 for f in frames}), 5)

    def test_grow_last_moves_only_the_last_candle(self):
        """値幅（縦の目盛り）と文字は動かない。変わる画素は最後の足の列だけ。"""
        frames = vision.render_frames(self.grow, self.candles, self.bid, self.ask, "btc_jpy")
        w = self.cfg.vision_width
        slot = (w - 2 * self.cfg.vision_margin_px) / len(self.candles)
        last_left = int(self.cfg.vision_margin_px + slot * (len(self.candles) - 1))
        for a, b in zip(frames, frames[1:]):
            for i in range(0, len(a.pixels), 3):
                if a.pixels[i:i + 3] != b.pixels[i:i + 3]:
                    self.assertGreaterEqual((i // 3) % w, last_left)

    def test_flat_last_candle_does_not_move(self):
        flat = (*self.candles[:-1], dataclasses.replace(self.candles[-1], close=self.candles[-1].open))
        frames = vision.render_frames(self.grow, flat, self.bid, self.ask, "btc_jpy")
        self.assertEqual(len({f.sha256 for f in frames}), 1)

    def test_growing_candle_keeps_its_color(self):
        """伸びている途中の実体も、最後の足と同じ色（上昇か下落か）で描く。"""
        last = self.candles[-1]
        down = dataclasses.replace(last, close=last.open - 30_000, low=min(last.low, last.open - 40_000))
        frames = vision.render_frames(self.grow, (*self.candles[:-1], down), self.bid, self.ask, "btc_jpy")
        up_color = bytes(self.cfg.vision_palette["up"])
        w = self.cfg.vision_width
        slot = (w - 2 * self.cfg.vision_margin_px) / len(self.candles)
        last_left = int(self.cfg.vision_margin_px + slot * (len(self.candles) - 1))
        for f in frames:
            for i in range(0, len(f.pixels), 3):
                if (i // 3) % w >= last_left:
                    self.assertNotEqual(f.pixels[i:i + 3], up_color)
