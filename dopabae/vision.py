"""ハエに見せる画像。15分足から 320×180 の RGB を決定的に描く。

同じ足・同じ気配・同じ設定なら同じバイト列（同じハッシュ）になる。
見せるのはチャート・ペア名・買い気配・売り気配だけで、残高・損益・建玉は
描かない（CLAUDE.md「ハエに自分の懐を見せない」）。

依存を増やさない。描画は自前のバッファ、PNG の書き出しは zlib だけで行う。
光受容細胞へのマッピング（Phase 2 後半）は配線図の選定後に別モジュールで足す。
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from .config import Config
from .orders import to_decimal

RGB = tuple[int, int, int]

# 5×7 のビットマップ書体。ペア名と気配を書くのに要る文字だけ持つ。
# 1 が点灯。行は上から。
_FONT: dict[str, tuple[str, ...]] = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11100", "10010", "10001", "10001", "10001", "10010", "11100"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "10001", "11001", "10101", "10011", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "10001", "01010", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
}
GLYPH_W, GLYPH_H, GLYPH_GAP = 5, 7, 1


class VisionError(Exception):
    """画像を描けない（足が足りない、設定が不正）。"""


@dataclass(frozen=True)
class Candle:
    timestamp_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal


@dataclass(frozen=True)
class Image:
    """描いた画像。`pixels` は行優先の RGB バイト列（width × height × 3）。"""

    width: int
    height: int
    pixels: bytes
    sha256: str
    candle_type: str
    candle_count: int
    candles_from_ms: int | None
    candles_to_ms: int | None
    # 何を描いたか。残高・損益・建玉が混ざっていないことを記録で確かめられるように残す。
    shown: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "candle_type": self.candle_type,
            "candle_count": self.candle_count,
            "candles_from_ms": self.candles_from_ms,
            "candles_to_ms": self.candles_to_ms,
            "shown": list(self.shown),
        }


class Canvas:
    def __init__(self, width: int, height: int, background: RGB) -> None:
        if width <= 0 or height <= 0:
            raise VisionError(f"画像の大きさが不正です: {width}×{height}")
        self.width = width
        self.height = height
        self.buf = bytearray(bytes(background) * (width * height))

    def put(self, x: int, y: int, color: RGB) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            i = (y * self.width + x) * 3
            self.buf[i : i + 3] = bytes(color)

    def rect(self, x0: int, y0: int, x1: int, y1: int, color: RGB) -> None:
        """両端を含む矩形。座標は自動で正順にする。"""
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        for y in range(max(0, y0), min(self.height - 1, y1) + 1):
            for x in range(max(0, x0), min(self.width - 1, x1) + 1):
                self.put(x, y, color)

    def hline(self, x0: int, x1: int, y: int, color: RGB, dotted: int = 0) -> None:
        if x0 > x1:
            x0, x1 = x1, x0
        for x in range(x0, x1 + 1):
            if dotted and (x - x0) % dotted >= dotted // 2:
                continue
            self.put(x, y, color)

    def text(self, x: int, y: int, s: str, color: RGB) -> int:
        """5×7 の書体で描く。持たない文字は空白にする。右端の x を返す。"""
        cx = x
        for ch in s.upper():
            glyph = _FONT.get(ch, _FONT[" "])
            for row, bits in enumerate(glyph):
                for col, bit in enumerate(bits):
                    if bit == "1":
                        self.put(cx + col, y + row, color)
            cx += GLYPH_W + GLYPH_GAP
        return cx


def parse_candles(rows: Sequence[dict]) -> tuple[Candle, ...]:
    candles = []
    for row in rows:
        try:
            candles.append(
                Candle(
                    timestamp_ms=int(row["timestamp"]),
                    open=to_decimal(row["open"]),
                    high=to_decimal(row["high"]),
                    low=to_decimal(row["low"]),
                    close=to_decimal(row["close"]),
                )
            )
        except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
            raise VisionError(f"足を解釈できません: {exc}") from exc
    return tuple(sorted(candles, key=lambda c: c.timestamp_ms))


def select_recent(candles: Sequence[Candle], count: int, now_ms: int) -> tuple[Candle, ...]:
    """現在時刻までの足から直近 count 本。足りなければ描かない。"""
    if count <= 0:
        raise VisionError("lookback_candles は 1 以上です")
    recent = [c for c in candles if c.timestamp_ms <= now_ms][-count:]
    if len(recent) < count:
        raise VisionError(f"足が {len(recent)} 本しかありません（{count} 本必要）")
    return tuple(recent)


def render(config: Config, candles: Sequence[Candle], bid: Decimal, ask: Decimal, pair: str) -> Image:
    """チャート・ペア名・気配だけを描く。残高・損益・建玉は引数にも取らない。"""
    if not candles:
        raise VisionError("描く足がありません")
    palette = config.vision_palette
    w, h = config.vision_width, config.vision_height
    margin = config.vision_margin_px
    text_rows = config.vision_text_rows_px
    canvas = Canvas(w, h, palette["background"])
    shown: list[str] = []

    if "pair" in config.vision_show:
        canvas.text(margin, margin, pair, palette["text"])
        shown.append("pair")
    if "bid" in config.vision_show or "ask" in config.vision_show:
        label = []
        if "bid" in config.vision_show:
            label.append(f"BID {bid}")
            shown.append("bid")
        if "ask" in config.vision_show:
            label.append(f"ASK {ask}")
            shown.append("ask")
        s = "  ".join(label)
        width_px = len(s) * (GLYPH_W + GLYPH_GAP)
        canvas.text(max(margin, w - margin - width_px), margin, s, palette["text"])

    # 値幅は足と気配の両方を含める。気配の線が枠の外に出ないようにするため。
    top = margin + text_rows + 2
    bottom = h - margin - 1
    left = margin
    right = w - margin - 1
    lows = [c.low for c in candles]
    highs = [c.high for c in candles]
    lo = min(*lows, bid, ask)
    hi = max(*highs, bid, ask)
    if hi <= lo:
        hi = lo + Decimal(1)
    span = hi - lo

    def y_of(price: Decimal) -> int:
        ratio = (price - lo) / span
        return int(bottom - ratio * (bottom - top))

    if "candles" in config.vision_show:
        n = len(candles)
        slot = (right - left + 1) / n
        body_w = max(1, int(slot * 0.6))
        for index, c in enumerate(candles):
            x_center = int(left + slot * index + slot / 2)
            color = palette["up"] if c.close >= c.open else palette["down"]
            canvas.rect(x_center, y_of(c.high), x_center, y_of(c.low), palette["wick"])
            half = body_w // 2
            y0, y1 = y_of(c.open), y_of(c.close)
            if y0 == y1:
                y1 = y0 + 1 if y0 < bottom else y0 - 1
            canvas.rect(x_center - half, y0, x_center - half + body_w - 1, y1, color)
        shown.append("candles")

    if "bid" in config.vision_show:
        canvas.hline(left, right, y_of(bid), palette["bid"], dotted=4)
    if "ask" in config.vision_show:
        canvas.hline(left, right, y_of(ask), palette["ask"], dotted=4)

    pixels = bytes(canvas.buf)
    return Image(
        width=w,
        height=h,
        pixels=pixels,
        sha256=hashlib.sha256(pixels).hexdigest(),
        candle_type=config.vision_candle_type,
        candle_count=len(candles),
        candles_from_ms=candles[0].timestamp_ms,
        candles_to_ms=candles[-1].timestamp_ms,
        shown=tuple(shown),
    )


def to_png(image: Image) -> bytes:
    """人間が見るための PNG。ハッシュは PNG ではなく生の RGB に対して取る。"""

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    stride = image.width * 3
    raw = b"".join(b"\x00" + image.pixels[y * stride : (y + 1) * stride] for y in range(image.height))
    ihdr = struct.pack(">IIBBBBB", image.width, image.height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
