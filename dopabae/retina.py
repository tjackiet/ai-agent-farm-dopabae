"""画像から光受容細胞への入力（Phase 2 後半）。

`vision.py` が描いた 320×180 の RGB を、光受容細胞ごとの入力値に落とす。
R1–R6 は輝度を、R8 は青／緑の代理値を受け取る。

**ここは売買を知らない。** 受け取るのは画像と配線図の座標だけで、残高・損益・
建玉は引数にも取らない。

**読み替えの責任はすべて設計者にある。** 実際の複眼は、レンズの光学・神経重畳・
眼の曲率を通して世界を見ている。ここで行うのは「平面の画像を六角格子へ配る」
という設計者が決めた対応づけであり、生体の視野との一致は検証されていない
（`docs/CONNECTOME_SURVEY.md` 5.1）。「ハエがチャートを見ている」とは書かない。

配線図データはリポジトリに含めない（同 3.6）。座標表は人間が MaleCNS v1.0 から
書き出し、`fly.retina.column_map_path` に置く。表が無ければ入力を作らず、
呼び出し側が HOLD できるように例外を送る。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .config import R8_SUBTYPE_CHANNELS, Config
from .vision import Image

# 座標表の列。この順で並んでいることを読み込み時に確かめる。
# side は眼の左右（"L" / "R"）。split_eyes で眼ごとに画像の別の場所へ置くのに使う。
COLUMNS = ("body_id", "cell_type", "hex1", "hex2", "side")
SIDES = ("L", "R")

# 線形 sRGB から輝度を出す係数（Rec. 709）。
LUMA = (0.2126, 0.7152, 0.0722)


class RetinaError(Exception):
    """入力を作れない（座標表が無い・壊れている・画像と噛み合わない）。"""


@dataclass(frozen=True)
class Column:
    """光受容細胞 1 個。`hex1` / `hex2` は配線図のカラム座標。"""

    body_id: int
    cell_type: str
    hex1: int | None
    hex2: int | None
    side: str | None = None

    @property
    def is_mapped(self) -> bool:
        return self.hex1 is not None and self.hex2 is not None


@dataclass(frozen=True)
class ColumnMap:
    """座標表。中身と出どころを指紋ごと持つ。"""

    columns: tuple[Column, ...]
    sha256: str
    source: str

    @property
    def mapped(self) -> tuple[Column, ...]:
        return tuple(c for c in self.columns if c.is_mapped)

    @property
    def unmapped_count(self) -> int:
        return len(self.columns) - len(self.mapped)


@dataclass(frozen=True)
class Field:
    """1 個の受容細胞が見る画素の範囲。両端を含む。"""

    body_id: int
    cell_type: str
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class Retina:
    """画像から作った入力。`currents` は body_id → 0〜1 の値。"""

    currents: dict[int, float]
    # 記録に残すための観測値。解釈は入れない。
    column_map_sha256: str
    image_sha256: str
    field_w: int
    field_h: int
    luminance_count: int
    blue_green_count: int
    unmapped_count: int
    layout: str = "shared"
    sampling: str = "area"

    def as_dict(self) -> dict:
        return {
            "column_map_sha256": self.column_map_sha256,
            "image_sha256": self.image_sha256,
            "layout": self.layout,
            "sampling": self.sampling,
            "field_px": [self.field_w, self.field_h],
            "luminance_count": self.luminance_count,
            "blue_green_count": self.blue_green_count,
            "unmapped_count": self.unmapped_count,
        }


def _cell(value: str, line_no: int, name: str) -> int | None:
    """空欄と `null` は None。数値でなければ拒否する。"""
    text = value.strip()
    if text == "" or text.lower() in ("null", "none", "na", "nan"):
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise RetinaError(f"座標表の {line_no} 行目、{name} が整数ではありません: {text!r}") from exc


def load_column_map(path: Path | str) -> ColumnMap:
    """TSV の座標表を読む。1 行目は見出しで、`COLUMNS` の順に並ぶ。

    観測できなかった座標は `null` のままにする。埋めない（`CLAUDE.md`）。
    """
    target = Path(path)
    try:
        blob = target.read_bytes()
    except OSError as exc:
        raise RetinaError(f"座標表が読めません: {target}") from exc

    digest = hashlib.sha256(blob).hexdigest()
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RetinaError(f"座標表が UTF-8 ではありません: {target}") from exc

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise RetinaError(f"座標表が空です: {target}")
    header = tuple(h.strip() for h in lines[0].split("\t"))
    if header != COLUMNS:
        raise RetinaError(
            f"座標表の見出しが違います。{chr(9).join(COLUMNS)} の順に並べる: {header}。"
            "古い形（side の列が無い）なら scripts/export_column_map.py で書き出し直す"
        )

    seen: set[int] = set()
    columns: list[Column] = []
    for offset, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != len(COLUMNS):
            raise RetinaError(f"座標表の {offset} 行目の列数が {len(parts)} です（{len(COLUMNS)} 列）")
        body_id = _cell(parts[0], offset, "body_id")
        if body_id is None:
            raise RetinaError(f"座標表の {offset} 行目に body_id がありません")
        if body_id in seen:
            raise RetinaError(f"座標表の body_id が重複しています: {body_id}")
        seen.add(body_id)
        cell_type = parts[1].strip()
        if not cell_type:
            raise RetinaError(f"座標表の {offset} 行目に cell_type がありません")
        side_text = parts[4].strip()
        if side_text.lower() in ("", "null", "none", "na", "nan"):
            side = None
        elif side_text in SIDES:
            side = side_text
        else:
            raise RetinaError(f"座標表の {offset} 行目、side は L / R / null です: {side_text!r}")
        columns.append(
            Column(
                body_id=body_id,
                cell_type=cell_type,
                hex1=_cell(parts[2], offset, "hex1"),
                hex2=_cell(parts[3], offset, "hex2"),
                side=side,
            )
        )
    return ColumnMap(columns=tuple(columns), sha256=digest, source=target.name)


def _axial_to_plane(hex1: int, hex2: int) -> tuple[float, float]:
    """六角格子の軸座標を平面へ置く。

    `hex2` の行は半個ぶんずれ、行間は √3/2 になる。格子の形をそのまま平面へ
    移しただけで、視野との対応づけではない。
    """
    return hex1 + hex2 / 2.0, hex2 * math.sqrt(3.0) / 2.0


def _field_size(
    config: Config, centers: Sequence[tuple[int, int]], image: Image
) -> tuple[int, int]:
    """1 個の受容細胞が見る範囲の大きさ。**画像の縁で切り取られる前の値である。**

    記録にはこちらを残す。切り取られた個体の大きさを残すと、格子の目の粗さを
    読み違える。
    """
    if config.retina_sampling == "point":
        return 1, 1
    if config.retina_field_px is not None:
        return config.retina_field_px, config.retina_field_px
    # 格子の目の粗さから決める。重なりも隙間も作らない大きさになる。
    nx = len({x for x, _ in centers})
    ny = len({y for _, y in centers})
    return max(1, round(image.width / nx)), max(1, round(image.height / ny))


def fields(config: Config, column_map: ColumnMap, image: Image) -> tuple[Field, ...]:
    """各受容細胞が見る画素の範囲を決める。

    置きかたは `fly.retina.layout`。split_eyes は両眼を画像の左右に置き（Stonkfly）、
    shared は両眼とも格子の広がりを画像いっぱいに引き伸ばす（Phase 2）。
    **どちらも設計者が決めた対応づけであり、複眼の視野角とは無関係である。**

    拾いかたは `fly.retina.sampling`。point は 1 画素（Stonkfly）、area は範囲の平均
    （Phase 2）。気配の線は 1 画素の破線なので、point では当たるかどうかが位置で決まり、
    area なら線の有無が濃さの差として残る。
    """
    return _build(config, column_map, image)[0]


def _build(
    config: Config, column_map: ColumnMap, image: Image
) -> tuple[tuple[Field, ...], int, int]:
    """範囲と、切り取られる前の範囲の大きさ。中心の計算を 1 度で済ませる。"""
    placed = _placed(config, column_map)
    centers = _centers(config, placed, image)
    field_w, field_h = _field_size(config, centers, image)
    half_w, half_h = field_w // 2, field_h // 2
    built = tuple(
        Field(
            body_id=column.body_id,
            cell_type=column.cell_type,
            x0=max(0, x - half_w),
            y0=max(0, y - half_h),
            x1=min(image.width - 1, x - half_w + field_w - 1),
            y1=min(image.height - 1, y - half_h + field_h - 1),
        )
        for column, (x, y) in zip(placed, centers)
    )
    return built, field_w, field_h


def _placed(config: Config, column_map: ColumnMap) -> tuple[Column, ...]:
    """画像に置ける受容細胞。split_eyes では左右の分からない細胞を置かない（値を作らない）。"""
    mapped = column_map.mapped
    if config.retina_layout == "split_eyes":
        mapped = tuple(c for c in mapped if c.side in SIDES)
    if not mapped:
        raise RetinaError("座標表に、画像へ置ける受容細胞が 1 個もありません")
    return mapped


def _stonkfly_plane(hex1: int, hex2: int) -> tuple[float, float]:
    """Stonkfly の置きかた（`hex1 − hex2/2`、行間 √3/2）。"""
    return hex1 - hex2 / 2.0, hex2 * math.sqrt(3.0) / 2.0


def _split_eye_uv(config: Config, placed: Sequence[Column]) -> list[tuple[float, float]]:
    """両眼を画像の左右に置く（Stonkfly と同じ。docs/CONNECTOME_SURVEY.md 1.1）。

    眼ごとに、輝度を受け取る型（R1-R6）の広がりを 0〜1 に引き伸ばし、左眼は画像の左
    `eye_width`、右眼は右 `eye_width` に置く。右眼は左右を反転する。hex2 が増える向きが上。
    """
    plane = [_stonkfly_plane(c.hex1, c.hex2) for c in placed]  # type: ignore[arg-type]
    extent: dict[str, tuple[float, float, float, float]] = {}
    for side in SIDES:
        anchors = [
            p for c, p in zip(placed, plane)
            if c.side == side and c.cell_type in config.retina_luminance_types
        ] or [p for c, p in zip(placed, plane) if c.side == side]
        if anchors:
            xs = [p[0] for p in anchors]
            ys = [p[1] for p in anchors]
            extent[side] = (min(xs), max(xs) - min(xs), min(ys), max(ys) - min(ys))

    width = config.retina_eye_width
    out = []
    for column, (px, py) in zip(placed, plane):
        x_lo, x_span, y_lo, y_span = extent[column.side]  # type: ignore[index]
        zx = 0.5 if x_span <= 0 else (px - x_lo) / x_span
        zy = 0.5 if y_span <= 0 else (py - y_lo) / y_span
        u = width * zx if column.side == "L" else (1.0 - width) + width * (1.0 - zx)
        out.append((min(1.0, max(0.0, u)), min(1.0, max(0.0, 1.0 - zy))))
    return out


def _centers(
    config: Config, mapped: Sequence[Column], image: Image
) -> tuple[tuple[int, int], ...]:
    """各受容細胞の中心画素。"""
    if config.retina_layout == "split_eyes":
        # Stonkfly と同じく、切り捨てで画素へ落とす。
        return tuple(
            (min(image.width - 1, int(u * (image.width - 1))), min(image.height - 1, int(v * (image.height - 1))))
            for u, v in _split_eye_uv(config, mapped)
        )

    # shared: 両眼を区別せず、格子の広がりを画像いっぱいに引き伸ばす（Phase 2）。
    plane = [_axial_to_plane(c.hex1, c.hex2) for c in mapped]  # type: ignore[arg-type]
    xs = [p[0] for p in plane]
    ys = [p[1] for p in plane]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    x_span = x_hi - x_lo
    y_span = y_hi - y_lo

    def place(value: float, lo: float, span: float, size: int, flip: bool) -> int:
        # 広がりが 0（1 列しかない）ときは真ん中へ置く。0 除算を避ける。
        ratio = 0.5 if span <= 0 else (value - lo) / span
        if flip:
            ratio = 1.0 - ratio
        return min(size - 1, max(0, int(round(ratio * (size - 1)))))

    return tuple(
        (
            place(px, x_lo, x_span, image.width, config.retina_flip_x),
            place(py, y_lo, y_span, image.height, config.retina_flip_y),
        )
        for px, py in plane
    )


# sRGB の 0〜255 を線形の 0〜1 へ。表にして毎回の計算を省く。
def _linearize(value: int) -> float:
    c = value / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


_LINEAR = tuple(_linearize(v) for v in range(256))


def mean_linear_rgb(image: Image, field: Field) -> tuple[float, float, float]:
    """範囲の平均を線形 sRGB で返す。ガンマを外してから平均する。"""
    total = [0.0, 0.0, 0.0]
    count = 0
    stride = image.width * 3
    for y in range(field.y0, field.y1 + 1):
        base = y * stride
        for x in range(field.x0, field.x1 + 1):
            i = base + x * 3
            total[0] += _LINEAR[image.pixels[i]]
            total[1] += _LINEAR[image.pixels[i + 1]]
            total[2] += _LINEAR[image.pixels[i + 2]]
            count += 1
    if count == 0:  # pragma: no cover - Field は必ず 1 画素以上を含む
        raise RetinaError(f"受容細胞 {field.body_id} の範囲が空です")
    return total[0] / count, total[1] / count, total[2] / count


def luminance(rgb: tuple[float, float, float]) -> float:
    """線形 sRGB の輝度。R1–R6 に渡す値。"""
    return LUMA[0] * rgb[0] + LUMA[1] * rgb[1] + LUMA[2] * rgb[2]


def blue_green(rgb: tuple[float, float, float], channel: str, cell_type: str = "") -> float:
    """R8 に渡す青／緑の代理値。

    実際の R8 は個眼ごとに Rh5（青）と Rh6（緑）へ分かれる。**その区別が
    MaleCNS v1.0 に注釈されているかは未確認**なので、いまは 1 本の代理値にする。
    注釈があると分かれば分けられる。どれを使うかは `agent.yaml` に置く。
    """
    if channel == "by_subtype":
        # R8p は青、R8y は緑（Stonkfly と同じ）。型から決められなければ値を作らない。
        if cell_type not in R8_SUBTYPE_CHANNELS:
            raise RetinaError(f"by_subtype では {cell_type!r} の色を決められません")
        channel = R8_SUBTYPE_CHANNELS[cell_type]
    if channel == "blue_green_mean":
        return (rgb[1] + rgb[2]) / 2.0
    if channel == "green":
        return rgb[1]
    if channel == "blue":
        return rgb[2]
    raise RetinaError(f"扱えない r8_channel です: {channel}")


def _matches(cell_type: str, names: Sequence[str]) -> bool:
    return cell_type in names


def activations(config: Config, image: Image, column_map: ColumnMap) -> Retina:
    """画像から受容細胞ごとの入力を作る。

    **座標の無い受容細胞には値を作らない。** 数だけ記録に残す
    （`CLAUDE.md`「観測していない値を書かない」）。

    返す値は 0〜1 に正規化した明るさである。LIF への電流に直す倍率は
    Phase 3 で実測してから `agent.yaml` に置く。ここでは決めない。
    """
    built, field_w, field_h = _build(config, column_map, image)
    currents: dict[int, float] = {}
    luminance_count = 0
    blue_green_count = 0

    for field in built:
        if _matches(field.cell_type, config.retina_luminance_types):
            currents[field.body_id] = luminance(mean_linear_rgb(image, field))
            luminance_count += 1
        elif _matches(field.cell_type, config.retina_blue_green_types):
            currents[field.body_id] = blue_green(
                mean_linear_rgb(image, field), config.retina_r8_channel, field.cell_type
            )
            blue_green_count += 1
        # どちらでもない型には値を作らない。座標表に混ざっていても無視する。

    # 型名が 1 つも噛み合わなければ、静かに空の入力を返さず落とす。
    # 空の入力は「真っ暗な世界」と区別がつかない。
    if luminance_count == 0 and blue_green_count == 0:
        raise RetinaError(
            "座標表の cell_type が fly.retina の型名と 1 つも噛み合いませんでした。"
            f"表にある型の例: {sorted({c.cell_type for c in column_map.mapped})[:5]}"
        )

    return Retina(
        currents=currents,
        column_map_sha256=column_map.sha256,
        image_sha256=image.sha256,
        field_w=field_w,
        field_h=field_h,
        luminance_count=luminance_count,
        blue_green_count=blue_green_count,
        unmapped_count=column_map.unmapped_count + (len(column_map.mapped) - len(built)),
        layout=config.retina_layout,
        sampling=config.retina_sampling,
    )
