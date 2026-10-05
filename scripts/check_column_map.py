#!/usr/bin/env python3
"""カラム座標表を点検する。

配線図データはリポジトリに含めない（`docs/CONNECTOME_SURVEY.md` 3.6）。
表は人間が MaleCNS v1.0 から書き出し、`fly.retina.column_map_path` へ置く。
このスクリプトは、置いた表が `dopabae/retina.py` の想定と噛み合うかを確かめる。

**表の中身は書き換えない。読んで数えるだけである。**

使いかた:

    python3 scripts/check_column_map.py
    python3 scripts/check_column_map.py var/column_map.tsv

出るもの:

- 表の SHA-256（実行のたびに判断ログへ残る指紋と同じもの）
- 受容細胞の数（輝度を受け取る型、青／緑を受け取る型、どちらでもない型）
- 座標の欠けている数。**欠けたものには値を作らない**
- 画像に置いたときの、1 個あたりが見る範囲

表の形（1 行目は見出し、タブ区切り）:

    body_id	cell_type	hex1	hex2
    12345	R1	3	-2
    12346	R8	3	-2
    12347	R2	null	null

座標が取れなかった行は `null`（または空欄）にする。埋めない。
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dopabae import retina, vision  # noqa: E402
from dopabae.config import REPO_ROOT, load  # noqa: E402


def main(argv: list[str]) -> int:
    config = load()
    target = Path(argv[1]) if len(argv) > 1 else REPO_ROOT / config.retina_column_map_path

    try:
        column_map = retina.load_column_map(target)
    except retina.RetinaError as exc:
        print(f"読めません: {exc}", file=sys.stderr)
        return 1

    types = Counter(c.cell_type for c in column_map.columns)
    luminance = {t: n for t, n in types.items() if t in config.retina_luminance_types}
    blue_green = {t: n for t, n in types.items() if t in config.retina_blue_green_types}
    other = {
        t: n
        for t, n in types.items()
        if t not in config.retina_luminance_types and t not in config.retina_blue_green_types
    }

    print(f"表: {target}")
    print(f"SHA-256: {column_map.sha256}")
    print(f"行数: {len(column_map.columns)}")
    print(f"座標あり: {len(column_map.mapped)} / 座標なし: {column_map.unmapped_count}")
    print()
    print(f"輝度を受け取る型（R1–R6）: {sum(luminance.values())}")
    for name, count in sorted(luminance.items()):
        print(f"  {name}: {count}")
    print(f"青／緑を受け取る型（R8）: {sum(blue_green.values())}  channel={config.retina_r8_channel}")
    for name, count in sorted(blue_green.items()):
        print(f"  {name}: {count}")
    if other:
        print(f"どちらでもない型（値を作らない）: {sum(other.values())}")
        for name, count in sorted(other.items())[:10]:
            print(f"  {name}: {count}")
        if len(other) > 10:
            print(f"  ... ほか {len(other) - 10} 型")
    print()

    if not luminance and not blue_green:
        print(
            "型名が 1 つも噛み合いませんでした。"
            "agent.yaml の fly.retina.luminance_types / blue_green_types を、"
            "表にある型名へ合わせる",
            file=sys.stderr,
        )
        return 1

    # 画像に置いたときの粗さを見せる。空の画像で構わない（範囲は座標だけで決まる）。
    blank = vision.Image(
        width=config.vision_width,
        height=config.vision_height,
        pixels=bytes(config.vision_width * config.vision_height * 3),
        sha256="",
        candle_type=config.vision_candle_type,
        candle_count=0,
        candles_from_ms=None,
        candles_to_ms=None,
        shown=(),
    )
    try:
        built = retina.fields(config, column_map, blank)
    except retina.RetinaError as exc:
        print(f"範囲を決められません: {exc}", file=sys.stderr)
        return 1

    nominal = retina._field_size(config, retina._centers(config, column_map, blank), blank)
    spread = len({(f.x0, f.y0) for f in built})
    print(f"画像: {blank.width}×{blank.height}")
    print(f"1 個が見る範囲: {nominal[0]}×{nominal[1]} 画素（縁では切り取られる）")
    print(f"中心の重複していない位置: {spread} / {len(built)}")
    if nominal[0] * nominal[1] <= 1:
        print(
            "範囲が 1 画素しかない。気配の線は 1 画素の破線なので、"
            "当たるかどうかが運で決まる。fly.retina.field_px で広げる",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
