#!/usr/bin/env python3
"""MaleCNS v1.0 から光受容細胞のカラム座標表を書き出す。

`scripts/fetch_connectome.py` で取ったデータを読み、`fly.retina.column_map_path`
（既定 `var/column_map.tsv`、Git 管理外）へ `dopabae/retina.py` が読む形で書く。

**光受容細胞そのものは `assignedOlHex1` / `assignedOlHex2` を持たない**
（2026-10-05 確認。R1-R6 も R7 / R8 も 0 個）。座標を持つのは L1・L2・Mi1・Tm20 など
視葉の細胞である。そこで、**光受容細胞が出力するシナプスの行き先**から決める
（`docs/CONNECTOME_SURVEY.md` 5.1 の nfly / Stonkfly の方式）。

1. 同じ側（左眼なら左の視葉）の、座標を持つ標的だけを数える
2. 標的のカラムごとにシナプス数を足す
3. いちばん多いカラムをその光受容細胞の座標にする

決められない細胞には**値を作らない**（`null`）。標的に座標を持つ細胞が無い、
1 位が同数で並ぶ、左右が分からない、のいずれか。Stonkfly はシナプス数で重み付けした
平均の座標に置くが、ここでは最多のカラムを採る。平均はカラムとカラムの間の、
格子に無い位置を作りうるためである。

**これは設計者が決めた読み替えである。** 生体の視野との一致は検証されていない。

使いかた:

    python3 -m pip install -r requirements-connectome.txt   # pyarrow
    python3 scripts/fetch_connectome.py
    python3 scripts/export_column_map.py
    python3 scripts/check_column_map.py

書き出しは決定的である。同じ入力からは同じ表（同じ SHA-256）が出る。
既にある表と中身が違えば止まる（`--force` で置き換える）。
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dopabae import connectome, retina  # noqa: E402
from dopabae.config import REPO_ROOT, Config, load  # noqa: E402

# 複眼の光受容細胞。R1-R6 / R7* / R8* の型名で、視葉の感覚細胞に分類されたもの。
PHOTORECEPTOR_TYPE = re.compile(r"^R[1-8]")
PHOTORECEPTOR_SUPERCLASS = "ol_sensory"

ANNOTATION_COLUMNS = ("bodyId", "type", "instance", "superclass", "assignedOlHex1", "assignedOlHex2")
WEIGHT_COLUMNS = ("body_pre", "body_post", "weight")

# 決められなかった理由。表には書かず、数だけを出す。
MAPPED = "mapped"
NO_TARGET = "no_target"
TIE = "tie"
NO_SIDE = "no_side"
REASONS = (MAPPED, NO_TARGET, TIE, NO_SIDE)


class ExportError(Exception):
    """書き出せない（入力が無い・指紋が合わない・形が想定と違う）。"""


@dataclass(frozen=True)
class Receptor:
    body_id: int
    cell_type: str
    side: str | None


@dataclass(frozen=True)
class HexCell:
    side: str
    hex1: int
    hex2: int


@dataclass(frozen=True)
class Assignment:
    body_id: int
    cell_type: str
    side: str | None
    hex1: int | None
    hex2: int | None
    reason: str
    # 座標を持つ標的へのシナプスのうち、選んだカラムへ行った割合。決めた細胞だけ。
    share: float | None


def side_of(instance: object) -> str | None:
    """`instance` の末尾（`_L` / `_R`）から左右を読む。読めなければ None。"""
    if isinstance(instance, str):
        if instance.endswith("_L"):
            return "L"
        if instance.endswith("_R"):
            return "R"
    return None


def to_hex(value: object, body_id: int) -> int:
    """カラム座標は整数のはず。小数部があれば形が想定と違うので止める。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportError(f"body {body_id} のカラム座標が数値ではありません: {value!r}")
    if value != int(value):
        raise ExportError(f"body {body_id} のカラム座標が整数ではありません: {value!r}")
    return int(value)


def assign(
    receptors: Mapping[int, Receptor],
    hex_cells: Mapping[int, HexCell],
    edges: Iterable[tuple[int, int, int]],
) -> tuple[list[Assignment], int]:
    """光受容細胞ごとに、出力シナプスがいちばん多く行くカラムを選ぶ。

    返すのは body_id 順の割り当てと、反対側の標的へ行ったため数えなかったシナプス数。
    """
    per_column: dict[int, Counter[tuple[int, int]]] = defaultdict(Counter)
    cross_side = 0
    for pre, post, weight in edges:
        receptor = receptors.get(pre)
        target = hex_cells.get(post)
        if receptor is None or target is None or weight <= 0 or receptor.side is None:
            continue
        if target.side != receptor.side:
            cross_side += weight
            continue
        per_column[pre][(target.hex1, target.hex2)] += weight

    out: list[Assignment] = []
    for body_id in sorted(receptors):
        receptor = receptors[body_id]
        columns = per_column.get(body_id)
        if receptor.side is None:
            reason, best, share = NO_SIDE, None, None
        elif not columns:
            reason, best, share = NO_TARGET, None, None
        else:
            # 同数のときに並びで勝ち負けが決まらないよう、座標でも並べてから比べる。
            ranked = sorted(columns.items(), key=lambda item: (-item[1], item[0]))
            if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                reason, best, share = TIE, None, None
            else:
                reason, best = MAPPED, ranked[0][0]
                share = ranked[0][1] / sum(columns.values())
        out.append(
            Assignment(
                body_id=body_id,
                cell_type=receptor.cell_type,
                side=receptor.side,
                hex1=None if best is None else best[0],
                hex2=None if best is None else best[1],
                reason=reason,
                share=share,
            )
        )
    return out, cross_side


def render_tsv(assignments: Iterable[Assignment]) -> bytes:
    """`dopabae/retina.py` が読む形。決められなかった座標は `null` のまま。"""
    lines = ["\t".join(retina.COLUMNS)]
    for a in assignments:
        hex1 = "null" if a.hex1 is None else str(a.hex1)
        hex2 = "null" if a.hex2 is None else str(a.hex2)
        lines.append(f"{a.body_id}\t{a.cell_type}\t{hex1}\t{hex2}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def verified(config: Config, name: str) -> Path:
    """`agent.yaml` の指紋と一致する入力だけを返す。"""
    try:
        return connectome.verified_input(config, name)
    except connectome.ConnectomeError as exc:
        raise ExportError(str(exc)) from exc


def read_inputs(
    annotations: Path, weights: Path
) -> tuple[dict[int, Receptor], dict[int, HexCell], list[tuple[int, int, int]]]:
    """配線図を読む。pyarrow はここでだけ使う（テストと定期実行には要らない）。"""
    try:
        import pyarrow.dataset as ds
        import pyarrow.feather as feather
    except ImportError as exc:
        raise ExportError(
            "pyarrow が要ります: python3 -m pip install -r requirements-connectome.txt"
        ) from exc

    table = feather.read_table(annotations, columns=list(ANNOTATION_COLUMNS))
    receptors: dict[int, Receptor] = {}
    hex_cells: dict[int, HexCell] = {}
    for row in table.to_pylist():
        body_id = int(row["bodyId"])
        cell_type = row["type"]
        if (
            isinstance(cell_type, str)
            and PHOTORECEPTOR_TYPE.match(cell_type)
            and row["superclass"] == PHOTORECEPTOR_SUPERCLASS
        ):
            receptors[body_id] = Receptor(body_id, cell_type, side_of(row["instance"]))
        hex1, hex2 = row["assignedOlHex1"], row["assignedOlHex2"]
        side = side_of(row["instance"])
        if hex1 is not None and hex2 is not None and side is not None:
            hex_cells[body_id] = HexCell(side, to_hex(hex1, body_id), to_hex(hex2, body_id))

    if not receptors:
        raise ExportError("光受容細胞が 1 つも見つかりません。注釈の形が想定と違う")
    if not hex_cells:
        raise ExportError("カラム座標を持つ細胞が 1 つも見つかりません。注釈の形が想定と違う")

    # 1 億 5 千万行を丸ごと読まない。光受容細胞から出る結合だけを取り出す。
    dataset = ds.dataset(weights, format="ipc")
    edges_table = dataset.to_table(
        columns=list(WEIGHT_COLUMNS),
        filter=ds.field("body_pre").isin(sorted(receptors)),
    )
    edges = list(
        zip(
            edges_table.column("body_pre").to_pylist(),
            edges_table.column("body_post").to_pylist(),
            edges_table.column("weight").to_pylist(),
        )
    )
    return receptors, hex_cells, edges


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def report(assignments: list[Assignment], cross_side: int, config: Config) -> None:
    """型ごとの内訳。**観測した数だけを出す。** 良し悪しの判断は書かない。"""
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    shares: dict[str, list[float]] = defaultdict(list)
    sides: dict[str, Counter[str]] = defaultdict(Counter)
    for a in assignments:
        by_type[a.cell_type][a.reason] += 1
        if a.reason == MAPPED and a.share is not None:
            shares[a.cell_type].append(a.share)
            sides[a.cell_type][a.side or "?"] += 1

    used = set(config.retina_luminance_types) | set(config.retina_blue_green_types)
    print("型ごとの内訳（* は fly.retina で値を作る型）")
    print("  型            計   決定  標的なし  同数  左右不明   左/右(決定)   最多カラムの割合 p10 / 中央")
    for cell_type in sorted(by_type):
        counts = by_type[cell_type]
        total = sum(counts.values())
        values = shares.get(cell_type, [])
        share = f"{_quantile(values, 0.1):.2f} / {_quantile(values, 0.5):.2f}" if values else "-"
        mark = "*" if cell_type in used else " "
        lr = f"{sides[cell_type]['L']}/{sides[cell_type]['R']}"
        print(
            f"{mark} {cell_type:<12}{total:>5}{counts[MAPPED]:>6}{counts[NO_TARGET]:>9}"
            f"{counts[TIE]:>6}{counts[NO_SIDE]:>9}   {lr:>11}   {share}"
        )
    print(f"反対側の標的へ行ったため数えなかったシナプス: {cross_side}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="書き出し先（既定は fly.retina.column_map_path）")
    parser.add_argument("--force", action="store_true", help="既にある表と中身が違っても置き換える")
    args = parser.parse_args(argv)

    config = load()
    target = args.out or REPO_ROOT / config.retina_column_map_path
    try:
        annotations = verified(config, "annotations")
        weights = verified(config, "weights")
        receptors, hex_cells, edges = read_inputs(annotations, weights)
    except ExportError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    assignments, cross_side = assign(receptors, hex_cells, edges)
    blob = render_tsv(assignments)
    digest = hashlib.sha256(blob).hexdigest()

    if target.exists() and target.read_bytes() != blob and not args.force:
        print(
            f"既にある表と中身が違います: {target}\n"
            "置き換えるなら --force を付ける（既にある表の SHA-256 は check_column_map.py で見られる）",
            file=sys.stderr,
        )
        return 1

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(blob)
    part.replace(target)

    # 書いたものを retina.py 自身に読ませて、形が噛み合うことを確かめる。
    column_map = retina.load_column_map(target)

    print(f"入力: annotations {config.connectome_files['annotations'].sha256}")
    print(f"      weights     {config.connectome_files['weights'].sha256}")
    print(f"光受容細胞: {len(receptors)} / 座標を持つ細胞: {len(hex_cells)} / 結合: {len(edges)}")
    report(assignments, cross_side, config)
    print()
    print(f"表: {target}")
    print(f"SHA-256: {digest}")
    print(f"行数: {len(column_map.columns)}（座標あり {len(column_map.mapped)} / なし {column_map.unmapped_count}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
