"""日次サマリ。判断ログから、その日に起きたことをまとめる。

**ハエの言葉と観測を混ぜない**（`personality.md`「数字の扱い」）。
本文はドパバエの言葉だけで書き、数字は別の節へ判断ログからそのまま転記する。
本文は空欄（`（未記入）`）で作り、`narrate.py` があとから埋める。埋められなければ
空欄のまま残り、観測だけは残る。**数字が残るほうが、言葉が残るより大事である。**

ナンピノニクスにある「学び」は作らない。ドパバエは反省も計画もしないし
（`personality.md`「やらないこと」）、学習が起きるのはシナプス重みだけである
（`CLAUDE.md`）。言葉で「次はこうする」と書けば、どちらの設計にも反する。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

from . import journal, timeutil
from .config import Config, REPO_ROOT
from .direction import APPROACH, AVOID, NONE

UNWRITTEN = "（未記入）"

# 方向の見出し。`personality.md` の対応表と揃える。
DIRECTION_LABELS = (
    (APPROACH, "近づいた"),
    (AVOID, "逃げた"),
    (NONE, "じっとしてた"),
)

# 檻の扱いの見出し。`cage.py` の定数と対応する。
CAGE_LABELS = {
    "bought": "買い指値を置いた",
    "sold": "売り指値を置いた",
    "wall": "壁で止まった",
    "cooldown": "クールダウン中だった",
    "avoid_flat": "逃げたが建玉がなかった",
    "still": "じっとしてた",
    "no_direction": "方向を得られなかった",
    "exit": "檻が手仕舞った",
    "recover": "止まっていたので指値を取り消した",
    "guard": "観測できず何もしなかった",
    "dust": "刻みに満たず出さなかった",
}


def daily_path(config: Config, date: str, root: Path | None = None) -> Path:
    base = root if root is not None else REPO_ROOT
    return base / config.daily_path.format(date=date)


def _counts(records: Sequence[dict]) -> tuple[dict[str, int], int, int, dict[str, int], int]:
    """方向・檻・画像の内訳を数える。"""
    directions = {value: 0 for value, _ in DIRECTION_LABELS}
    unavailable = not_consulted = drawn = 0
    cages: dict[str, int] = {}
    for record in records:
        direction = record.get("direction")
        if isinstance(direction, dict):
            value = direction.get("value")
            if value in directions:
                directions[value] += 1
            else:
                unavailable += 1
        else:
            not_consulted += 1
        label = record.get("cage")
        if isinstance(label, str):
            cages[label] = cages.get(label, 0) + 1
        if isinstance(record.get("vision"), dict):
            drawn += 1
    return directions, unavailable, not_consulted, cages, drawn


def _fills(records: Sequence[dict]) -> tuple[int, int]:
    """その日に**実際に出した**注文の数。dry_run の回は数えない。"""
    buys = sells = 0
    for record in records:
        for order in record.get("orders") or []:
            if not isinstance(order, dict) or order.get("op") != "place":
                continue
            if not order.get("executed"):
                continue
            if order.get("side") == "buy":
                buys += 1
            elif order.get("side") == "sell":
                sells += 1
    return buys, sells


def _equity(records: Sequence[dict]) -> tuple[float | None, float | None]:
    """その日の総資産の、最初と最後の**観測値**。観測できなければ None。"""
    values = [
        record["account"]["equity_jpy"]
        for record in records
        if isinstance(record.get("account"), dict)
        and isinstance(record["account"].get("equity_jpy"), (int, float))
    ]
    return (values[0], values[-1]) if values else (None, None)


def build_daily(config: Config, date: str, records: Sequence[dict]) -> str:
    """その日の日誌を組み立てる。本文は空欄のままにする。"""
    directions, unavailable, not_consulted, cages, drawn = _counts(records)
    buys, sells = _fills(records)
    first, last = _equity(records)
    run_ids = [r.get("run_id") for r in records if isinstance(r.get("run_id"), str)]

    lines = [
        f"# {date} — ドパバエ",
        "",
        "## おれの一日",
        "",
        UNWRITTEN,
        "",
        "## 観測",
        "",
        "ハエの言葉には数字を混ぜない。数字はここにだけ置く。",
        "",
        "| 項目 | 値 |",
        "| --- | --- |",
        f"| 回した回数 | {len(records)} |",
    ]
    for value, label in DIRECTION_LABELS:
        lines.append(f"| {label}（{value}） | {directions[value]} |")
    lines.append(f"| 方向を得られなかった | {unavailable} |")
    if not_consulted:
        lines.append(f"| 方向を諮らなかった（観測に失敗） | {not_consulted} |")
    lines.append(f"| 画像を描けた回 | {drawn} |")
    lines.append(f"| 出した買い注文 | {buys} |")
    lines.append(f"| 出した売り注文 | {sells} |")

    if first is not None and last is not None:
        lines.append(f"| 総資産（最初の観測） | {first:,.0f} |")
        lines.append(f"| 総資産（最後の観測） | {last:,.0f} |")
        if first > 0:
            lines.append(f"| 増減 | {(last - first) / first * 100:+.2f}% |")
    else:
        # 観測できなかった値は null のままにする（CLAUDE.md）。
        lines.append("| 総資産 | 観測できなかった |")

    if cages:
        lines.extend(["", "### 檻の扱い", "", "| 扱い | 回数 |", "| --- | --- |"])
        for label, count in sorted(cages.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"| {CAGE_LABELS.get(label, label)}（{label}） | {count} |")

    source = config.decisions_path.format(date=date)
    span = f"{run_ids[0]} 〜 {run_ids[-1]}" if run_ids else "（記録なし）"
    lines.extend(["", f"出典: `{source}`（{len(records)} 件、{span}）", ""])
    return "\n".join(lines)


def ensure(config: Config, now: datetime, root: Path | None = None) -> list[Path]:
    """書けるようになった日誌を書き出す。書いたファイルを返す。

    **既にあるファイルは触らない。** あとから埋めた本文を消さないため。
    その日が終わるまでは書かない。途中で書くと、半端な一日をハエに語らせ、
    あとから書き直すこともできなくなる。
    """
    written: list[Path] = []
    today = timeutil.date_key(now)
    for date in journal.recorded_dates(config, root):
        if date == today and now.strftime("%H:%M") < config.daily_write_at:
            continue
        path = daily_path(config, date, root)
        if path.exists():
            continue
        records = journal.read_day(config, date, root)
        if not records:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build_daily(config, date, records), encoding="utf-8")
        written.append(path)
    return written
