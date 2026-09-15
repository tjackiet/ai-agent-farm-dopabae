"""判断ログの追記。

action が HOLD でも必ず1行残す。何もしなかったことも判断である。
ハエの答えと、檻がそれをどう扱ったかを別々に残す（docs/DESIGN_MEMO.md 3.6 節）。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

from . import timeutil
from .cli import Source
from .config import Config, REPO_ROOT


def path_for(config: Config, now: datetime, root: Path | None = None) -> Path:
    base = root if root is not None else REPO_ROOT
    return base / config.decisions_path.format(date=timeutil.date_key(now))


def append(config: Config, now: datetime, record: dict, root: Path | None = None) -> Path:
    target = path_for(config, now, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False))
        handle.write("\n")
    return target


def read_day(config: Config, date: str, root: Path | None = None) -> list[dict]:
    """その日の判断ログを読む。壊れている行は飛ばす。"""
    base = root if root is not None else REPO_ROOT
    path = base / config.decisions_path.format(date=date)
    if not path.is_file():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def read_recent(config: Config, now: datetime, limit: int, root: Path | None = None) -> list[dict]:
    """直近の判断ログを古い順に返す。日付をまたぐため前日ぶんも読む。"""
    if limit <= 0:
        return []
    records: list[dict] = []
    for day in (now - timedelta(days=1), now):
        records.extend(read_day(config, timeutil.date_key(day), root))
    return records[-limit:]


def build_record(
    *,
    run_id: str,
    config_version: str | None,
    fingerprint: str | None,
    state_label: str,
    pair: str,
    market: object | None,
    position_amount: float | None,
    avg_cost: float | None,
    direction: dict | None,
    cage: str | None,
    action: str,
    reason: str,
    orders: Sequence[dict],
    sources: Sequence[Source],
    warnings: Sequence[str] = (),
    error: str | None = None,
) -> dict:
    return {
        "run_id": run_id,
        # その回どの設定で動いていたか。値を変えた前後のラウンドを混ぜないために残す。
        "config": {"version": config_version, "fingerprint": fingerprint},
        "state": state_label,
        "pair": pair,
        "price": float(market.last) if market is not None else None,
        "bid": float(market.bid) if market is not None else None,
        "ask": float(market.ask) if market is not None else None,
        "position": {"amount": position_amount, "avg_cost": avg_cost},
        # ハエの答え（観測値）と、檻の扱い（決定的コードの結果）は分けて残す。
        "direction": direction,
        "cage": cage,
        "action": action,
        "reason": reason,
        "orders": list(orders),
        "sources": [s.as_dict() for s in sources],
        "warnings": list(warnings),
        "error": error,
    }
