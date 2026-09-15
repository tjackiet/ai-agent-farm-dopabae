"""運用実績の集計。

総資産は判断ログに残した**観測値**（`paper assets` 由来）から取る。
現金と建玉を積み直して復元しない。復元すると、観測していない値を作ることになる
（`CLAUDE.md`「観測していない値を書かない」）。

この物差しは評価（`evaluate.py`）が腕ごとに使う。同じ関数で測らないと、
腕の差が実装の差なのか挙動の差なのか分からなくなる。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Sequence

import yaml

from . import journal, timeutil
from .config import Config, REPO_ROOT


@dataclass(frozen=True)
class Point:
    """ある回に観測した総資産。すべて判断ログに残っている値。"""

    at: datetime
    price: Decimal
    equity_jpy: Decimal
    position: Decimal


def points_from(records: Sequence[dict], tz_name: str) -> list[Point]:
    """判断ログから、総資産を観測できた回だけを取り出す。

    観測に失敗した回は `price` も `account` も欠けるので飛ばす。
    飛ばした回を前の値で埋めない。
    """
    points: list[Point] = []
    for record in records:
        run_id = record.get("run_id")
        price = record.get("price")
        account = record.get("account")
        if not isinstance(run_id, str) or not isinstance(price, (int, float)):
            continue
        if not isinstance(account, dict):
            continue
        equity = account.get("equity_jpy")
        if not isinstance(equity, (int, float)):
            continue
        position = record.get("position") or {}
        amount = position.get("amount") if isinstance(position, dict) else None
        try:
            at = timeutil.from_iso(run_id, tz_name)
        except ValueError:
            continue
        points.append(
            Point(
                at=at,
                price=Decimal(str(price)),
                equity_jpy=Decimal(str(equity)),
                position=Decimal(str(amount)) if isinstance(amount, (int, float)) else Decimal(0),
            )
        )
    return sorted(points, key=lambda p: p.at)


def max_drawdown_pct(points: Sequence[Point]) -> Decimal:
    """過去最高資産からの最大下落率。正の数で返す。"""
    peak = Decimal(0)
    worst = Decimal(0)
    for point in points:
        peak = max(peak, point.equity_jpy)
        if peak > 0:
            worst = max(worst, (peak - point.equity_jpy) / peak * Decimal(100))
    return worst


def buy_and_hold_pct(points: Sequence[Point]) -> Decimal | None:
    """同じ期間、最初の観測価格で買って持ち続けた場合の騰落率。

    ハエの成績はこれと比べる。上げ相場ではどんな買い手も上手に見えるため
    （Stonkfly `docs/model.md` の警告）。
    """
    if len(points) < 2 or points[0].price <= 0:
        return None
    return (points[-1].price - points[0].price) / points[0].price * Decimal(100)


def _pct(value: Decimal | None, digits: int = 2) -> float | None:
    return None if value is None else round(float(value), digits)


@dataclass(frozen=True)
class Performance:
    """成績。観測できた回が少なければ、どの数字も意味を持たない。"""

    observations: int
    first_at: str | None
    last_at: str | None
    initial_equity_jpy: float | None
    final_equity_jpy: float | None
    total_return_pct: float | None
    max_drawdown_pct: float | None
    buy_and_hold_pct: float | None
    excess_vs_buy_and_hold_pct: float | None

    def as_dict(self) -> dict:
        return {
            "observations": self.observations,
            "first_at": self.first_at,
            "last_at": self.last_at,
            "initial_equity_jpy": self.initial_equity_jpy,
            "final_equity_jpy": self.final_equity_jpy,
            "total_return_pct": self.total_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "buy_and_hold_pct": self.buy_and_hold_pct,
            "excess_vs_buy_and_hold_pct": self.excess_vs_buy_and_hold_pct,
        }


def measure(points: Sequence[Point]) -> Performance:
    """点列から成績を出す。点が無ければ、すべて None を返す。"""
    if not points:
        return Performance(0, None, None, None, None, None, None, None, None)
    first, last = points[0], points[-1]
    total = (
        (last.equity_jpy - first.equity_jpy) / first.equity_jpy * Decimal(100)
        if first.equity_jpy > 0
        else None
    )
    bh = buy_and_hold_pct(points)
    excess = total - bh if total is not None and bh is not None else None
    return Performance(
        observations=len(points),
        first_at=timeutil.to_iso(first.at),
        last_at=timeutil.to_iso(last.at),
        initial_equity_jpy=float(first.equity_jpy),
        final_equity_jpy=float(last.equity_jpy),
        total_return_pct=_pct(total),
        max_drawdown_pct=_pct(max_drawdown_pct(points)),
        buy_and_hold_pct=_pct(bh),
        excess_vs_buy_and_hold_pct=_pct(excess),
    )


def build(config: Config, now: datetime, records: Sequence[dict]) -> dict:
    """運用実績の文書。`agent.performance_output` へ書き出す。"""
    points = points_from(records, config.timezone)
    document = {
        "schema_version": 1,
        "agent_id": config.agent_id,
        "agent_version": config.version,
        "recorded_at": timeutil.to_iso(now),
        "source": "paper",
        "direction_source": config.direction_source,
    }
    document.update(measure(points).as_dict())
    return document


def write(document: dict, path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# ペーパートレード実績（運用の産物。Git 管理外）\n"
        "#\n"
        "# 総資産は判断ログに残した観測値（paper assets 由来）から取っています。\n"
        "# 観測できなかった回は飛ばし、前の値で埋めていません。\n"
    )
    body = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
    target.write_text(f"{header}\n{body}", encoding="utf-8")
    return target


def all_records(config: Config, root: Path | None = None) -> list[dict]:
    """残っているすべての判断ログを、古い順に読む。"""
    records: list[dict] = []
    for date in journal.recorded_dates(config, root):
        records.extend(journal.read_day(config, date, root))
    return records


def refresh(config: Config, now: datetime, root: Path | None = None) -> Path:
    base = root if root is not None else REPO_ROOT
    document = build(config, now, all_records(config, root))
    return write(document, base / config.performance_output)
