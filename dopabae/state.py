"""現在状態の導出と status.yaml の書き出し。

状態は記憶せず、bitbank paper の実測から毎回導出する。
ナンピノニクスの `state.py` から階段（Ladder）を外し、檻が使う
「直近の約定・クールダウン・当日の約定回数」だけを `Activity` として残した。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Sequence

import yaml

from . import timeutil
from .config import Config, REPO_ROOT
from .orders import to_decimal

# 建玉ゼロとみなす許容誤差。刻み（0.0001）より十分小さくとる。
DUST = Decimal("0.00000001")


@dataclass(frozen=True)
class Trade:
    id: str
    side: str
    order_type: str
    amount: Decimal
    fill_price: Decimal
    fee_quote: Decimal
    filled_at: datetime


@dataclass(frozen=True)
class OpenOrder:
    id: str
    side: str
    price: Decimal
    amount: Decimal
    created_at: datetime


@dataclass(frozen=True)
class Position:
    amount: Decimal
    avg_cost_jpy: Decimal | None
    opened_at: datetime | None
    age_days: float | None

    @property
    def cost_basis_jpy(self) -> Decimal:
        if self.avg_cost_jpy is None:
            return Decimal(0)
        return self.amount * self.avg_cost_jpy


@dataclass(frozen=True)
class Activity:
    """檻のペース制御に要る、直近の約定の情報。"""

    last_fill_price_jpy: Decimal | None
    last_fill_at: datetime | None
    cooldown_until: datetime | None
    fills_today: int
    buys_in_round: int


@dataclass(frozen=True)
class Account:
    initial_jpy: Decimal
    cash_total_jpy: Decimal
    cash_locked_jpy: Decimal
    cash_available_jpy: Decimal
    base_total: Decimal
    base_available: Decimal
    equity_jpy: Decimal

    @property
    def drawdown_pct(self) -> Decimal:
        if self.initial_jpy <= 0:
            return Decimal(0)
        return (self.equity_jpy - self.initial_jpy) / self.initial_jpy * Decimal(100)


@dataclass(frozen=True)
class State:
    position: Position
    activity: Activity
    account: Account
    pending_buy: tuple[OpenOrder, ...]
    pending_sell: tuple[OpenOrder, ...]
    realized_pnl_jpy: Decimal
    unrealized_pnl_jpy: Decimal
    position_mismatch: bool


def last_tick_at(config: Config, root: Path | None = None) -> datetime | None:
    """ペーパー口座が最後に約定判定をした時刻を読む。分からなければ None。"""
    base = root if root is not None else REPO_ROOT
    path = Path(config.state_path)
    if not path.is_absolute():
        path = base / path
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = raw.get("lastTickAt") if isinstance(raw, dict) else None
    if not isinstance(value, str):
        return None
    try:
        return timeutil.from_iso(value, config.timezone)
    except ValueError:
        return None


def stopped_hours(config: Config, now: datetime, root: Path | None = None) -> float | None:
    """前回の約定判定から何時間空いたか。分からなければ None。"""
    previous = last_tick_at(config, root)
    if previous is None:
        return None
    return max(0.0, (now - previous).total_seconds() / 3600)


def parse_trades(rows: Sequence[dict], pair: str, tz_name: str) -> tuple[Trade, ...]:
    trades = [
        Trade(
            id=str(row["id"]),
            side=str(row["side"]),
            order_type=str(row["type"]),
            amount=to_decimal(row["amount"]),
            fill_price=to_decimal(row["fillPrice"]),
            fee_quote=to_decimal(row.get("feeQuote", 0)),
            filled_at=timeutil.from_iso(str(row["filledAt"]), tz_name),
        )
        for row in rows
        if row.get("pair") == pair
    ]
    return tuple(sorted(trades, key=lambda t: t.filled_at))


def parse_open_orders(rows: Sequence[dict], pair: str, tz_name: str) -> tuple[OpenOrder, ...]:
    return tuple(
        OpenOrder(
            id=str(row["id"]),
            side=str(row["side"]),
            price=to_decimal(row["price"]),
            amount=to_decimal(row["amount"]),
            created_at=timeutil.from_iso(str(row["createdAt"]), tz_name),
        )
        for row in rows
        if row.get("pair") == pair
    )


def sellable(position: Decimal, balance: Decimal, unit: Decimal | None) -> Decimal:
    """約定履歴の建玉と口座残高の少ないほうを、取引単位へ切り捨てた数量。

    残高は浮動小数点で積まれるため、約定履歴から数えた建玉とわずかにずれる。
    発注は残高に対して検証されるので、少ないほうに合わせなければ弾かれる。
    CLI は取引単位の倍数でない数量を丸めずに拒否するため、切り捨てる。
    """
    held = min(position, balance)
    if unit is None or unit <= 0:
        return held if held > DUST else Decimal(0)
    held = (held / unit).to_integral_value(rounding=ROUND_FLOOR) * unit
    return held if held >= unit else Decimal(0)


def is_flat(position: Decimal, unit: Decimal | None) -> bool:
    """その建玉を、もう無いものとして扱ってよいか（売れない端数は無いものとする）。"""
    if unit is None or unit <= 0:
        return position <= DUST
    return sellable(position, position - DUST, unit) <= 0


def sellable_now(state: State, unit: Decimal | None, freed: Decimal = Decimal(0)) -> Decimal:
    """いま新しく売りに出せる数量。`freed` は同じ回で取り消す売り指値が解放するぶん。"""
    return sellable(state.position.amount, state.account.base_available + freed, unit)


def current_round(trades: Sequence[Trade], unit: Decimal | None = None) -> tuple[Trade, ...]:
    """いまの建玉ラウンド（買い始めてから売り切るまで）の約定だけを取り出す。"""
    position = Decimal(0)
    round_trades: list[Trade] = []
    for trade in trades:
        if is_flat(position, unit) and trade.side == "buy":
            round_trades = []
        round_trades.append(trade)
        position += trade.amount if trade.side == "buy" else -trade.amount
        if is_flat(position, unit):
            position = Decimal(0)
            round_trades = []
    return tuple(round_trades)


@dataclass(frozen=True)
class Round:
    """建玉1回ぶん。買い始めてから、全部売り切るまで。報酬（Phase 5）はここから取る。"""

    opened_at: datetime
    closed_at: datetime | None
    buys: int
    amount: Decimal
    cost_jpy: Decimal
    proceeds_jpy: Decimal
    avg_cost_jpy: Decimal
    realized_pnl_jpy: Decimal

    @property
    def is_closed(self) -> bool:
        return self.closed_at is not None


def rounds(trades: Sequence[Trade], unit: Decimal | None = None) -> tuple[Round, ...]:
    """約定履歴を建玉のラウンドごとに区切る。計算は `bitbank paper pnl` に合わせる。"""
    result: list[Round] = []
    current: list[Trade] = []
    position = Decimal(0)
    for trade in trades:
        if is_flat(position, unit) and trade.side == "buy":
            current = []
        current.append(trade)
        position += trade.amount if trade.side == "buy" else -trade.amount
        if is_flat(position, unit) and current:
            result.append(_build_round(current, closed=True))
            current = []
            position = Decimal(0)
    if current:
        result.append(_build_round(current, closed=False))
    return tuple(result)


def _build_round(trades: Sequence[Trade], closed: bool) -> Round:
    position = Decimal(0)
    avg_cost = Decimal(0)
    cost = Decimal(0)
    proceeds = Decimal(0)
    realized = Decimal(0)
    buys = 0
    for trade in trades:
        if trade.side == "buy":
            buys += 1
            per_unit_fee = trade.fee_quote / trade.amount if trade.amount else Decimal(0)
            new_position = position + trade.amount
            avg_cost = (avg_cost * position + (trade.fill_price + per_unit_fee) * trade.amount) / new_position
            position = new_position
            cost += trade.fill_price * trade.amount + trade.fee_quote
        else:
            realized += (trade.fill_price - avg_cost) * trade.amount - trade.fee_quote
            proceeds += trade.fill_price * trade.amount - trade.fee_quote
            position -= trade.amount
    return Round(
        opened_at=trades[0].filled_at,
        closed_at=trades[-1].filled_at if closed else None,
        buys=buys,
        amount=sum((t.amount for t in trades if t.side == "buy"), Decimal(0)),
        cost_jpy=cost,
        proceeds_jpy=proceeds,
        avg_cost_jpy=avg_cost,
        realized_pnl_jpy=realized,
    )


def derive_activity(
    trades: Sequence[Trade], config: Config, now: datetime, flat: bool, unit: Decimal | None
) -> Activity:
    """直近の約定から、ペース制御に要る値を出す。`flat` なら今のラウンドは無い。"""
    round_trades = () if flat else current_round(trades, unit)
    buys = [t for t in round_trades if t.side == "buy"]
    # クールダウンはラウンドをまたいでも効かせる。売り切った直後に買い直すのを防ぐ。
    last_buy = next((t for t in reversed(trades) if t.side == "buy"), None)
    cooldown = (
        last_buy.filled_at + timedelta(hours=config.cooldown_hours_after_fill) if last_buy else None
    )
    today = timeutil.date_key(now)
    fills_today = sum(1 for t in trades if t.side == "buy" and timeutil.date_key(t.filled_at) == today)
    return Activity(
        last_fill_price_jpy=last_buy.fill_price if last_buy else None,
        last_fill_at=last_buy.filled_at if last_buy else None,
        cooldown_until=cooldown,
        fills_today=fills_today,
        buys_in_round=len(buys),
    )


def derive_position(
    trades: Sequence[Trade], pnl_row: dict | None, now: datetime, amount: Decimal, unit: Decimal | None
) -> Position:
    round_trades = current_round(trades, unit)
    opened_at = round_trades[0].filled_at if round_trades else None
    avg_cost = to_decimal(pnl_row["avgCost"]) if pnl_row and amount > DUST else None
    age_days = (now - opened_at).total_seconds() / 86400 if opened_at and amount > DUST else None
    return Position(
        amount=amount if amount > DUST else Decimal(0),
        avg_cost_jpy=avg_cost,
        opened_at=opened_at if amount > DUST else None,
        age_days=age_days,
    )


def derive(
    config: Config,
    now: datetime,
    last_price: Decimal,
    assets_rows: Sequence[dict],
    pnl_report: dict,
    order_rows: Sequence[dict],
    history_rows: Sequence[dict],
    unit_amount: Decimal | None = None,
) -> State:
    """CLI の実測から現在状態を組み立てる。"""
    trades = parse_trades(history_rows, config.pair, config.timezone)
    orders = parse_open_orders(order_rows, config.pair, config.timezone)

    per_pair = pnl_report.get("perPair") if isinstance(pnl_report, dict) else None
    # 建玉ゼロかつ実現損益ゼロのペアは出力されない。無い＝建玉なし。
    pnl_row = per_pair.get(config.pair) if isinstance(per_pair, dict) else None

    quote = config.pair.split("_")[-1]
    base = config.pair.split("_")[0]
    cash_total = cash_locked = cash_available = Decimal(0)
    base_total = base_available = Decimal(0)
    for row in assets_rows:
        if row.get("asset") == quote:
            cash_total = to_decimal(row["total"])
            cash_locked = to_decimal(row["locked"])
            cash_available = to_decimal(row["available"])
        elif row.get("asset") == base:
            base_total = to_decimal(row["total"])
            base_available = to_decimal(row.get("available", row["total"]))

    unit = to_decimal(unit_amount) if unit_amount is not None else None
    raw_position = to_decimal(pnl_row["position"]) if pnl_row else Decimal(0)
    if raw_position < 0:
        # paper は現物のみ。負の建玉は CLI 側でもエラーになる（docs/DESIGN_MEMO.md 6 章）。
        raise ValueError(f"負の建玉が観測された（現物のみのはず）: {raw_position}")
    # 建玉の数量はロックを引く前の残高で決める。発注できる量は `sellable_now` で別に求める。
    held = sellable(raw_position, base_total, unit)
    position = derive_position(trades, pnl_row, now, held, unit)
    activity = derive_activity(trades, config, now, flat=held <= 0, unit=unit)
    account = Account(
        initial_jpy=to_decimal(config.initial_jpy),
        cash_total_jpy=cash_total,
        cash_locked_jpy=cash_locked,
        cash_available_jpy=cash_available,
        base_total=base_total,
        base_available=base_available,
        # 総資産は実残高で評価する。売れない端数も資産ではある。
        equity_jpy=cash_total + base_total * last_price,
    )
    return State(
        position=position,
        activity=activity,
        account=account,
        pending_buy=tuple(o for o in orders if o.side == "buy"),
        pending_sell=tuple(o for o in orders if o.side == "sell"),
        realized_pnl_jpy=to_decimal(pnl_row["realizedPnl"]) if pnl_row else Decimal(0),
        unrealized_pnl_jpy=to_decimal(pnl_row["unrealizedPnl"]) if pnl_row else Decimal(0),
        # 端数のずれは日常的に起きる。1単位を超えて食い違ったときだけ異常とする。
        position_mismatch=abs(base_total - raw_position) > (unit or DUST),
    )


# 表示用の気分。personality.md「状態別のセリフ」の見出しに対応する。
MOOD_BY_STATE = {
    "NOT_INITIALIZED": "ドパ切れ",
    "IDLE": "ドパ切れ",
    "HOLDING": "期待",
    "HIBERNATING": "不満",
    "HALTED": "停止",
}


def count_closed_positions(trades: Sequence[Trade], unit: Decimal | None = None) -> int:
    """建玉がゼロへ戻った回数。区切りかたは `rounds` と揃える。"""
    return sum(1 for r in rounds(trades, unit) if r.is_closed)


def _num(value: Decimal | None, digits: int | None = None) -> float | None:
    """YAML へ書く数値。観測できなかった値は None のままにする。"""
    if value is None:
        return None
    return round(float(value), digits) if digits is not None else float(value)


def _iso(value: datetime | None) -> str | None:
    return timeutil.to_iso(value) if value is not None else None


def build_status(
    config: Config,
    now: datetime,
    run_id: str,
    state_label: str,
    market: object,
    state: State | None,
    trades: Sequence[Trade],
    action: str | None,
    reason: str | None,
    direction: dict | None,
    price_source: str | None,
    unit: Decimal | None = None,
) -> dict:
    """status.yaml の内容を組み立てる。状態の正ではなくスナップショット。"""
    document: dict = {
        "schema_version": 1,
        "updated_at": timeutil.to_iso(now),
        "run_id": run_id,
        "state": state_label,
        "mood": MOOD_BY_STATE.get(state_label, "ドパ切れ"),
        "account": {
            "initial_jpy": float(config.initial_jpy),
            "cash_jpy": None,
            "equity_jpy": None,
            "drawdown_pct": None,
        },
        "market": {
            "pair": config.pair,
            "last_price": None,
            "bid": None,
            "ask": None,
            "source": price_source,
            "fetched_at": None,
        },
        "position": {
            "amount": 0,
            "avg_cost_jpy": None,
            "cost_basis_jpy": 0,
            "unrealized_pnl_jpy": None,
            "unrealized_pnl_pct": None,
            "opened_at": None,
            "age_days": None,
            "buys_in_round": 0,
        },
        "orders": {"pending_buy": [], "pending_sell": []},
        "last_direction": direction,
        "last_decision": {"action": action, "reason": reason, "at": timeutil.to_iso(now)},
        "counters": {
            "fills_today": 0,
            "total_fills": len(trades),
            "closed_positions": count_closed_positions(trades, unit),
            "realized_pnl_jpy": 0,
            "cooldown_until": None,
        },
        "notes": None,
    }

    if market is not None:
        document["market"].update(
            {
                "last_price": _num(getattr(market, "last", None)),
                "bid": _num(getattr(market, "bid", None)),
                "ask": _num(getattr(market, "ask", None)),
                "fetched_at": _iso(getattr(market, "observed_at", None)),
            }
        )

    if state is None:
        return document

    unrealized_pct = None
    if state.position.cost_basis_jpy > 0:
        unrealized_pct = _num(state.unrealized_pnl_jpy / state.position.cost_basis_jpy * Decimal(100), 2)

    document["account"].update(
        {
            "cash_jpy": _num(state.account.cash_total_jpy),
            "equity_jpy": _num(state.account.equity_jpy),
            "drawdown_pct": _num(state.account.drawdown_pct, 2),
        }
    )
    document["position"].update(
        {
            "amount": _num(state.position.amount),
            "avg_cost_jpy": _num(state.position.avg_cost_jpy),
            "cost_basis_jpy": _num(state.position.cost_basis_jpy),
            "unrealized_pnl_jpy": _num(state.unrealized_pnl_jpy),
            "unrealized_pnl_pct": unrealized_pct,
            "opened_at": _iso(state.position.opened_at),
            "age_days": _num(to_decimal(state.position.age_days), 2)
            if state.position.age_days is not None
            else None,
            "buys_in_round": state.activity.buys_in_round,
        }
    )
    document["orders"] = {
        side: [
            {
                "order_id": o.id,
                "price": _num(o.price),
                "amount": _num(o.amount),
                "placed_at": timeutil.to_iso(o.created_at),
            }
            for o in orders
        ]
        for side, orders in (("pending_buy", state.pending_buy), ("pending_sell", state.pending_sell))
    }
    document["counters"].update(
        {
            "fills_today": state.activity.fills_today,
            "realized_pnl_jpy": _num(state.realized_pnl_jpy),
            "cooldown_until": _iso(state.activity.cooldown_until),
        }
    )
    return document


def write_status(document: dict, path: Path | str) -> Path:
    """スナップショットを書き出す。実行が成功したときだけ呼ぶ。

    書き出し先は agent.yaml の `agent.status_output`（Git 管理外）。
    リポジトリ直下の status.yaml はスキーマの見本であり、実行では触らない。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# ドパバエの現在状態\n"
        "#\n"
        "# 本ファイルは状態の正ではありません。判断に使う状態は毎回 bitbank paper の\n"
        "# 実測から導出します。本ファイルは人間が読むためのスナップショットです。\n"
        "#\n"
        "# 15分ごとの実行で毎回上書きされます。\n"
        "# null は「まだ観測していない」を意味し、推測値を入れてはいけません。\n"
    )
    body = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
    target.write_text(f"{header}\n{body}", encoding="utf-8")
    return target
