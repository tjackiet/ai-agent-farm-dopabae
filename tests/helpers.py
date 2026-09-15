"""テスト用の CLI 応答フィクスチャと、差し替え用のランナー。外には出ない。"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from decimal import Decimal
from typing import Any, Sequence

from dopabae import config as config_module
from dopabae import timeutil
from dopabae.direction import APPROACH, Direction
from dopabae.observe import Guards, Market
from dopabae.orders import PairSpec
from dopabae.state import Account, Activity, OpenOrder, Position, State

TZ = "Asia/Tokyo"
# テストの基準時刻（2026-09-15T09:00:00+09:00）をエポックミリ秒で持つ。
NOW_ISO = "2026-09-15T09:00:00+09:00"
NOW_MS = 1789430400000


def load_config() -> config_module.Config:
    """正本の agent.yaml をそのまま使う（テストが設定の検証も兼ねる）。"""
    return config_module.load()


def at(text: str) -> datetime:
    return timeutil.from_iso(text, TZ)


PAIR_ROW = {
    "name": "btc_jpy",
    "base_asset": "btc",
    "quote_asset": "jpy",
    "maker_fee_rate_base": 0,
    "taker_fee_rate_base": 0.0012,
    "maker_fee_rate_quote": 0,
    "taker_fee_rate_quote": 0.0012,
    "unit_amount": 0.0001,
    "limit_max_amount": 1000,
    "market_max_amount": 500,
    "price_digits": 0,
    "amount_digits": 4,
    "is_enabled": True,
    "stop_order": False,
    "stop_order_and_cancel": False,
}


def pair_spec(**overrides: Any) -> PairSpec:
    values: dict[str, Any] = {
        "unit_amount": Decimal("0.0001"),
        "limit_max_amount": Decimal(1000),
        "market_max_amount": Decimal(500),
        "price_digits": 0,
        "maker_fee_rate_quote": Decimal(0),
        "taker_fee_rate_quote": Decimal("0.0012"),
        "is_enabled": True,
    }
    values.update(overrides)
    return PairSpec(**values)


def market(last: str = "14700000", bid: str = "14699000", ask: str = "14701000", age_sec: float = 5.0) -> Market:
    return Market(
        pair="btc_jpy",
        last=Decimal(last),
        bid=Decimal(bid),
        ask=Decimal(ask),
        observed_at=at(NOW_ISO),
        age_sec=age_sec,
        source_cmd="bitbank ticker btc_jpy --format=json --machine",
    )


def guards(exchange: str = "NORMAL", circuit: str = "NONE") -> Guards:
    return Guards(exchange_status=exchange, circuit_mode=circuit)


def direction(value: str | None = APPROACH, source: str = "test", error: str | None = None) -> Direction:
    return Direction(source=source, value=value, reason="テスト", error=error)


def state(
    *,
    position: str = "0",
    avg_cost: str | None = None,
    cash: str = "1000000",
    cash_available: str | None = None,
    equity: str | None = None,
    opened_at: str | None = None,
    age_days: float | None = None,
    last_fill_at: str | None = None,
    cooldown_until: str | None = None,
    fills_today: int = 0,
    buys_in_round: int = 0,
    pending_buy: Sequence[OpenOrder] = (),
    pending_sell: Sequence[OpenOrder] = (),
    mismatch: bool = False,
    base_available: str | None = None,
) -> State:
    amount = Decimal(position)
    available = Decimal(cash_available if cash_available is not None else cash)
    base_free = Decimal(base_available) if base_available is not None else amount
    return State(
        position=Position(
            amount=amount,
            avg_cost_jpy=Decimal(avg_cost) if avg_cost else None,
            opened_at=at(opened_at) if opened_at else None,
            age_days=age_days,
        ),
        activity=Activity(
            last_fill_price_jpy=Decimal(avg_cost) if avg_cost else None,
            last_fill_at=at(last_fill_at) if last_fill_at else None,
            cooldown_until=at(cooldown_until) if cooldown_until else None,
            fills_today=fills_today,
            buys_in_round=buys_in_round,
        ),
        account=Account(
            initial_jpy=Decimal(1000000),
            cash_total_jpy=Decimal(cash),
            cash_locked_jpy=Decimal(cash) - available,
            cash_available_jpy=available,
            base_total=amount,
            base_available=base_free,
            equity_jpy=Decimal(equity) if equity else Decimal(cash) + amount * Decimal("14700000"),
        ),
        pending_buy=tuple(pending_buy),
        pending_sell=tuple(pending_sell),
        realized_pnl_jpy=Decimal(0),
        unrealized_pnl_jpy=Decimal(0),
        position_mismatch=mismatch,
    )


def open_order(order_id: str, side: str, price: str, amount: str, created_at: str = "2026-09-15T08:45:00+09:00") -> OpenOrder:
    return OpenOrder(id=order_id, side=side, price=Decimal(price), amount=Decimal(amount), created_at=at(created_at))


class FakeCli:
    """argv からキーを作り、あらかじめ用意した応答を返す。"""

    def __init__(self, responses: dict[str, Any], errors: dict[str, str] | None = None) -> None:
        self.responses = responses
        self.errors = errors or {}
        self.calls: list[str] = []

    @staticmethod
    def key_of(argv: Sequence[str]) -> str:
        parts = [a for a in argv[1:] if not a.startswith("--")]
        return " ".join(parts[:2]) if parts and parts[0] == "paper" else (parts[0] if parts else "")

    def __call__(self, argv: Sequence[str], env: dict[str, str], timeout: int) -> "subprocess.CompletedProcess[str]":
        self.calls.append(" ".join(argv))
        key = self.key_of(argv)
        if key in self.errors:
            body = {"success": False, "error": self.errors[key], "exitCode": 1}
            return subprocess.CompletedProcess(list(argv), 1, json.dumps(body), "")
        if key not in self.responses:
            raise AssertionError(f"応答が用意されていません: {key} ({' '.join(argv)})")
        body = {"success": True, "data": self.responses[key]}
        return subprocess.CompletedProcess(list(argv), 0, json.dumps(body), "")


def candles(count: int = 12, last: int = 14_700_000, interval_min: int = 15, end_ms: int | None = None) -> list[dict]:
    """最後の1本が進行中の足になるように、現在時刻から遡って並べる。"""
    end = end_ms if end_ms is not None else NOW_MS
    step = interval_min * 60_000
    rows = []
    for index in range(count):
        close = last - (count - 1 - index) * 20_000
        rows.append({
            "open": close - 5_000, "high": close + 15_000, "low": close - 15_000,
            "close": close, "vol": 10, "timestamp": end - (count - 1 - index) * step,
        })
    return rows


def default_responses(
    *,
    last: int = 14_700_000,
    assets: Sequence[dict] | None = None,
    pnl: dict | None = None,
    active_orders: Sequence[dict] = (),
    history: Sequence[dict] = (),
    ticker_ms: int = NOW_MS - 5_000,
) -> dict[str, Any]:
    return {
        "status": [{"pair": "btc_jpy", "status": "NORMAL", "min_amount": "0.0001"}],
        "circuit-break": {"mode": "NONE", "fee_type": "NORMAL", "timestamp": ticker_ms},
        "ticker": {
            "sell": last + 1000,
            "buy": last - 1000,
            "high": last + 50000,
            "low": last - 50000,
            "open": last,
            "last": last,
            "vol": 100,
            "timestamp": ticker_ms,
        },
        "candles": candles(last=last),
        "pairs": [PAIR_ROW],
        "paper tick": {"filled": [], "warnings": [], "lastTickAt": "2026-09-15T00:00:00.000Z"},
        "paper assets": list(assets)
        if assets is not None
        else [{"asset": "jpy", "total": 1000000, "locked": 0, "available": 1000000}],
        "paper pnl": pnl if pnl is not None else {"perPair": {}, "total": {"realizedPnl": 0, "unrealizedPnl": 0, "totalPnl": 0}},
        "paper active-orders": list(active_orders),
        "paper trade-history": list(history),
        "paper create-order": {"placed": {"id": "new-order"}},
        "paper cancel-order": {"canceled": {"id": "old-order"}},
    }
