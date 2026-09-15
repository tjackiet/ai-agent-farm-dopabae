"""市場と取引所の観測。

観測できなかった値は None のままにする。推測で埋めない（CLAUDE.md）。
檻が使うのは気配（買い気配・売り気配）と鮮度だけ。ハエに見せる足（Phase 2）は
ここに足す。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from . import timeutil
from .cli import Client
from .config import Config
from .orders import PairSpec, to_decimal

# 取引所の稼働状態のうち、判断してよいもの。ここに無い値は理由を問わず HOLD。
TRADABLE_EXCHANGE_STATUS = frozenset({"NORMAL", "BUSY", "VERY_BUSY"})
# サーキットブレイクが発動していないことを表す mode。ここに無い値は HOLD。
NORMAL_CIRCUIT_MODES = frozenset({"NONE", "NORMAL"})


@dataclass(frozen=True)
class Guards:
    exchange_status: str | None
    circuit_mode: str | None

    @property
    def exchange_ok(self) -> bool:
        return self.exchange_status in TRADABLE_EXCHANGE_STATUS

    @property
    def circuit_ok(self) -> bool:
        return self.circuit_mode in NORMAL_CIRCUIT_MODES


@dataclass(frozen=True)
class Market:
    pair: str
    last: Decimal
    bid: Decimal  # 買い気配（ticker の buy）。買い指値はここに置く
    ask: Decimal  # 売り気配（ticker の sell）。売り指値はここに置く
    observed_at: datetime
    age_sec: float
    # 価格そのものの出典。判断に使った数値には、その数値を返したコマンドを添える。
    source_cmd: str

    def price_from(self, source: str) -> Decimal:
        if source == "best_bid":
            return self.bid
        if source == "best_ask":
            return self.ask
        raise ValueError(f"扱えない価格の出どころです: {source}")


def check_guards(client: Client, config: Config) -> Guards:
    """取引所の稼働状態とサーキットブレイクを見る。"""
    statuses = client.status().data
    status = None
    if isinstance(statuses, list):
        for row in statuses:
            if isinstance(row, dict) and row.get("pair") == config.pair:
                status = str(row.get("status"))
                break

    circuit = client.circuit_break(config.pair).data
    mode = str(circuit.get("mode")) if isinstance(circuit, dict) else None
    return Guards(exchange_status=status, circuit_mode=mode)


def observe_market(client: Client, config: Config, now: datetime) -> Market:
    response = client.ticker(config.pair)
    ticker = response.data
    if not isinstance(ticker, dict):
        raise ValueError("ticker の応答が辞書ではありません")
    for key in ("last", "buy", "sell", "timestamp"):
        if ticker.get(key) is None:
            raise ValueError(f"ticker の {key} が取得できませんでした")
    observed_at = timeutil.from_epoch_ms(float(ticker["timestamp"]), config.timezone)
    bid = to_decimal(ticker["buy"])
    ask = to_decimal(ticker["sell"])
    if bid <= 0 or ask <= 0 or bid > ask:
        raise ValueError(f"気配が不正です: buy={bid} sell={ask}")
    return Market(
        pair=config.pair,
        last=to_decimal(ticker["last"]),
        bid=bid,
        ask=ask,
        observed_at=observed_at,
        age_sec=(now - observed_at).total_seconds(),
        source_cmd=response.source.cmd,
    )


def observe_pair_spec(client: Client, config: Config) -> PairSpec:
    rows = client.pairs().data
    if not isinstance(rows, list):
        raise ValueError("pairs の応答が配列ではありません")
    for row in rows:
        if isinstance(row, dict) and row.get("name") == config.pair:
            return PairSpec(
                unit_amount=to_decimal(row["unit_amount"]),
                limit_max_amount=to_decimal(row["limit_max_amount"]),
                market_max_amount=to_decimal(row["market_max_amount"]),
                price_digits=int(row["price_digits"]),
                maker_fee_rate_quote=to_decimal(row["maker_fee_rate_quote"]),
                taker_fee_rate_quote=to_decimal(row["taker_fee_rate_quote"]),
                is_enabled=bool(row["is_enabled"]),
            )
    raise ValueError(f"pairs に {config.pair} がありません")
