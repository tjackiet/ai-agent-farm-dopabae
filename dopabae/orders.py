"""数量と価格の計算、丸め、発注・取消の実行。

CLI は数量も価格も丸めない（unit_amount の倍数でなければエラー）。
丸めはこちらの責任である（strategy.md「発注数量の求め方」）。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from typing import Sequence

from .cli import Client, CliError, Response, is_already_filled


@dataclass(frozen=True)
class PairSpec:
    """bitbank pairs から取る、発注に必要な仕様。"""

    unit_amount: Decimal
    limit_max_amount: Decimal
    market_max_amount: Decimal
    price_digits: int
    maker_fee_rate_quote: Decimal
    taker_fee_rate_quote: Decimal
    is_enabled: bool


@dataclass(frozen=True)
class PlaceOrder:
    """これから出す注文。price が None なら成行。"""

    side: str
    order_type: str
    amount: Decimal
    price: Decimal | None
    label: str


def to_decimal(value: object) -> Decimal:
    """CLI 由来の数値を Decimal にする。float の誤差を持ち込まない。"""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise TypeError(f"数値ではありません: {value!r}")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(repr(value))
    if isinstance(value, str):
        return Decimal(value)
    raise TypeError(f"数値ではありません: {value!r}")


def floor_to_unit(value: Decimal, unit: Decimal) -> Decimal:
    """unit_amount の倍数へ切り捨てる。"""
    if unit <= 0:
        raise ValueError("unit_amount は正の数でなければなりません")
    return (value / unit).to_integral_value(rounding=ROUND_FLOOR) * unit


def floor_price(value: Decimal, digits: int) -> Decimal:
    """price_digits に収まるよう切り捨てる。買いは下へ丸めるほうが保守的。"""
    if digits < 0:
        raise ValueError("price_digits は 0 以上でなければなりません")
    return value.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_FLOOR)


def format_number(value: Decimal) -> str:
    """CLI へ渡す文字列。指数表記にしない。"""
    return format(value.normalize(), "f")


def buy_amount(budget_jpy: Decimal, price: Decimal, spec: PairSpec) -> Decimal:
    """予算から買い指値の数量を求める。

    買い指値は `価格 × 数量 × (1 + maker手数料率)` を JPY 側でロックするため、
    予算ちょうどの数量では残高不足になる。maker がリベート（負）のときは
    0 として扱い、CLI 側のフォールバック（0.0012）でロックされても足りるようにする。
    """
    if budget_jpy <= 0 or price <= 0:
        return Decimal(0)
    maker = spec.maker_fee_rate_quote if spec.maker_fee_rate_quote > 0 else Decimal(0)
    raw = budget_jpy / (price * (Decimal(1) + maker))
    amount = floor_to_unit(raw, spec.unit_amount)
    if amount > spec.limit_max_amount:
        amount = floor_to_unit(spec.limit_max_amount, spec.unit_amount)
    if amount < spec.unit_amount:
        return Decimal(0)
    return amount


def sell_amount(position: Decimal, ratio: Decimal, spec: PairSpec) -> Decimal:
    """建玉の比率から売り数量を求める。刻みに満たなければ 0。"""
    amount = floor_to_unit(position * ratio, spec.unit_amount)
    if amount < spec.unit_amount:
        return Decimal(0)
    return amount


class AlreadyFilled(Exception):
    """取消しようとした注文が、その直前に約定していた。"""


@dataclass
class Executor:
    """判断の結果を CLI へ流す。dry_run のときは組み立てるだけ。"""

    client: Client
    pair: str
    dry_run: bool

    def cancel(self, order_id: str) -> dict[str, object]:
        record: dict[str, object] = {
            "op": "cancel",
            "order_id": order_id,
            "cmd": f"{self.client.config.cli_command} paper cancel-order --id={order_id}",
            "executed": not self.dry_run,
        }
        if self.dry_run:
            return record
        try:
            self.client.paper_cancel_order(order_id)
        except CliError as exc:
            if is_already_filled(exc):
                # 取消より約定が優先される。失敗ではなく約定として扱う。
                raise AlreadyFilled(str(exc)) from exc
            raise
        return record

    def place(self, order: PlaceOrder) -> dict[str, object]:
        price = None if order.price is None else format_number(order.price)
        amount = format_number(order.amount)
        parts = [
            f"{self.client.config.cli_command} paper create-order",
            f"--pair={self.pair}",
            f"--side={order.side}",
            f"--type={order.order_type}",
            f"--amount={amount}",
        ]
        if price is not None:
            parts.append(f"--price={price}")
        record: dict[str, object] = {
            "op": "place",
            "label": order.label,
            "side": order.side,
            "type": order.order_type,
            "price": None if price is None else float(order.price),
            "amount": float(order.amount),
            "cmd": " ".join(parts),
            "executed": not self.dry_run,
        }
        if self.dry_run:
            return record
        response: Response = self.client.paper_create_order(
            pair=self.pair,
            side=order.side,
            order_type=order.order_type,
            amount=amount,
            price=price,
        )
        record["result"] = response.data
        return record


def execute(
    executor: Executor,
    cancel_ids: Sequence[str],
    place_orders: Sequence[PlaceOrder],
) -> tuple[list[dict[str, object]], str | None]:
    """取消 → 発注の順に実行する。

    取消が「すでに約定していた」で返った時点で打ち切る。その回は新しい注文を
    出さない（risk-policy.md「発注と取消の競合」）。
    """
    records: list[dict[str, object]] = []
    for order_id in cancel_ids:
        try:
            records.append(executor.cancel(order_id))
        except AlreadyFilled as exc:
            return records, f"取消の直前に約定していたため、この回は発注しない: {exc}"
    for order in place_orders:
        records.append(executor.place(order))
    return records, None
