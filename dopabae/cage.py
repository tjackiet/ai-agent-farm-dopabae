"""檻。方向を受け取り、数量・価格・上限・諦めを決める。

このモジュールは純関数だけで構成する。I/O を持たない。
ハエが決めるのは方向（APPROACH / AVOID / NONE）だけで、ここにある値は
すべて `agent.yaml` から来る。ハエがどう答えても最悪ケースはここで決まる
（CLAUDE.md「ハエ脳の関与の制約」）。判断できないときは HOLD。

板の扱い：前回置いた指値は、この回までに約定しなければ取り消す。
檻は板の状態を覚えず、毎回「いまの方向」から組み直す。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from .config import Config
from .direction import APPROACH, AVOID, NONE, Direction
from .observe import Guards, Market
from .orders import PairSpec, PlaceOrder, buy_amount, floor_price, floor_to_unit, to_decimal
from .state import State, sellable_now

HOLD = "HOLD"
BUY = "BUY"
SELL = "SELL"

# 檻が方向をどう扱ったか。判断ログに残し、personality.md の対応表で言葉にする。
GUARD = "guard"  # 観測できない・取引所異常・データが古い
EXIT = "exit"  # 檻の諦め（強制手仕舞い・時間切れ）
RECOVER = "recover"  # 長く止まっていたあとの復帰
NO_DIRECTION = "no_direction"  # 方向を得られなかった（見えなかった）
BOUGHT = "bought"  # APPROACH → 買い指値を置いた
SOLD = "sold"  # AVOID → 売り指値を置いた
WALL = "wall"  # APPROACH だが上限・冬眠・回数で止めた
COOLDOWN = "cooldown"  # APPROACH だがクールダウン中
AVOID_FLAT = "avoid_flat"  # AVOID だが建玉がない（現物のみ。何もしない）
STILL = "still"  # NONE
DUST = "dust"  # 刻みに満たず発注できない


@dataclass(frozen=True)
class Decision:
    action: str
    state: str
    reason: str
    cage: str
    cancel: tuple[str, ...] = ()
    place: tuple[PlaceOrder, ...] = ()


def _hold(state_name: str, reason: str, cage: str, cancel: tuple[str, ...] = ()) -> Decision:
    return Decision(action=HOLD, state=state_name, reason=reason, cage=cage, cancel=cancel)


def state_name(config: Config, state: State | None) -> str:
    if state is None:
        return "NOT_INITIALIZED"
    drawdown = state.account.drawdown_pct
    if drawdown <= to_decimal(config.forced_exit_pct):
        return "HALTED"
    if drawdown <= to_decimal(config.halt_new_buys_pct):
        return "HIBERNATING"
    if state.position.amount > 0:
        return "HOLDING"
    return "IDLE"


def _all_pending(state: State) -> tuple[str, ...]:
    return tuple(o.id for o in (*state.pending_buy, *state.pending_sell))


def _exit_all(config: Config, spec: PairSpec, state: State, name: str, reason: str) -> Decision:
    """全建玉を成行で手仕舞いする（強制手仕舞い・時間切れ）。"""
    cancel = _all_pending(state)
    freed = sum((o.amount for o in state.pending_sell), Decimal(0))
    amount = sellable_now(state, spec.unit_amount, freed)
    if amount > spec.market_max_amount:
        amount = floor_to_unit(spec.market_max_amount, spec.unit_amount)
    if amount < spec.unit_amount:
        return _hold(name, f"{reason}（売却できる建玉なし）", EXIT, cancel)
    return Decision(
        action=SELL,
        state=name,
        reason=reason,
        cage=EXIT,
        cancel=cancel,
        place=(PlaceOrder(side="sell", order_type="market", amount=amount, price=None, label="exit"),),
    )


def _recover(config: Config, state: State, name: str, stopped_hours: float) -> Decision:
    """長く止まっていたあとの復帰。板に残った指値をすべて取り消す。

    `paper tick` が遡れるのは直近24時間まで。それを超えて止まると、その間に
    指値へ届いた値動きは評価されない。古い指値を信用せず、いったん全部消す。
    """
    cancel = _all_pending(state)
    reason = (
        f"{stopped_hours:.1f} 時間停止していた。"
        f"遡れる上限（{config.stale_tick_hours:.0f} 時間）を超えたため、"
        "未約定の指値を取り消してこの回は何もしない"
    )
    if not cancel:
        reason = f"{reason}（取り消す注文はなし）"
    return _hold(name, reason, RECOVER, cancel)


def _buy_budget(config: Config, state: State, freed_jpy: Decimal) -> Decimal:
    """この回の買いに使える予算。リスク制約のうち最も厳しいものに合わせる。

    `freed_jpy` は、同じ回で取り消す買い指値のロックが解放されるぶん。
    """
    initial = to_decimal(config.initial_jpy)
    return min(
        to_decimal(config.budget_jpy_per_order),
        to_decimal(config.per_order_max_jpy),
        initial * to_decimal(config.max_position_ratio) - state.position.cost_basis_jpy,
        state.account.cash_available_jpy + freed_jpy - initial * to_decimal(config.min_cash_reserve_ratio),
    )


def _approach(
    config: Config, market: Market, spec: PairSpec, state: State, name: str, now: datetime,
    cancel: tuple[str, ...],
) -> Decision:
    """APPROACH。檻の中で買えるなら固定額の買い指値を買い気配に置く。"""
    if name == "HIBERNATING":
        return _hold(
            name, f"総資産が {state.account.drawdown_pct:.1f}% のため新規買いを停止している", WALL, cancel
        )
    if config.max_pending_buy_orders < 1:
        return _hold(name, "買い指値の上限が 0 本のため出さない", WALL, cancel)
    if state.activity.fills_today >= config.max_fills_per_day:
        return _hold(name, f"本日すでに {state.activity.fills_today} 回約定している", WALL, cancel)
    cooldown = state.activity.cooldown_until
    if cooldown is not None and now < cooldown:
        return _hold(
            name, f"クールダウン中。次の買いは {cooldown.isoformat(timespec='minutes')} 以降", COOLDOWN, cancel
        )

    price = floor_price(market.price_from(config.entry_price_source), spec.price_digits)
    if price <= 0:
        return _hold(name, "指値価格を算出できない", GUARD, cancel)
    freed = sum((o.price * o.amount for o in state.pending_buy), Decimal(0))
    budget = _buy_budget(config, state, freed)
    if budget <= 0:
        return _hold(name, "リスク制約により使える予算が残っていない", WALL, cancel)
    amount = buy_amount(budget, price, spec)
    if amount <= 0:
        return _hold(name, "刻みに満たないため発注しない", DUST, cancel)
    return Decision(
        action=BUY,
        state=name,
        reason=f"近づいた。買い気配 {price} に {budget:.0f} JPY ぶんの買い指値を置く",
        cage=BOUGHT,
        cancel=cancel,
        place=(PlaceOrder(side="buy", order_type=config.entry_order_type, amount=amount, price=price, label="approach"),),
    )


def _avoid(
    config: Config, market: Market, spec: PairSpec, state: State, name: str, cancel: tuple[str, ...]
) -> Decision:
    """AVOID。建玉があれば売り気配に売り指値を置く。無ければ何もしない（現物のみ）。"""
    freed = sum((o.amount for o in state.pending_sell), Decimal(0))
    held = sellable_now(state, spec.unit_amount, freed)
    if held <= 0:
        return _hold(name, "逃げた。建玉がないので何もしない（現物のみ）", AVOID_FLAT, cancel)
    amount = floor_to_unit(held * to_decimal(config.sell_ratio), spec.unit_amount)
    if amount > spec.limit_max_amount:
        amount = floor_to_unit(spec.limit_max_amount, spec.unit_amount)
    if amount < spec.unit_amount:
        return _hold(name, "刻みに満たないため発注しない", DUST, cancel)
    price = floor_price(market.price_from(config.exit_price_source), spec.price_digits)
    if price <= 0:
        return _hold(name, "指値価格を算出できない", GUARD, cancel)
    return Decision(
        action=SELL,
        state=name,
        reason=f"逃げた。売り気配 {price} に {amount} の売り指値を置く",
        cage=SOLD,
        cancel=cancel,
        place=(PlaceOrder(side="sell", order_type=config.exit_order_type, amount=amount, price=price, label="avoid"),),
    )


def decide(
    config: Config,
    guards: Guards,
    market: Market | None,
    spec: PairSpec | None,
    state: State | None,
    direction: Direction,
    now: datetime,
    stopped_hours: float | None = None,
) -> Decision:
    """観測・状態・方向から、この回の行動を決める。"""
    name = state_name(config, state)

    if state is None or market is None or spec is None:
        return _hold(name, "ペーパートレード口座または市場データを観測できていない", GUARD)
    if config.skip_on_exchange_maintenance and not guards.exchange_ok:
        return _hold(name, f"取引所の状態が {guards.exchange_status} のため判断しない", GUARD)
    if config.skip_on_circuit_break and not guards.circuit_ok:
        return _hold(name, f"サーキットブレイク（mode={guards.circuit_mode}）のため判断しない", GUARD)
    if market.age_sec > config.max_data_age_sec:
        return _hold(name, f"データが {int(market.age_sec)} 秒前と古い", GUARD)
    if not spec.is_enabled:
        return _hold(name, f"{config.pair} が取引可能な状態ではない", GUARD)
    if state.position_mismatch:
        return _hold(name, "建玉と残高が一致しない。人間の確認が必要", GUARD)

    # 檻の諦めは方向より先に効く。ハエがどう言おうと変わらない。
    if name == "HALTED":
        return _exit_all(
            config, spec, state, name, f"総資産が {state.account.drawdown_pct:.1f}% で強制手仕舞いの水準"
        )
    if stopped_hours is not None and stopped_hours > config.stale_tick_hours:
        return _recover(config, state, name, stopped_hours)
    age_days = state.position.age_days
    if age_days is not None and age_days > config.time_stop_days:
        return _exit_all(config, spec, state, name, f"建玉の保有が {age_days:.1f} 日で上限を超えた")

    # 前回の指値は、この回までに約定しなければ取り消す。以降はまっさらな板として組む。
    stale = _all_pending(state)

    if not direction.ok:
        return _hold(name, f"方向を得られなかったため何もしない: {direction.error}", NO_DIRECTION, stale)
    if direction.value == AVOID:
        return _avoid(config, market, spec, state, name, stale)
    if direction.value == NONE:
        return _hold(name, "じっとしてた", STILL, stale)
    if direction.value == APPROACH:
        return _approach(config, market, spec, state, name, now, stale)
    return _hold(name, f"扱えない方向です: {direction.value}", NO_DIRECTION, stale)
