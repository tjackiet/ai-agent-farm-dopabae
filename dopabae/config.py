"""agent.yaml の読み込みと検証。

数値パラメータの入口はこのモジュールだけとする。
値の唯一の正は agent.yaml であり、既定値をここに書かない
（欠けていれば ConfigError で落とす。黙って補わない）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    """agent.yaml が読めない、または必要な値が欠けている。"""


def _get(raw: Any, path: str) -> Any:
    node: Any = raw
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"agent.yaml に {path} がありません")
        node = node[key]
    return node


def _num(raw: Any, path: str) -> float:
    value = _get(raw, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"agent.yaml の {path} が数値ではありません: {value!r}")
    return float(value)


def _int(raw: Any, path: str) -> int:
    value = _num(raw, path)
    if value != int(value):
        raise ConfigError(f"agent.yaml の {path} が整数ではありません: {value!r}")
    return int(value)


def _str(raw: Any, path: str) -> str:
    value = _get(raw, path)
    if not isinstance(value, str):
        raise ConfigError(f"agent.yaml の {path} が文字列ではありません: {value!r}")
    return value


def _bool(raw: Any, path: str) -> bool:
    value = _get(raw, path)
    if not isinstance(value, bool):
        raise ConfigError(f"agent.yaml の {path} が真偽値ではありません: {value!r}")
    return value


def _choice(raw: Any, path: str, choices: tuple[str, ...]) -> str:
    value = _str(raw, path)
    if value not in choices:
        raise ConfigError(
            f"agent.yaml の {path} は {' / '.join(choices)} のいずれかです: {value}"
        )
    return value


# 判断ログに残す、檻と方向の値の指紋。**設定を変えた前後のラウンドを混ぜないために使う。**
# ペーパーで値を試すと、同じ口座に違う設定の結果が並ぶ。どの回がどの設定だったかが
# 残っていないと、あとの評価（docs/IMPLEMENTATION_PLAN.md Phase 4）が設定違いを
# 平均した数字を出す。
FINGERPRINTED = ("strategy", "risk", "direction", "fly")


def fingerprint(raw: Any) -> str:
    """`strategy` / `risk` / `direction` / `fly` の値から8桁の指紋を作る。"""
    subject = {key: raw.get(key) for key in FINGERPRINTED if isinstance(raw, dict)}
    packed = json.dumps(subject, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()[:8]


# 方向の出どころ。fly は Phase 3 で追加する。
DIRECTION_SOURCES = ("always_approach", "random", "fly")
# 方向を得られなかったときの扱い。hold しか認めない（CLAUDE.md「判断できないときは HOLD」）。
ON_FAILURE_CHOICES = ("hold",)
# 指値の価格をどこから取るか。
PRICE_SOURCES = ("best_bid", "best_ask")


# 使わないモデル。料金が上位帯のため、設定に書かれていても受け付けない。
# 事故で高いモデルが選ばれることを防ぐ。
FORBIDDEN_MODEL_MARKERS = ("fable", "mythos")


def _model(raw: Any, path: str) -> str:
    value = _str(raw, path)
    lowered = value.lower()
    for marker in FORBIDDEN_MODEL_MARKERS:
        if marker in lowered:
            raise ConfigError(f"agent.yaml の {path} に {marker} 系のモデルは指定できません: {value}")
    return value


@dataclass(frozen=True)
class LlmSettings:
    """LLM の呼びかた。記録の言語化が使う。売買の判断には関与しない。"""

    writer: str
    command: str
    bare: bool
    timeout_sec: int
    model: str
    effort: str
    max_tokens: int


@dataclass(frozen=True)
class Config:
    """agent.yaml の内容。すべて読み取り専用。"""

    version: str
    agent_id: str
    phase: str
    pair: str
    dry_run: bool
    timezone: str
    max_runtime_sec: int
    stale_tick_hours: float
    lock_path: str

    cli_command: str
    global_flags: tuple[str, ...]
    state_path: str
    forbidden: tuple[str, ...]

    initial_jpy: float

    # 檻（strategy）
    spot_only: bool
    entry_order_type: str
    entry_price_source: str
    budget_jpy_per_order: float
    max_pending_buy_orders: int
    cooldown_hours_after_fill: float
    max_fills_per_day: int
    exit_order_type: str
    exit_price_source: str
    sell_ratio: float
    time_stop_days: float

    # 檻（risk）
    max_position_ratio: float
    min_cash_reserve_ratio: float
    per_order_max_jpy: float
    halt_new_buys_pct: float
    forced_exit_pct: float
    skip_on_circuit_break: bool
    skip_on_exchange_maintenance: bool
    max_data_age_sec: int

    # 方向の出どころ
    direction_source: str
    direction_on_failure: str
    direction_random_seed: int | None
    direction_random_weights: dict[str, float] | None

    status_output: str
    performance_output: str

    decisions_path: str
    decisions_read_last_n: int
    daily_path: str
    daily_write_at: str
    daily_read_last_n: int

    # 記録の言語化。売買の判断には関与しない。
    narrate_enabled: bool
    narrate_writer: str
    narrate_command: str
    narrate_bare: bool
    narrate_timeout_sec: int
    narrate_model: str
    narrate_effort: str
    narrate_max_tokens: int
    narrate_targets: dict

    # ハエに見せる画像（fly.vision）
    vision_width: int
    vision_height: int
    vision_candle_type: str
    vision_lookback_candles: int
    vision_show: tuple[str, ...]
    vision_hide: tuple[str, ...]
    vision_palette: dict[str, tuple[int, int, int]]
    vision_margin_px: int
    vision_text_rows_px: int
    vision_path: str

    retina_column_map_path: str
    retina_luminance_types: tuple[str, ...]
    retina_blue_green_types: tuple[str, ...]
    retina_r8_channel: str
    retina_field_px: int | None
    retina_flip_x: bool
    retina_flip_y: bool

    raw: dict


# ハエに見せてはならないもの。show に入っていたら設定として拒否する（CLAUDE.md）。
NEVER_SHOWN = ("balance", "pnl", "position")
PALETTE_KEYS = ("background", "text", "up", "down", "wick", "bid", "ask")
# R8 に渡す青／緑の代理値の採りかた。どれも設計者の読み替えである。
R8_CHANNELS = ("blue_green_mean", "blue", "green")


def _rgb(raw: Any, path: str) -> tuple[int, int, int]:
    value = _get(raw, path)
    if not isinstance(value, list) or len(value) != 3 or any(
        isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255 for v in value
    ):
        raise ConfigError(f"agent.yaml の {path} は 0〜255 の整数3つです: {value!r}")
    return (int(value[0]), int(value[1]), int(value[2]))


def _str_list(raw: Any, path: str) -> tuple[str, ...]:
    value = _get(raw, path)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"agent.yaml の {path} は文字列の配列です: {value!r}")
    return tuple(value)


def load(path: Path | str | None = None) -> Config:
    """agent.yaml を読み込む。欠けている値があれば ConfigError。"""
    target = Path(path) if path is not None else REPO_ROOT / "agent.yaml"
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"agent.yaml が見つかりません: {target}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"agent.yaml を解釈できません: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("agent.yaml の内容が辞書ではありません")

    pairs = _get(raw, "market.pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ConfigError("agent.yaml の market.pairs が空です")

    forbidden = _get(raw, "cli.forbidden")
    if not isinstance(forbidden, list) or not forbidden:
        raise ConfigError("agent.yaml の cli.forbidden が空です")

    flags = _get(raw, "cli.global_flags")
    if not isinstance(flags, list):
        raise ConfigError("agent.yaml の cli.global_flags が配列ではありません")

    if _str(raw, "cli.mode") != "paper":
        raise ConfigError("agent.yaml の cli.mode は paper でなければなりません")

    spot_only = _bool(raw, "strategy.spot_only")
    if not spot_only:
        raise ConfigError("agent.yaml の strategy.spot_only は true でなければなりません（現物のみ）")

    weights_raw = _get(raw, "direction.random_weights")
    weights: dict[str, float] | None = None
    if weights_raw is not None:
        if not isinstance(weights_raw, dict) or not weights_raw:
            raise ConfigError(f"agent.yaml の direction.random_weights は辞書か null です: {weights_raw!r}")
        weights = {}
        for key, value in weights_raw.items():
            if key not in ("APPROACH", "AVOID", "NONE"):
                raise ConfigError(f"agent.yaml の direction.random_weights に不明な方向があります: {key}")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ConfigError(f"agent.yaml の direction.random_weights.{key} は 0 以上の数値です: {value!r}")
            weights[str(key)] = float(value)
        if sum(weights.values()) <= 0:
            raise ConfigError("agent.yaml の direction.random_weights の合計が 0 です")

    seed_raw = _get(raw, "direction.random_seed")
    if seed_raw is not None and (isinstance(seed_raw, bool) or not isinstance(seed_raw, int)):
        raise ConfigError(f"agent.yaml の direction.random_seed は整数か null です: {seed_raw!r}")

    show = _str_list(raw, "fly.vision.show")
    hide = _str_list(raw, "fly.vision.hide")
    for item in NEVER_SHOWN:
        if item in show or item not in hide:
            raise ConfigError(f"agent.yaml の fly.vision で {item} をハエに見せてはなりません")
    if _str(raw, "fly.vision.background") != "light":
        raise ConfigError("agent.yaml の fly.vision.background は light です（暗い背景では KC が発火しない）")

    luminance_types = _str_list(raw, "fly.retina.luminance_types")
    blue_green_types = _str_list(raw, "fly.retina.blue_green_types")
    if not luminance_types or not blue_green_types:
        raise ConfigError("agent.yaml の fly.retina の型名が空です")
    overlap = set(luminance_types) & set(blue_green_types)
    if overlap:
        # 同じ型が両方に入っていると、どちらの値を渡したか記録から読めなくなる。
        raise ConfigError(
            f"agent.yaml の fly.retina で型が重複しています: {sorted(overlap)}"
        )

    field_px = _get(raw, "fly.retina.field_px")
    if field_px is not None:
        field_px = _int(raw, "fly.retina.field_px")
        if field_px <= 0:
            raise ConfigError("agent.yaml の fly.retina.field_px は 1 以上か null です")

    halt = _num(raw, "risk.drawdown.halt_new_buys_pct")
    forced = _num(raw, "risk.drawdown.forced_exit_pct")
    if forced >= halt:
        raise ConfigError("risk.drawdown.forced_exit_pct は halt_new_buys_pct より小さくなければなりません")

    return Config(
        version=_str(raw, "version"),
        agent_id=_str(raw, "agent.id"),
        phase=_str(raw, "agent.phase"),
        pair=str(pairs[0]),
        dry_run=_bool(raw, "runtime.dry_run"),
        timezone=_str(raw, "runtime.timezone"),
        max_runtime_sec=_int(raw, "runtime.max_runtime_sec"),
        stale_tick_hours=_num(raw, "runtime.stale_tick_hours"),
        lock_path=_str(raw, "runtime.lock_path"),
        cli_command=_str(raw, "cli.command"),
        global_flags=tuple(str(f) for f in flags),
        state_path=_str(raw, "cli.state_path"),
        forbidden=tuple(str(f) for f in forbidden),
        initial_jpy=_num(raw, "capital.initial_jpy"),
        spot_only=spot_only,
        entry_order_type=_choice(raw, "strategy.entry.order_type", ("limit",)),
        entry_price_source=_choice(raw, "strategy.entry.price_source", PRICE_SOURCES),
        budget_jpy_per_order=_num(raw, "strategy.entry.budget_jpy_per_order"),
        max_pending_buy_orders=_int(raw, "strategy.entry.max_pending_buy_orders"),
        cooldown_hours_after_fill=_num(raw, "strategy.entry.cooldown_hours_after_fill"),
        max_fills_per_day=_int(raw, "strategy.entry.max_fills_per_day"),
        exit_order_type=_choice(raw, "strategy.exit.order_type", ("limit",)),
        exit_price_source=_choice(raw, "strategy.exit.price_source", PRICE_SOURCES),
        sell_ratio=_num(raw, "strategy.exit.sell_ratio"),
        time_stop_days=_num(raw, "strategy.exit.time_stop_days"),
        max_position_ratio=_num(raw, "risk.max_position_ratio"),
        min_cash_reserve_ratio=_num(raw, "risk.min_cash_reserve_ratio"),
        per_order_max_jpy=_num(raw, "risk.per_order_max_jpy"),
        halt_new_buys_pct=halt,
        forced_exit_pct=forced,
        skip_on_circuit_break=_bool(raw, "risk.guards.skip_on_circuit_break"),
        skip_on_exchange_maintenance=_bool(raw, "risk.guards.skip_on_exchange_maintenance"),
        max_data_age_sec=_int(raw, "risk.guards.max_data_age_sec"),
        direction_source=_choice(raw, "direction.source", DIRECTION_SOURCES),
        direction_on_failure=_choice(raw, "direction.on_failure", ON_FAILURE_CHOICES),
        direction_random_seed=seed_raw,
        direction_random_weights=weights,
        status_output=_str(raw, "agent.status_output"),
        performance_output=_str(raw, "agent.performance_output"),
        decisions_path=_str(raw, "memory.decisions.path"),
        decisions_read_last_n=_int(raw, "memory.decisions.read_last_n"),
        daily_path=_str(raw, "memory.daily.path"),
        daily_write_at=_str(raw, "memory.daily.write_at"),
        daily_read_last_n=_int(raw, "memory.daily.read_last_n"),
        narrate_enabled=_bool(raw, "narrate.enabled"),
        narrate_writer=_choice(raw, "narrate.writer", ("claude_code", "api")),
        narrate_command=_str(raw, "narrate.command"),
        narrate_bare=_bool(raw, "narrate.bare"),
        narrate_timeout_sec=_int(raw, "narrate.timeout_sec"),
        narrate_model=_model(raw, "narrate.model"),
        narrate_effort=_str(raw, "narrate.effort"),
        narrate_max_tokens=_int(raw, "narrate.max_tokens"),
        narrate_targets={str(k): bool(v) for k, v in _get(raw, "narrate.targets").items()},
        vision_width=_int(raw, "fly.vision.width"),
        vision_height=_int(raw, "fly.vision.height"),
        vision_candle_type=_str(raw, "fly.vision.candle_type"),
        vision_lookback_candles=_int(raw, "fly.vision.lookback_candles"),
        vision_show=show,
        vision_hide=hide,
        vision_palette={key: _rgb(raw, f"fly.vision.palette.{key}") for key in PALETTE_KEYS},
        vision_margin_px=_int(raw, "fly.vision.margin_px"),
        vision_text_rows_px=_int(raw, "fly.vision.text_rows_px"),
        vision_path=_str(raw, "memory.vision.path"),
        retina_column_map_path=_str(raw, "fly.retina.column_map_path"),
        retina_luminance_types=luminance_types,
        retina_blue_green_types=blue_green_types,
        retina_r8_channel=_choice(raw, "fly.retina.r8_channel", R8_CHANNELS),
        retina_field_px=field_px,
        retina_flip_x=_bool(raw, "fly.retina.flip_x"),
        retina_flip_y=_bool(raw, "fly.retina.flip_y"),
        raw=raw,
    )
