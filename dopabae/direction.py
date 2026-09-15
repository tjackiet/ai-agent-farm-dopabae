"""方向の出どころ。

方向は APPROACH / AVOID / NONE の三値。出どころは `agent.yaml` の `direction.source`
で切り替える。どの出どころも同じ契約を満たす。

- 入力：観測（この段階では使わない。ハエ版は画像を受け取る）
- 出力：方向と根拠。失敗したら方向 None と失敗理由

方向を得られなかったときの扱いは `direction.on_failure`（hold のみ）。
檻はこの結果をそのまま受け取り、方向がなければ何もしない。

always_approach と random は評価用の対照群（docs/DESIGN_MEMO.md 4 章）。
ハエ版（fly）は Phase 3 で足す。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import Config

APPROACH = "APPROACH"
AVOID = "AVOID"
NONE = "NONE"
DIRECTIONS = (APPROACH, AVOID, NONE)


@dataclass(frozen=True)
class Direction:
    """方向の判定結果。判断ログにそのまま残す。"""

    source: str
    value: str | None  # APPROACH / AVOID / NONE。得られなければ None
    reason: str = ""
    error: str | None = None
    seed: int | None = None  # source=random のとき、この回に使った種

    @property
    def ok(self) -> bool:
        return self.value in DIRECTIONS and self.error is None

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "value": self.value,
            "reason": self.reason,
            "error": self.error,
            "seed": self.seed,
        }


def failed(source: str, error: str) -> Direction:
    return Direction(source=source, value=None, error=error[:300])


def always_approach() -> Direction:
    """檻だけを回す対照群。毎回近づく。"""
    return Direction(source="always_approach", value=APPROACH, reason="対照群：常に近づく")


def random_direction(
    seed: int | None, run_key: str, weights: dict[str, float] | None = None
) -> Direction:
    """ランダムに方向を出す対照群。

    種は `agent.yaml` の `direction.random_seed`。null なら実行ごとに作り、
    判断ログに残す（あとから同じ列を再現できるようにするため）。

    `weights` は三値の出しかた。null なら等確率。比べる相手（ハエ）が同じ期間に
    出した頻度を写して使う。頻度が違う対照群と比べると、方向の**中身**の差ではなく
    頻度の差を見てしまう（docs/DESIGN_MEMO.md 4 章の評価 4）。
    """
    actual = seed if seed is not None else random.SystemRandom().randrange(2**31)
    rng = random.Random(f"{actual}:{run_key}")
    if weights:
        population = [d for d in DIRECTIONS if weights.get(d, 0) > 0]
        if not population:
            return failed("random", "random_weights がすべて 0 です")
        value = rng.choices(population, weights=[weights[d] for d in population], k=1)[0]
        detail = ", ".join(f"{d}={weights.get(d, 0):g}" for d in DIRECTIONS)
        reason = f"対照群：ランダム（seed={actual}, {detail}）"
    else:
        value = rng.choice(DIRECTIONS)
        reason = f"対照群：ランダム（seed={actual}, 等確率）"
    return Direction(source="random", value=value, reason=reason, seed=actual)


def resolve(config: Config, run_key: str) -> Direction:
    """設定に従って方向を得る。出どころの例外はすべて失敗として返す。"""
    source = config.direction_source
    try:
        if source == "always_approach":
            return always_approach()
        if source == "random":
            return random_direction(
                config.direction_random_seed, run_key, config.direction_random_weights
            )
        if source == "fly":
            return failed(source, "ハエ版はまだ実装されていない（Phase 3）")
    except Exception as exc:  # noqa: BLE001 - 出どころの失敗で落とさず、檻に HOLD させる
        return failed(source, f"{type(exc).__name__}: {exc}")
    return failed(source, f"扱えない出どころです: {source}")
