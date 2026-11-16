"""成績の内訳。腕どうしの成績の差がどこから来たかを、判断ログの観測値だけで分ける。

`evaluate.py` は方向の**あと**の値動きと成績を出す。本モジュールは、その横で次の 3 つを測る。

| 何を | なぜ |
| --- | --- |
| 建玉の重さ（総資産に占める BTC の割合）と損益の内訳 | 下げ相場では、どれだけ持っていたかで成績がほぼ決まる。方向の中身の差か、持っていた量の差かを分ける |
| 方向の**手前**の値動き | ハエはチャート画像を見て答える。答えが直前の値動きで違うなら、それは画像への反応であり、相場の洞察ではない。p 値は答えの続きやすさを考えた循環ずらし検定で出す（`shift_test`） |
| 方向の続きかた | 頻度が同じでも、同じ答えが続くかどうかで建玉の持ちかたは変わる |

ランダムの腕も同じ物差しで測る。ランダムは画像を見ないので、そこに出る差は偶然の幅の目安になる。

## 観測していない値を作らない

- 建玉の重さは、判断ログの `account.equity_jpy` と `account.cash_jpy`（どちらも `paper assets` 由来）から
  `(総資産 − 現金) / 総資産` で出す。総資産は「現金 + BTC 残高 × 最終価格」で記録されているので、
  BTC 残高は `(総資産 − 現金) / 価格` で読み戻せる。現金と建玉を推定で積み直すことはしない
- 損益の内訳は、値がそろった回から次にそろった回までの区間ごとに分ける。
  ペーパー口座はローカルの状態ファイルで、`bitbank paper` のコマンドを走らせたときにしか変わらない。
  間に観測の欠けた回があっても、その回が paper のコマンドを 1 つも走らせていなければ
  （判断ログの `sources` と `error` で確かめる）、建玉は区間の始めのまま変わっていないので分ける。
  走らせていた回・確かめられない回をまたぐ区間は、約定したかが分からないので `unattributed_jpy` に入れ、分けない
- 定期実行が止まっていた間（記録の無い空白）と、観測の欠けた回をまたぐ区間は `gaps` に一覧で出す
- 方向の手前の値動きは、ハエに見せた画像そのものではなく、判断ログの価格（ticker の最終価格）で測る

**結論は書かない。** 数字を出すだけで、読むのは人間。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

from . import config as config_module, evaluate, timeutil
from .direction import APPROACH, AVOID, DIRECTIONS, NONE

# 足の種類 → 分。画像に写る範囲（fly.vision.candle_type × lookback_candles）を求めるのに使う。
CANDLE_MINUTES = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1hour": 60,
    "4hour": 240,
    "8hour": 480,
    "12hour": 720,
    "1day": 1440,
}
# 建玉の重さがこれ未満の回を「ほぼ持っていない」として数える（売れない端数を持ち続ける腕があるため）。
NEARLY_FLAT_EXPOSURE = 0.01
# 定期実行の間隔（分）の既定。空白の判定と、本来の回数の見積もりに使う（scripts/launchd・systemd は 15 分）。
DEFAULT_INTERVAL_MINUTES = 15
# 記録の数が本来の回数のこの割合を下回ったら、notes に出す。
LOW_COVERAGE_RATIO = 0.9


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


@dataclass(frozen=True)
class Snapshot:
    """ある回に判断ログへ残した観測値。欠けた値は None のまま持つ。"""

    at: datetime
    price: float | None
    equity_jpy: float | None
    cash_jpy: float | None
    direction: str | None
    # その回に paper のコマンドを走らせたか。判断ログから確かめられなければ None。
    touched_account: bool | None = None

    @property
    def complete(self) -> bool:
        """価格・総資産・現金がそろっていて、建玉の重さを出せる回か。"""
        return (
            self.price is not None
            and self.price > 0
            and self.equity_jpy is not None
            and self.equity_jpy > 0
            and self.cash_jpy is not None
        )

    @property
    def btc_value_jpy(self) -> float:
        return self.equity_jpy - self.cash_jpy

    @property
    def exposure(self) -> float:
        """総資産に占める BTC の割合（0〜1）。"""
        return self.btc_value_jpy / self.equity_jpy

    @property
    def btc_amount(self) -> float:
        return self.btc_value_jpy / self.price


def _mentions_paper(text: str) -> bool:
    return "paper" in re.split(r"[\s（）()]+", text)


def touched_account(record: dict) -> bool | None:
    """その回に `bitbank paper` のコマンドを走らせたか。

    成功したコマンドは `sources` に、失敗したコマンドは `error` の末尾（`…（<コマンド>）`）に残る。
    どちらかに paper のコマンドがあれば True。`sources` が無い・読めない回は確かめられないので None。
    """
    sources = record.get("sources")
    if not isinstance(sources, list):
        return None
    texts = [s.get("cmd") for s in sources if isinstance(s, dict)]
    texts.append(record.get("error"))
    return any(_mentions_paper(t) for t in texts if isinstance(t, str))


def snapshots(records: Sequence[dict], tz_name: str) -> list[Snapshot]:
    """判断ログの各回を、観測時刻の順に並べる。run_id が読めない回だけは落とす。"""
    result: list[Snapshot] = []
    for record in records:
        run_id = record.get("run_id")
        if not isinstance(run_id, str):
            continue
        try:
            at = timeutil.from_iso(run_id, tz_name)
        except ValueError:
            continue
        account = record.get("account")
        account = account if isinstance(account, dict) else {}
        direction = record.get("direction")
        value = direction.get("value") if isinstance(direction, dict) else None
        result.append(
            Snapshot(
                at=at,
                price=_number(record.get("price")),
                equity_jpy=_number(account.get("equity_jpy")),
                cash_jpy=_number(account.get("cash_jpy")),
                direction=value if value in DIRECTIONS else None,
                touched_account=touched_account(record),
            )
        )
    return sorted(result, key=lambda s: s.at)


@dataclass(frozen=True)
class Interval:
    """値がそろった回から、次にそろった回まで。`skipped` は間に挟まった観測の欠けた回。"""

    start: Snapshot
    end: Snapshot
    skipped: tuple[Snapshot, ...]

    @property
    def minutes(self) -> float:
        return (self.end.at - self.start.at).total_seconds() / 60

    @property
    def attributable(self) -> bool:
        """区間のあいだ建玉が始めのままだったと言えるか。挟まった回がどれも paper に触っていなければ言える。"""
        return all(s.touched_account is False for s in self.skipped)

    @property
    def account_touched(self) -> bool | None:
        """挟まった回のどれかが paper に触ったか。触った回があれば True、確かめられない回があれば None。"""
        flags = [s.touched_account for s in self.skipped]
        if any(f is True for f in flags):
            return True
        if any(f is None for f in flags):
            return None
        return False

    @property
    def market_jpy(self) -> float:
        return self.start.btc_amount * (self.end.price - self.start.price)

    @property
    def equity_change_jpy(self) -> float:
        return self.end.equity_jpy - self.start.equity_jpy

    @property
    def price_return(self) -> float:
        return (self.end.price - self.start.price) / self.start.price


def intervals(snaps: Sequence[Snapshot]) -> list[Interval]:
    result: list[Interval] = []
    start: Snapshot | None = None
    skipped: list[Snapshot] = []
    for s in snaps:
        if not s.complete:
            if start is not None:
                skipped.append(s)
            continue
        if start is not None:
            result.append(Interval(start=start, end=s, skipped=tuple(skipped)))
        start, skipped = s, []
    return result


def _point(snapshot: Snapshot) -> dict:
    return {
        "at": timeutil.to_iso(snapshot.at),
        "price": snapshot.price,
        "equity_jpy": round(snapshot.equity_jpy, 2),
        "cash_jpy": round(snapshot.cash_jpy, 2),
        "btc_amount": round(snapshot.btc_amount, 8),
        "exposure": round(snapshot.exposure, 4),
    }


def holdings(snaps: Sequence[Snapshot]) -> dict:
    """建玉の重さと、損益の内訳。

    損益の内訳（値がそろった回から次にそろった回までの区間ごと。`Interval`）:

    - `market_jpy`: 区間の始めに持っていた BTC × 価格の変化。持っていたことによる損益
    - `trading_jpy`: 総資産の変化のうち、上の残り。約定値と最終価格の差・手数料など
    - `unattributed_jpy`: paper に触った回・確かめられない回をまたぐ区間の増減。分けない

    3 つの和は、期間の最初と最後の総資産の差に一致する。
    `bridged_intervals` は、観測の欠けた回をまたいだが paper に触っていなかったので分けた区間の数。

    建玉の重さと値動きの関係（同じ区間の騰落率の**単純和**。複利ではない）:

    - `market_sum_pct`: ずっと全額 BTC だったときの値（騰落率の和）
    - `level_pct`: 平均の重さ × `market_sum_pct`。平均してどれだけ持っていたかで決まる分
    - `timing_pct`: 重さの平均からのずれ × 騰落率の和。下げの前に軽く、上げの前に重いほど正になる
    - `level_pct + timing_pct` は、重さで重みを付けた騰落率の和に一致する
    """
    points = [s for s in snaps if s.complete]
    if not points:
        return {
            "observations": 0,
            "start": None,
            "end": None,
            "exposure": None,
            "pnl": None,
            "timing": None,
            "daily": [],
        }

    exposures = [s.exposure for s in points]
    spans = intervals(snaps)
    attributed = [iv for iv in spans if iv.attributable]
    market = sum(iv.market_jpy for iv in attributed)
    trading = sum(iv.equity_change_jpy - iv.market_jpy for iv in attributed)
    total = points[-1].equity_jpy - points[0].equity_jpy

    timing = None
    if attributed:
        weights = [iv.start.exposure for iv in attributed]
        returns = [iv.price_return for iv in attributed]
        mean_weight = statistics.fmean(weights)
        market_sum = sum(returns) * 100
        timing = {
            "intervals": len(attributed),
            "mean_exposure": round(mean_weight, 4),
            "market_sum_pct": round(market_sum, 4),
            "level_pct": round(mean_weight * market_sum, 4),
            "timing_pct": round(sum((w - mean_weight) * r for w, r in zip(weights, returns)) * 100, 4),
        }

    by_day: dict[str, list[Snapshot]] = {}
    for s in points:
        by_day.setdefault(timeutil.date_key(s.at), []).append(s)
    daily = [
        {
            "date": day,
            "observations": len(rows),
            "mean_exposure": round(statistics.fmean(s.exposure for s in rows), 4),
            "close_price": rows[-1].price,
            "close_equity_jpy": round(rows[-1].equity_jpy, 2),
        }
        for day, rows in sorted(by_day.items())
    ]

    return {
        "observations": len(points),
        # 期間の始めと終わりに持っていたもの。腕どうしで開始条件がそろっているかを見る。
        "start": _point(points[0]),
        "end": _point(points[-1]),
        "exposure": {
            "mean": round(statistics.fmean(exposures), 4),
            "median": round(statistics.median(exposures), 4),
            "max": round(max(exposures), 4),
            "nearly_flat_ratio": round(sum(e < NEARLY_FLAT_EXPOSURE for e in exposures) / len(exposures), 4),
            "nearly_flat_below": NEARLY_FLAT_EXPOSURE,
        },
        "pnl": {
            "total_jpy": round(total, 2),
            "market_jpy": round(market, 2),
            "trading_jpy": round(trading, 2),
            "unattributed_jpy": round(total - market - trading, 2),
            "attributed_intervals": len(attributed),
            "bridged_intervals": sum(1 for iv in attributed if iv.skipped),
            "unattributed_intervals": len(spans) - len(attributed),
        },
        "timing": timing,
        "daily": daily,
    }


def gaps(snaps: Sequence[Snapshot], interval_minutes: int, tolerance_minutes: int) -> dict:
    """空白の一覧。定期実行が止まっていた区間（間隔 + 許容より長い）と、観測の欠けた回をまたぐ区間。

    `attributed` が真の区間は、建玉が始めのままだったと言えるので `market_jpy` を出す。
    偽の区間は、間に約定したかが分からないので `market_jpy` は null。
    """
    limit = interval_minutes + tolerance_minutes
    rows = [iv for iv in intervals(snaps) if iv.skipped or iv.minutes > limit]
    listed = [
        {
            "from": timeutil.to_iso(iv.start.at),
            "to": timeutil.to_iso(iv.end.at),
            "minutes": round(iv.minutes),
            "skipped_rounds": len(iv.skipped),
            "account_touched": iv.account_touched,
            "attributed": iv.attributable,
            "exposure_at_start": round(iv.start.exposure, 4),
            "price_change_pct": round(iv.price_return * 100, 4),
            "equity_change_jpy": round(iv.equity_change_jpy, 2),
            "market_jpy": round(iv.market_jpy, 2) if iv.attributable else None,
        }
        for iv in rows
    ]
    return {
        "longer_than_minutes": limit,
        "count": len(rows),
        "minutes": round(sum(iv.minutes for iv in rows)),
        "attributed": sum(1 for iv in rows if iv.attributable),
        "unattributed": sum(1 for iv in rows if not iv.attributable),
        # 空白のあいだの騰落率の単純和。期間全体の値動きのうち、どれだけが空白の間に起きたかの目安。
        "price_change_sum_pct": round(sum(iv.price_return for iv in rows) * 100, 4),
        "equity_change_jpy": round(sum(iv.equity_change_jpy for iv in rows), 2),
        "market_jpy": round(sum(iv.market_jpy for iv in rows if iv.attributable), 2),
        "list": listed,
    }


def coverage(snaps: Sequence[Snapshot], interval_minutes: int) -> dict:
    """本来の回数（最初の回から最後の回までを定期実行の間隔で割った数）に対して、何回記録が残っているか。"""
    if not snaps:
        return {"expected_rounds": 0, "records": 0, "complete": 0, "ratio": None}
    span = (snaps[-1].at - snaps[0].at).total_seconds() / 60
    expected = int(span // interval_minutes) + 1
    return {
        "expected_rounds": expected,
        "records": len(snaps),
        "complete": sum(1 for s in snaps if s.complete),
        "ratio": round(len(snaps) / expected, 4),
    }


def backward_pairs(
    obs: Sequence[evaluate.Observation], lookback_minutes: int, tolerance_minutes: int
) -> list[tuple[str, float]]:
    """各回の方向と、その手前 `lookback_minutes` の騰落率（%）の組を、時刻の順に返す。

    騰落率はその回の価格 ÷ 手前の価格 − 1。手前の観測が欠けていれば飛ばす。
    前後の値で埋めない（`evaluate.forward_returns` と同じ）。
    """
    pairs: list[tuple[str, float]] = []
    lookback = timedelta(minutes=lookback_minutes)
    tolerance = timedelta(minutes=tolerance_minutes)
    for index, end in enumerate(obs):
        target = end.at - lookback
        best = None
        for earlier in (obs[j] for j in range(index - 1, -1, -1)):
            gap = target - earlier.at
            if gap > tolerance:
                break
            if abs(gap) <= tolerance and (best is None or abs(earlier.at - target) < abs(best.at - target)):
                best = earlier
        if best is None:
            continue
        pairs.append((end.direction, (end.price - best.price) / best.price * 100))
    return pairs


def backward_returns(
    obs: Sequence[evaluate.Observation], lookback_minutes: int, tolerance_minutes: int
) -> dict[str, list[float]]:
    """方向ごとの、手前 `lookback_minutes` の騰落率（%）。"""
    by_direction: dict[str, list[float]] = {d: [] for d in DIRECTIONS}
    for direction, value in backward_pairs(obs, lookback_minutes, tolerance_minutes):
        by_direction[direction].append(value)
    return by_direction


def _approach_minus_avoid(labels: Sequence[str], values: Sequence[float]) -> float | None:
    approach = [v for label, v in zip(labels, values) if label == APPROACH]
    avoid = [v for label, v in zip(labels, values) if label == AVOID]
    if not approach or not avoid:
        return None
    return statistics.fmean(approach) - statistics.fmean(avoid)


def shift_test(pairs: Sequence[tuple[str, float]]) -> tuple[float | None, float | None, int]:
    """循環ずらし検定。答えの並びを丸ごと時間方向にずらして値動きと組み直し、
    APPROACH − AVOID の平均の差が、ずらしたときより極端かを見る。両側 p 値と、使ったずらしの数を返す。

    ふつうの並べ替え検定（`evaluate.permutation_test`）は、答えが 1 回ずつ独立に出ることを前提にする。
    同じ答えが続きやすい腕では、隣どうしの区間が重なった値動きも似るので、p 値が小さく出すぎる
    （値動きと無関係な答えでも、続きやすさ 0.7 で p < 0.05 になる割合が 3 割近くになった）。
    ずらし検定は、答えの続きかたと値動きの続きかたをどちらも保ったまま組み合わせだけを崩す。
    ずらしが小さいと元の組み合わせとほぼ同じなので、前後それぞれ全体の 5% までのずらしは使わない。
    """
    labels = [label for label, _ in pairs]
    values = [value for _, value in pairs]
    observed = _approach_minus_avoid(labels, values)
    if observed is None:
        return None, None, 0
    n = len(pairs)
    margin = max(1, n // 20)
    extreme = used = 0
    for shift in range(margin, n - margin + 1):
        diff = _approach_minus_avoid(labels[shift:] + labels[:shift], values)
        if diff is None:
            continue
        used += 1
        if abs(diff) >= abs(observed):
            extreme += 1
    if used == 0:
        return observed, None, 0
    return observed, (extreme + 1) / (used + 1), used


def reaction_check(pairs: Sequence[tuple[str, float]], lookback_minutes: int) -> dict:
    """APPROACH の手前と AVOID の手前で、値動きが違うか。

    違えば、答えは直前の値動き（画像に写っているもの）で変わっている。それは画像への反応であって、
    この先の値動きを当てているかとは別の話（そちらは `evaluate.py` の `forward_returns`）。

    p 値は 2 つ出す。読むのは `p_value_shift`（答えの続きやすさを考えた値）。
    `p_value_permutation` は `evaluate.py` と同じ検定で、答えが続く腕では小さく出すぎる。比べるために残す。
    """
    by_direction: dict[str, list[float]] = {d: [] for d in DIRECTIONS}
    for direction, value in pairs:
        by_direction[direction].append(value)
    approach = by_direction[APPROACH]
    avoid = by_direction[AVOID]
    difference, p_permutation = evaluate.permutation_test(approach, avoid)
    _, p_shift, shifts = shift_test(pairs)
    minimum = evaluate.MIN_SAMPLES_PER_DIRECTION
    enough = len(approach) >= minimum and len(avoid) >= minimum
    return {
        "lookback_minutes": lookback_minutes,
        "before_approach": evaluate._summarize(approach),
        "before_avoid": evaluate._summarize(avoid),
        "before_none": evaluate._summarize(by_direction[NONE]),
        "difference_pct": _round(difference, 4),
        "p_value_shift": _round(p_shift, 5),
        "shifts": shifts,
        "p_value_permutation": _round(p_permutation, 5),
        "permutation_trials": evaluate.PERMUTATION_TRIALS,
        "permutation_seed": evaluate.PERMUTATION_SEED,
        "enough_samples": enough,
        "sample_warning": None
        if enough
        else (
            f"標本が少ない（APPROACH {len(approach)} 件 / AVOID {len(avoid)} 件、"
            f"各 {minimum} 件以上ほしい）。この差は偶然と区別できない"
        ),
    }


def persistence(snaps: Sequence[Snapshot]) -> dict:
    """同じ答えが続くか。続けて記録された 2 回の両方で方向が出た組だけを数える。

    前の回と関係なく出る答え（ランダム）なら、同じ答えが続く割合は頻度の二乗和に、
    ある方向のあとに同じ方向が出る割合はその方向の頻度に近くなる。
    """
    decided = [s.direction for s in snaps if s.direction is not None]
    pairs = [(a.direction, b.direction) for a, b in zip(snaps, snaps[1:]) if a.direction and b.direction]
    if not decided or not pairs:
        return {"pairs": len(pairs), "same_as_previous_ratio": None, "independent_expectation": None, "stay": {}}
    frequency = {d: decided.count(d) / len(decided) for d in DIRECTIONS}
    stay = {}
    for d in DIRECTIONS:
        after = [b for a, b in pairs if a == d]
        stay[d] = {
            "after": len(after),
            "stayed_ratio": round(after.count(d) / len(after), 4) if after else None,
            "frequency": round(frequency[d], 4),
        }
    return {
        "pairs": len(pairs),
        "same_as_previous_ratio": round(sum(a == b for a, b in pairs) / len(pairs), 4),
        "independent_expectation": round(sum(f * f for f in frequency.values()), 4),
        "stay": stay,
    }


def image_lookbacks(config) -> list[int]:
    """手前の値動きを測る長さの既定。最後の 1 本と、画像に写る範囲（agent.yaml の fly.vision）。"""
    minutes = CANDLE_MINUTES.get(config.vision_candle_type)
    if minutes is None:
        raise ValueError(f"足の種類を分に直せない: {config.vision_candle_type}")
    return sorted({minutes, minutes * config.vision_lookback_candles})


def arm_report(
    arm: evaluate.Arm,
    lookbacks: Sequence[int] | None,
    tolerance_minutes: int,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
) -> dict:
    tz = arm.config.timezone
    snaps = snapshots(arm.records, tz)
    obs = evaluate.observations(arm.records, tz)
    minutes = list(lookbacks) if lookbacks else image_lookbacks(arm.config)
    return {
        "label": arm.label,
        "direction_source": arm.source,
        "config_fingerprints": list(arm.fingerprints),
        "mixed_config": len(arm.fingerprints) > 1,
        "coverage": coverage(snaps, interval_minutes),
        "holdings": holdings(snaps),
        "gaps": gaps(snaps, interval_minutes, tolerance_minutes),
        "reaction": [reaction_check(backward_pairs(obs, m, tolerance_minutes), m) for m in minutes],
        "persistence": persistence(snaps),
    }


def breakdown(
    arms: Sequence[evaluate.Arm],
    lookbacks: Sequence[int] | None,
    tolerance_minutes: int,
    window: dict | None = None,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
) -> dict:
    """腕ごとの内訳を並べる。結論は書かない。"""
    reports = [arm_report(arm, lookbacks, tolerance_minutes, interval_minutes) for arm in arms]
    start, end = evaluate.overlap(arms)
    notes: list[str] = []
    if len(arms) < 2:
        notes.append("腕が1つしかない。腕どうしの比較にはならない")
    for report in reports:
        label = report["label"]
        if report["mixed_config"]:
            notes.append(
                f"{label}: 設定の指紋が {len(report['config_fingerprints'])} 種類ある。値を変えた前後の回が混ざっている"
            )
        cover = report["coverage"]
        if cover["ratio"] is not None and cover["ratio"] < LOW_COVERAGE_RATIO:
            notes.append(
                f"{label}: 記録は本来の回数の {cover['ratio']:.0%}（{cover['records']} / {cover['expected_rounds']} 回）。"
                "止まっていた間の値動きは gaps に出す"
            )
        pnl = report["holdings"]["pnl"]
        if pnl and pnl["unattributed_intervals"]:
            notes.append(
                f"{label}: paper に触った回・確かめられない回をまたぐ区間が {pnl['unattributed_intervals']} 個ある。"
                "その増減は unattributed_jpy に入れ、分けていない"
            )
        for check in report["reaction"]:
            if not check["enough_samples"]:
                notes.append(f"{label}（手前 {check['lookback_minutes']} 分）: {check['sample_warning']}")
    return {
        # 2: 観測の欠けた回をまたぐ区間も、paper に触っていなければ分けるようにした。coverage と gaps を足した
        "schema_version": 2,
        "window": window,
        "tolerance_minutes": tolerance_minutes,
        "interval_minutes": interval_minutes,
        "overlap": {"from": start, "to": end},
        "arms": reports,
        "notes": notes,
        # 内訳は差の出どころの手がかりであって、結論ではない。読んだ人間が判断する。
        "verdict": None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m dopabae.breakdown",
        description="腕ごとの成績の内訳（記録の数と空白・建玉の重さ・方向の手前の値動き・方向の続きかた）。読み取りのみ。発注しない",
    )
    parser.add_argument(
        "--config", action="append", default=None, metavar="PATH",
        help="腕の agent.yaml。繰り返し指定する（scripts/make_arm.py で作る）",
    )
    parser.add_argument(
        "--lookback-minutes", action="append", type=int, default=None, metavar="N",
        help="方向の手前の何分の値動きを見るか。繰り返し指定できる。"
             "省略すると最後の 1 本と画像に写る範囲（agent.yaml の fly.vision）",
    )
    parser.add_argument("--tolerance-minutes", type=int, default=8, help="観測時刻のずれの許容（既定は15分足の半分）")
    parser.add_argument(
        "--interval-minutes", type=int, default=DEFAULT_INTERVAL_MINUTES,
        help="定期実行の間隔。空白の判定（間隔 + 許容より長い区間）と、本来の回数の見積もりに使う",
    )
    parser.add_argument("--out", default=None, help="JSON の書き出し先。省略すると標準出力だけ")
    parser.add_argument("--since", default=None, metavar="ISO8601",
                        help="この時刻以降の回だけで見る（例: 2026-10-16T11:27:00+09:00）")
    parser.add_argument("--until", default=None, metavar="ISO8601", help="この時刻までの回だけで見る")
    args = parser.parse_args(argv)

    if args.lookback_minutes and any(m <= 0 for m in args.lookback_minutes):
        print(json.dumps({"error": "--lookback-minutes は正の整数にする"}, ensure_ascii=False))
        return 1
    if args.interval_minutes <= 0:
        print(json.dumps({"error": "--interval-minutes は正の整数にする"}, ensure_ascii=False))
        return 1

    paths = args.config or [None]
    try:
        arms = [
            evaluate.load_arm(config_module.load(path), label=Path(path).stem if path else None,
                              since=args.since, until=args.until)
            for path in paths
        ]
        report = breakdown(arms, args.lookback_minutes, args.tolerance_minutes,
                           {"since": args.since, "until": args.until}, args.interval_minutes)
    except (config_module.ConfigError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{text}\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
