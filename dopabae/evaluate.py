"""評価。ハエの方向がランダムと区別できるかを測る。

設計メモ 4 章の評価 1・3・4 を回すための土台。**儲かったかではなく、
ランダムと区別できるかを見る。** 区別できなければ、ハエは「たまに買って、
たまに売る乱数」と同じである。

## 腕（arm）の作りかた

replay（過去の足を使った再実行）はしない。`bitbank paper` の約定判定を自前で
真似ると、真似が本物とずれた分だけ評価が嘘になる。代わりに、**方向の出どころだけを
変えた設定を並べて同時に走らせる。** ペーパー口座の状態ファイルと判断ログを腕ごとに
分ければ、同じ生の相場を見ながら独立に動く（`scripts/make_arm.py`）。

## 測るもの

| 何を                 | なぜ                                                       |
| -------------------- | ---------------------------------------------------------- |
| 方向の頻度と偏り     | いつも同じ方向なら、それは相場の洞察ではなくネットワークの癖 |
| 檻の扱いの内訳       | 方向が壁で消えているなら、成績の差は方向の差ではない        |
| 方向のあとの値動き   | APPROACH のあとと AVOID のあとで違わなければ、方向に情報がない |
| 成績                 | 総資産・最大ドローダウン・Buy&Hold 比較                     |

方向のあとの値動きの差には並べ替え検定を当てる。分布を仮定せず、同じ乱数種で
何度でも再現できる。**標本が少ないうちは、どの数字も意味を持たない。**
`sample_warning` にその旨を出す。
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Sequence

from . import config as config_module, performance as performance_module, timeutil
from .direction import APPROACH, AVOID, DIRECTIONS, NONE

# 並べ替え検定の既定。種を固定するので、同じ入力なら同じ p 値が出る。
PERMUTATION_TRIALS = 10_000
PERMUTATION_SEED = 20260915
# これを下回る標本で出た差は、偶然と区別できない。数字は出すが、警告を添える。
MIN_SAMPLES_PER_DIRECTION = 30
# 偏りがこれを超えたら「ほぼ一方向」として記録する（Stonkfly の「回転の癖が買い癖になる」）。
BIAS_ALERT_RATIO = 0.9


def _pct(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


@dataclass(frozen=True)
class DirectionStats:
    """方向の出かた。`unavailable` は方向を得られなかった回。"""

    total: int
    counts: dict[str, int]
    unavailable: int
    not_consulted: int

    @property
    def decided(self) -> int:
        return sum(self.counts.values())

    @property
    def frequencies(self) -> dict[str, float]:
        if self.decided == 0:
            return {d: 0.0 for d in DIRECTIONS}
        return {d: round(self.counts.get(d, 0) / self.decided, 4) for d in DIRECTIONS}

    @property
    def bias(self) -> tuple[str | None, float]:
        """最も多かった方向と、その割合。"""
        if self.decided == 0:
            return None, 0.0
        top = max(DIRECTIONS, key=lambda d: self.counts.get(d, 0))
        return top, round(self.counts.get(top, 0) / self.decided, 4)

    def as_dict(self) -> dict:
        top, ratio = self.bias
        return {
            "total": self.total,
            "decided": self.decided,
            "counts": {d: self.counts.get(d, 0) for d in DIRECTIONS},
            "frequencies": self.frequencies,
            "unavailable": self.unavailable,
            "not_consulted": self.not_consulted,
            "bias_direction": top,
            "bias_ratio": ratio,
            # 相場の洞察ではなくネットワークの癖として記録する（設計メモ 4 章）。
            "nearly_one_sided": ratio >= BIAS_ALERT_RATIO and self.decided > 0,
        }


def direction_stats(records: Sequence[dict]) -> DirectionStats:
    counts = {d: 0 for d in DIRECTIONS}
    unavailable = not_consulted = 0
    for record in records:
        direction = record.get("direction")
        if not isinstance(direction, dict):
            not_consulted += 1
            continue
        value = direction.get("value")
        if value in counts:
            counts[value] += 1
        else:
            unavailable += 1
    return DirectionStats(
        total=len(records), counts=counts, unavailable=unavailable, not_consulted=not_consulted
    )


def cage_stats(records: Sequence[dict]) -> dict[str, int]:
    """檻がその回をどう扱ったかの内訳。方向が壁で消えている割合を見る。"""
    counts: dict[str, int] = {}
    for record in records:
        label = record.get("cage")
        if isinstance(label, str):
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


@dataclass(frozen=True)
class Observation:
    at: object
    price: float
    direction: str


def observations(records: Sequence[dict], tz_name: str) -> list[Observation]:
    """方向と価格の両方が観測できた回だけを、時刻順に取り出す。"""
    result: list[Observation] = []
    for record in records:
        direction = record.get("direction")
        price = record.get("price")
        run_id = record.get("run_id")
        if not isinstance(direction, dict) or not isinstance(price, (int, float)):
            continue
        if not isinstance(run_id, str) or direction.get("value") not in DIRECTIONS:
            continue
        if price <= 0:
            continue
        try:
            at = timeutil.from_iso(run_id, tz_name)
        except ValueError:
            continue
        result.append(Observation(at=at, price=float(price), direction=str(direction["value"])))
    return sorted(result, key=lambda o: o.at)


def forward_returns(
    obs: Sequence[Observation], horizon_minutes: int, tolerance_minutes: int
) -> dict[str, list[float]]:
    """各方向のあと、`horizon_minutes` 後までの騰落率（%）。

    観測が欠けた区間は飛ばす。前の値で埋めると、観測していない値を作ることになる。
    """
    by_direction: dict[str, list[float]] = {d: [] for d in DIRECTIONS}
    horizon = timedelta(minutes=horizon_minutes)
    tolerance = timedelta(minutes=tolerance_minutes)
    for index, start in enumerate(obs):
        target = start.at + horizon
        best = None
        for later in obs[index + 1 :]:
            gap = later.at - target
            if gap > tolerance:
                break
            if abs(gap) <= tolerance:
                if best is None or abs(later.at - target) < abs(best.at - target):
                    best = later
        if best is None:
            continue
        by_direction[start.direction].append((best.price - start.price) / start.price * 100)
    return by_direction


def permutation_test(
    a: Sequence[float], b: Sequence[float], trials: int = PERMUTATION_TRIALS, seed: int = PERMUTATION_SEED
) -> tuple[float | None, float | None]:
    """2群の平均の差と、その両側 p 値。分布を仮定しない。

    どちらかが空なら (None, None)。p 値は (超えた回数 + 1) / (試行 + 1) で、
    0 にはならない。試行回数で切れる精度より細かいことは言わない。
    """
    if not a or not b:
        return None, None
    observed = statistics.fmean(a) - statistics.fmean(b)
    pooled = list(a) + list(b)
    split = len(a)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(trials):
        rng.shuffle(pooled)
        diff = statistics.fmean(pooled[:split]) - statistics.fmean(pooled[split:])
        if abs(diff) >= abs(observed):
            extreme += 1
    return observed, (extreme + 1) / (trials + 1)


def _summarize(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0, "mean_pct": None, "median_pct": None, "stdev_pct": None}
    return {
        "n": len(values),
        "mean_pct": _pct(statistics.fmean(values)),
        "median_pct": _pct(statistics.median(values)),
        "stdev_pct": _pct(statistics.stdev(values)) if len(values) > 1 else None,
    }


def signal_check(by_direction: dict[str, list[float]]) -> dict:
    """APPROACH のあとと AVOID のあとで、値動きが違うか。

    違わなければ、方向に相場の情報は乗っていない。**違っても、それが
    利益になるとは限らない**（檻が方向をどう扱うかは別の話）。
    """
    approach = by_direction.get(APPROACH, [])
    avoid = by_direction.get(AVOID, [])
    difference, p_value = permutation_test(approach, avoid)
    enough = len(approach) >= MIN_SAMPLES_PER_DIRECTION and len(avoid) >= MIN_SAMPLES_PER_DIRECTION
    return {
        "after_approach": _summarize(approach),
        "after_avoid": _summarize(avoid),
        "after_none": _summarize(by_direction.get(NONE, [])),
        "difference_pct": _pct(difference),
        "p_value": None if p_value is None else round(p_value, 5),
        "permutation_trials": PERMUTATION_TRIALS,
        "permutation_seed": PERMUTATION_SEED,
        "enough_samples": enough,
        "sample_warning": None
        if enough
        else (
            f"標本が少ない（APPROACH {len(approach)} 件 / AVOID {len(avoid)} 件、"
            f"各 {MIN_SAMPLES_PER_DIRECTION} 件以上ほしい）。この差は偶然と区別できない"
        ),
    }


@dataclass
class Arm:
    """比べる対象1つぶん。方向の出どころだけが違う。"""

    label: str
    source: str
    fingerprints: tuple[str, ...]
    records: list[dict] = field(repr=False, default_factory=list)
    config: object = field(repr=False, default=None)

    def report(self, horizon_minutes: int, tolerance_minutes: int) -> dict:
        tz = self.config.timezone
        obs = observations(self.records, tz)
        points = performance_module.points_from(self.records, tz)
        stats = direction_stats(self.records)
        return {
            "label": self.label,
            "direction_source": self.source,
            # 設定を変えた前後の回が混ざっていないか。2つ以上なら混ざっている。
            "config_fingerprints": list(self.fingerprints),
            "mixed_config": len(self.fingerprints) > 1,
            "directions": stats.as_dict(),
            "cage": cage_stats(self.records),
            "forward_returns": signal_check(
                forward_returns(obs, horizon_minutes, tolerance_minutes)
            ),
            "performance": performance_module.measure(points).as_dict(),
            # この腕の頻度。ランダム対照群の direction.random_weights へ写す。
            "matched_random_weights": stats.frequencies,
        }


def load_arm(config, root: Path | None = None, label: str | None = None) -> Arm:
    """1つの設定（＝1つの腕）の判断ログを読む。"""
    records = performance_module.all_records(config, root)
    fingerprints = sorted(
        {
            fp
            for record in records
            if isinstance(record.get("config"), dict)
            and isinstance(fp := record["config"].get("fingerprint"), str)
        }
    )
    return Arm(
        label=label or config.direction_source,
        source=config.direction_source,
        fingerprints=tuple(fingerprints),
        records=records,
        config=config,
    )


def overlap(arms: Sequence[Arm]) -> tuple[str | None, str | None]:
    """腕どうしが共通して観測していた期間。ここから外れた回は比べられない。"""
    starts, ends = [], []
    for arm in arms:
        points = performance_module.points_from(arm.records, arm.config.timezone)
        if points:
            starts.append(points[0].at)
            ends.append(points[-1].at)
    if not starts:
        return None, None
    latest_start, earliest_end = max(starts), min(ends)
    if latest_start > earliest_end:
        return None, None
    return timeutil.to_iso(latest_start), timeutil.to_iso(earliest_end)


def compare(arms: Sequence[Arm], horizon_minutes: int, tolerance_minutes: int) -> dict:
    """腕を並べて比べる。結論は書かない。読んだ人間が判断する。"""
    reports = [arm.report(horizon_minutes, tolerance_minutes) for arm in arms]
    start, end = overlap(arms)
    notes: list[str] = []
    if len(arms) < 2:
        notes.append("腕が1つしかない。比較（設計メモ 4 章の評価 3・4）にはならない")
    if start is None and len(arms) > 1:
        notes.append("腕どうしで重なる観測期間がない。同じ期間で比べていない")
    for report in reports:
        if report["mixed_config"]:
            notes.append(
                f"{report['label']}: 設定の指紋が {len(report['config_fingerprints'])} 種類ある。"
                "値を変えた前後の回が混ざっている"
            )
        if report["directions"]["nearly_one_sided"]:
            notes.append(
                f"{report['label']}: 方向が {report['directions']['bias_direction']} に"
                f"{report['directions']['bias_ratio']:.0%} 偏っている。"
                "相場の洞察ではなくネットワークの癖として扱う"
            )
        if not report["forward_returns"]["enough_samples"]:
            notes.append(f"{report['label']}: {report['forward_returns']['sample_warning']}")
    return {
        "schema_version": 1,
        "horizon_minutes": horizon_minutes,
        "tolerance_minutes": tolerance_minutes,
        "overlap": {"from": start, "to": end},
        "arms": reports,
        "notes": notes,
        # 評価はここで結論を出さない。設計メモ 4 章の 1〜4 が揃うまで「学んだ」とは言わない。
        "verdict": None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m dopabae.evaluate",
        description="腕（方向の出どころだけが違う設定）を並べて比べる。読み取りのみ。発注しない",
    )
    parser.add_argument(
        "--config", action="append", default=None, metavar="PATH",
        help="腕の agent.yaml。繰り返し指定する（scripts/make_arm.py で作る）",
    )
    parser.add_argument("--horizon-minutes", type=int, default=60, help="方向のあと何分の値動きを見るか")
    parser.add_argument("--tolerance-minutes", type=int, default=8, help="観測時刻のずれの許容（既定は15分足の半分）")
    parser.add_argument("--out", default=None, help="JSON の書き出し先。省略すると標準出力だけ")
    args = parser.parse_args(argv)

    paths = args.config or [None]
    try:
        arms = [
            load_arm(config_module.load(path), label=Path(path).stem if path else None)
            for path in paths
        ]
    except config_module.ConfigError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1

    report = compare(arms, args.horizon_minutes, args.tolerance_minutes)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{text}\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
