#!/usr/bin/env python3
"""読み出しの候補が、画像・嗅覚の回路の点火・観測の繰り返しでどう発火するかを測る（Phase 3）。

`docs/IMPLEMENTATION_PLAN.md` 3 節 5 の「読み出しに使うニューロン群」を選ぶ材料を作る。
**結論は書かない。どの群を使うかは人間が決める。** 数字だけを出す。

測りかた:

- 画像: 決定的に作った合成のチャート 5 種（上昇・下落・横ばい・乱高下・急騰）と、
  背景だけの無地。**どれも相場の観測ではない**
- 順序: 状態を引き継ぎ、無地で 2 回ならしてから、5 種を `--cycles` 巡見せる
  （1 観測 = 神経時間 0.5 秒）。同じ画像を何度も見せるので、「画像による差」と
  「同じ画像の繰り返しでの揺れ」を比べられる
- 点火（Phase 3「状態」の件）: 3 通りで比べる
  - 止めた: 嗅覚の局所細胞 lLN1_bc に −500 mV の一定電流を流し続ける。**計測のための
    操作であって、モデルの選択ではない**（−30 mV では途中で点火した）
  - 自然: 何も足さない
  - 点火させた: 最初のならしのあいだだけ lLN1_bc に強いポアソン入力を与える

出すもの（群ごと。発火率は 1 細胞あたりの Hz）:

- 無地での発火率、画像での平均
- 画像による幅: 5 種の平均のうち最大 − 最小
- 揺れ: 同じ画像を繰り返したときの標準偏差（5 種の平均）
- p: 「画像によって発火率が違う」の並べ替え検定（画像の札を入れ替える。F 値を比べる）。
  **観測は状態を引き継いでいて独立ではない**ので、目安として読む
- 2 細胞の群は右 − 左。Stonkfly の閾値 2 Hz を超えた観測の割合も出す
- `--screen`: 「自然」の観測で、網のすべての型（発火した型）に同じ検定を当て、
  画像の情報がどの段階まで届くかを superclass ごとに数える（追加のシミュレーションは無い）

使いかた:

    python3 scripts/probe_readout.py            # 4 巡
    python3 scripts/probe_readout.py --cycles 2 --screen

結果は `var/bench/` に JSON で残る（Git 管理外）。
**発注しない。取引所にも触れない。** 配線図データだけを使う。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dopabae import connectome, lif, retina, vision  # noqa: E402
from dopabae.config import REPO_ROOT, Config, load  # noqa: E402

# 読み出しの候補。由来を残す。選ぶのは人間（3 節 5）。
CANDIDATES = (
    ("LC4", ("LC4",), None, "設計メモ 3.3（回避側）"),
    ("LPLC2", ("LPLC2",), None, "設計メモ 3.3（回避側）"),
    ("DNp01", ("DNp01",), None, "設計メモ 3.3（回避側）"),
    ("MBON", (), "MBON", "設計メモ 3.3（接近側・回避側）"),
    ("DNa02", ("DNa02",), None, "Stonkfly の読み出し候補"),
    ("DNp09", ("DNp09",), None, "Stonkfly の読み出し候補"),
    ("DNp20", ("DNp20",), None, "Stonkfly の読み出し（左右の差）"),
    ("DNpe017", ("DNpe017",), None, "Stonkfly の読み出し（関門）"),
    ("MDN", ("MDN",), None, "Stonkfly の読み出し候補"),
    ("MN9", ("MN9",), None, "Stonkfly の読み出し候補"),
)
# 比べるための群。読み出しの候補ではない。
REFERENCES = (
    ("R1-R6", ("R1-R6",), None, "入力"),
    ("ラミナ", ("L1", "L2", "L3", "L5"), None, "背景の電流を受ける"),
    ("lLN1_bc", ("lLN1_bc",), None, "点火する回路"),
    ("KC", (), "KC", "点火に左右される"),
)
IGNITION_TYPE = "lLN1_bc"
IGNITION_RATE_HZ = 200.0
IGNITION_WEIGHT_SYNAPSES = 250.0
KNOCKOUT_MV = -500.0
CONDITIONS = (("knockout", "止めた"), ("natural", "自然"), ("ignited", "点火させた"))
TIMELINE = ("lLN1_bc", "KC", "MBON", "DNa02", "DNp20", "MDN", "MN9")
PAIRS = ("DNa02", "DNp20", "DNpe017", "MN9")
STONKFLY_LR_HZ = 2.0
PERMUTATION_TRIALS = 2000
PERMUTATION_SEED = 20261006

# 合成のチャート。終値の動き（円）を 8 本ぶん。相場の観測ではない。
CHARTS = {
    "上昇": (30_000,) * 8,
    "下落": (-30_000,) * 8,
    "横ばい": (5_000, -5_000) * 4,
    "乱高下": (90_000, -110_000, 120_000, -80_000, 100_000, -120_000, 90_000, -60_000),
    "急騰": (0,) * 7 + (200_000,),
}
BLANK = "無地"
WARMUP = 2


def chart_candles(moves: tuple[int, ...], now_ms: int) -> tuple[vision.Candle, ...]:
    step = 15 * 60_000
    close = 14_700_000
    candles = []
    for i, move in enumerate(moves):
        open_ = close
        close = open_ + move
        candles.append(
            vision.Candle(
                timestamp_ms=now_ms - (len(moves) - 1 - i) * step,
                open=Decimal(open_),
                high=Decimal(max(open_, close) + 10_000),
                low=Decimal(min(open_, close) - 10_000),
                close=Decimal(close),
            )
        )
    return tuple(candles)


def blank_image(config: Config) -> vision.Image:
    """背景の色だけの画像。チャートも文字も描かない。"""
    pixels = bytes(config.vision_palette["background"]) * (config.vision_width * config.vision_height)
    return vision.Image(
        width=config.vision_width,
        height=config.vision_height,
        pixels=pixels,
        sha256=hashlib.sha256(pixels).hexdigest(),
        candle_type=config.vision_candle_type,
        candle_count=0,
        candles_from_ms=None,
        candles_to_ms=None,
        shown=(),
    )


def images(config: Config) -> dict[str, vision.Image]:
    now_ms = 1_789_430_400_000
    out = {BLANK: blank_image(config)}
    for name, moves in CHARTS.items():
        candles = chart_candles(moves, now_ms)
        if len(candles) != config.vision_lookback_candles:
            raise SystemExit(f"合成のチャートは {config.vision_lookback_candles} 本で作る（いま {len(candles)} 本）")
        last = candles[-1].close
        out[name] = vision.render(config, candles, last - 500, last + 500, config.pair)
    return out


def groups(network: connectome.Network) -> dict[str, dict]:
    out = {}

    def add(name, index, origin, candidate):
        out[name] = {
            "index": index,
            "left": index[network.side[index] == "L"],
            "right": index[network.side[index] == "R"],
            "origin": origin,
            "candidate": candidate,
        }

    for name, names, prefix, origin in CANDIDATES:
        add(name, network.type_index(names, prefix), origin, True)
    for name, names, prefix, origin in REFERENCES:
        add(name, network.type_index(names, prefix), origin, False)
    for name in sorted({t for t in network.cell_type if t.startswith("MBON")}):
        add(f"  {name}", network.type_index((name,)), "MBON の型", True)
    return out


def rates(counts, group: dict, seconds: float) -> dict:
    def rate(index):
        return float(counts[index].sum()) / len(index) / seconds if len(index) else None

    return {"all": rate(group["index"]), "left": rate(group["left"]), "right": rate(group["right"])}


def run(network, params, lights, sequence, condition, duration_ms, seed):
    import numpy as np

    index = network.type_index((IGNITION_TYPE,))
    drive = stimulus = None
    if condition == "ignited":
        drive = lif.Drive(index=index, rate_hz=np.full(len(index), IGNITION_RATE_HZ),
                          weight_mv=IGNITION_WEIGHT_SYNAPSES * params.weight_per_synapse_mv)
    if condition == "knockout":
        stimulus = lif.Stimulus(index=index, current_mv=KNOCKOUT_MV)
    state = None
    records = []
    for step, name in enumerate(sequence):
        result = lif.simulate(network, params, duration_ms, light=lights[name],
                              drive=drive if step == 0 else None, stimulus=stimulus,
                              state=state, seed=seed + step)
        state = result.state
        records.append({"image": name, "counts": result.spike_counts, "wall_sec": result.wall_sec,
                        "total_spikes": result.total_spikes})
        print(f"    {step + 1:2d} {name:4s} {result.wall_sec:5.1f} 秒  発火 {result.total_spikes:>9,}", flush=True)
    return records


def f_statistic(labels: list[str], values: list[float]) -> float | None:
    """一元配置の F 値（群間の分散 / 群内の分散）。群内が 0 なら None。"""
    by: dict[str, list[float]] = {}
    for label, value in zip(labels, values):
        by.setdefault(label, []).append(value)
    grand = statistics.fmean(values)
    k, n = len(by), len(values)
    between = sum(len(v) * (statistics.fmean(v) - grand) ** 2 for v in by.values()) / (k - 1)
    within = sum(sum((x - statistics.fmean(v)) ** 2 for x in v) for v in by.values()) / (n - k)
    return None if within == 0 else between / within


def image_permutation_test(labels: list[str], values: list[float]) -> tuple[float | None, float | None]:
    """画像の札を入れ替えて F 値を比べる。p は (超えた回数 + 1) / (試行 + 1)。"""
    observed = f_statistic(labels, values)
    if observed is None:
        return None, None
    rng = random.Random(PERMUTATION_SEED)
    shuffled = list(labels)
    extreme = 0
    for _ in range(PERMUTATION_TRIALS):
        rng.shuffle(shuffled)
        f = f_statistic(shuffled, values)
        if f is not None and f >= observed:
            extreme += 1
    return observed, (extreme + 1) / (PERMUTATION_TRIALS + 1)


def screen_types(config: Config, network: connectome.Network, records, top: int = 15) -> dict:
    """全型の画像による差の並べ替え検定（F 値と同順の群間平方和で比べる）。numpy で一度に回す。"""
    import numpy as np
    import pyarrow.feather as feather

    shown = records[WARMUP:]
    charts = list(CHARTS)
    labels = np.array([charts.index(r["image"]) for r in shown])
    counts = np.array([r["counts"] for r in shown], dtype=np.float64)  # 観測 × 細胞
    names, inverse = np.unique(network.cell_type, return_inverse=True)
    by_type = np.zeros((len(names), len(shown)))
    for j in range(len(shown)):
        by_type[:, j] = np.bincount(inverse, weights=counts[j], minlength=len(names))
    active = by_type.std(axis=1) > 0
    values, names = by_type[active], names[active]

    one_hot = np.eye(len(charts))

    def between(order):
        groups = one_hot[order]
        sums = values @ groups
        return (sums ** 2 / groups.sum(axis=0)).sum(axis=1) - values.sum(axis=1) ** 2 / values.shape[1]

    observed = between(labels)
    rng = np.random.default_rng(PERMUTATION_SEED)
    extreme = np.zeros(len(values))
    for _ in range(PERMUTATION_TRIALS):
        extreme += between(rng.permutation(labels)) >= observed - 1e-9
    p_values = (extreme + 1) / (PERMUTATION_TRIALS + 1)

    from dopabae.connectome import verified_input

    table = feather.read_table(verified_input(config, "annotations"), columns=["type", "superclass"]).to_pandas()
    superclass = table.dropna(subset=["type"]).groupby("type").superclass.agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else None
    )
    by_superclass: dict[str, dict[str, int]] = {}
    for name, p in zip(names, p_values):
        key = str(superclass.get(name, "untyped") if name else "untyped")
        row = by_superclass.setdefault(key, {"active": 0, "p_le_0.01": 0, "p_le_0.001": 0})
        row["active"] += 1
        row["p_le_0.01"] += int(p <= 0.01)
        row["p_le_0.001"] += int(p <= 0.001)
    order = np.argsort(p_values, kind="stable")
    return {
        "active_types": int(active.sum()),
        "all_types": int(len(active)),
        "by_superclass": dict(sorted(by_superclass.items(), key=lambda kv: -kv[1]["active"])),
        "top": [{"type": str(names[i]), "superclass": str(superclass.get(names[i])), "p": float(p_values[i])}
                for i in order[:top]],
    }


def summarize(group_map, runs, seconds):
    """群ごとの数字。観測した発火率から計算するだけで、良し悪しは書かない。"""
    summary = {}
    for name, group in group_map.items():
        per = {}
        for label, records in runs.items():
            series = [(r["image"], rates(r["counts"], group, seconds)) for r in records]
            shown = series[WARMUP:]
            labels = [img for img, _ in shown]
            values = [v["all"] for _, v in shown]
            by_chart: dict[str, list[float]] = {}
            for img, v in shown:
                by_chart.setdefault(img, []).append(v["all"])
            means = {k: statistics.fmean(v) for k, v in by_chart.items()}
            f_value, p_value = image_permutation_test(labels, values)
            entry = {
                "blank_hz": series[WARMUP - 1][1]["all"],
                "chart_mean_hz": statistics.fmean(means.values()),
                "chart_range_hz": max(means.values()) - min(means.values()),
                "within_sd_hz": statistics.fmean(statistics.stdev(v) for v in by_chart.values()),
                "f_value": f_value,
                "p_value": p_value,
                "by_chart_hz": by_chart,
                "timeline_hz": [v["all"] for _, v in series],
            }
            if len(group["left"]) and len(group["right"]):
                lr = [v["right"] - v["left"] for _, v in shown]
                entry["right_minus_left"] = {
                    "mean_hz": statistics.fmean(lr),
                    "share_at_least_plus": sum(x >= STONKFLY_LR_HZ for x in lr) / len(lr),
                    "share_at_most_minus": sum(x <= -STONKFLY_LR_HZ for x in lr) / len(lr),
                    "by_observation_hz": lr,
                }
            per[label] = entry
        summary[name] = {
            "cells": int(len(group["index"])),
            "origin": group["origin"],
            "candidate": group["candidate"],
            **per,
            "ignition_effect_hz": per["natural"]["chart_mean_hz"] - per["knockout"]["chart_mean_hz"],
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cycles", type=int, default=4, help="5 種を何巡見せるか（既定 4）")
    parser.add_argument("--screen", action="store_true", help="全型の検定（自然の観測で）も出す")
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.cycles < 2:
        parser.error("--cycles は 2 以上（揺れを測るため）")

    config = load()
    params = lif.Params.from_settings(config.simulation)
    duration_ms = config.simulation.neural_time_sec * 1000.0
    seconds = duration_ms / 1000.0
    spec = connectome.spec_from_config(config)
    try:
        network, _ = connectome.load_or_compile(config, spec)
        column_map = retina.load_column_map(REPO_ROOT / config.retina_column_map_path)
    except (connectome.ConnectomeError, retina.RetinaError) as exc:
        print(f"準備できません: {exc}", file=sys.stderr)
        return 1

    pictures = images(config)
    lights = {name: lif.light_from_values(network, retina.activations(config, img, column_map).currents)[0]
              for name, img in pictures.items()}
    sequence = [BLANK] * WARMUP + list(CHARTS) * args.cycles
    group_map = groups(network)

    print(f"網: 細胞 {network.n_neurons:,} / 結合 {network.n_edges:,}（{spec.key()}）")
    print(f"順序: 無地 × {WARMUP} → チャート 5 種 × {args.cycles} 巡（各 {duration_ms:g} ms、状態を引き継ぐ）")
    runs = {}
    for key, label in CONDITIONS:
        print(f"  {label}")
        runs[key] = run(network, params, lights, sequence, key, duration_ms, args.seed)
    summary = summarize(group_map, runs, seconds)

    def fmt(value, width=7, digits=2):
        return " " * (width - 1) + "-" if value is None else f"{value:{width}.{digits}f}"

    print()
    print("群ごとの発火率（Hz / 細胞）。幅 = 画像 5 種の平均の最大 − 最小、揺れ = 同じ画像の繰り返しの標準偏差、"
          "p = 画像による差の並べ替え検定、点火の差 = 自然 − 止めた")
    header = f"{'群':14s}{'細胞':>5s}"
    for _, label in CONDITIONS:
        header += f" | {label + ': 画像':>10s}{'幅':>7s}{'揺れ':>7s}{'p':>7s}"
    header += f" | {'点火の差':>7s}"
    print(header)
    for name, s_ in summary.items():
        if name.startswith("  ") and all(s_[k]["chart_mean_hz"] == 0 for k, _ in CONDITIONS):
            continue  # どの条件でも発火しない MBON の型は表から省く（JSON には残る）
        row = f"{name:14s}{s_['cells']:5d}"
        for key, _ in CONDITIONS:
            c = s_[key]
            row += (f" | {fmt(c['chart_mean_hz'], 10)}{fmt(c['chart_range_hz'])}"
                    f"{fmt(c['within_sd_hz'])}{fmt(c['p_value'], 7, 3)}")
        row += f" | {fmt(s_['ignition_effect_hz'])}"
        print(row)

    print()
    print(f"観測ごとの推移（Hz / 細胞）: 無地 × {WARMUP} → " + " → ".join(CHARTS) + f" × {args.cycles}")
    for name in TIMELINE:
        for key, label in CONDITIONS:
            values = " ".join(f"{v:5.1f}" for v in summary[name][key]["timeline_hz"])
            print(f"  {name:8s} {label:5s} {values}")

    print()
    print(f"2 細胞の群の右 − 左。平均（Hz）と、+{STONKFLY_LR_HZ:g} Hz 以上 / −{STONKFLY_LR_HZ:g} Hz 以下だった観測の割合"
          "（Stonkfly はこの閾値で買い / 売りを提案する）")
    for name in PAIRS:
        for key, label in CONDITIONS:
            lr = summary[name][key].get("right_minus_left")
            if lr:
                print(f"  {name:8s} {label:5s}: 平均 {lr['mean_hz']:+6.1f}  "
                      f"+{STONKFLY_LR_HZ:g} 以上 {lr['share_at_least_plus']:5.0%}  "
                      f"−{STONKFLY_LR_HZ:g} 以下 {lr['share_at_most_minus']:5.0%}")

    screen = None
    if args.screen:
        screen = screen_types(config, network, runs["natural"])
        trials = screen["active_types"]
        print()
        print(f"全型の検定（自然）: 発火した型 {trials:,} / {screen['all_types']:,}。"
              f"偶然でも p ≦ 0.01 は約 {trials * 0.01:.0f} 型、p ≦ 0.001 は約 {trials * 0.001:.0f} 型出る")
        print(f"  {'superclass':22s}{'発火':>7s}{'p≦0.01':>9s}{'p≦0.001':>10s}")
        for key, row in screen["by_superclass"].items():
            print(f"  {key:22s}{row['active']:7d}{row['p_le_0.01']:9d}{row['p_le_0.001']:10d}")

    started = datetime.now(timezone.utc)
    report = {
        "started_at": started.isoformat(),
        "network_key": spec.key(),
        "sequence": sequence,
        "duration_ms": duration_ms,
        "conditions": {
            "knockout": {"type": IGNITION_TYPE, "current_mv": KNOCKOUT_MV, "observations": "all"},
            "natural": {},
            "ignited": {"type": IGNITION_TYPE, "rate_hz": IGNITION_RATE_HZ,
                        "weight_synapses": IGNITION_WEIGHT_SYNAPSES, "observations": [1]},
        },
        "permutation": {"trials": PERMUTATION_TRIALS, "seed": PERMUTATION_SEED, "statistic": "F"},
        "images": {name: img.sha256 for name, img in pictures.items()},
        "column_map_sha256": column_map.sha256,
        "wall_sec": {k: [r["wall_sec"] for r in v] for k, v in runs.items()},
        "total_spikes": {k: [r["total_spikes"] for r in v] for k, v in runs.items()},
        "groups": summary,
        "screen": screen,
    }
    out = args.out or REPO_ROOT / "var" / "bench" / f"readout-{started:%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"結果: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
