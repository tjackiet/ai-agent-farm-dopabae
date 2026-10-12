#!/usr/bin/env python3
"""全型の検定で画像による差が出た型を、別の画像の組で確かめ直す（Phase 3）。

`probe_readout.py --screen` は、合成のチャート 5 種で型を選び、同じ観測で p を出している。
そのまま読み出しに使うと、偶然の差を拾う（選んだデータで評価しているため）。
ここでは、確かめる型の一覧を `--discovery` の JSON から**走らせる前に決め**、
別の画像の組で検定を当て直す。**結論は書かない。数字だけを出す。**

別の画像の組:

- 合成のチャート 6 種。上げと下げを上下に映した 3 組（V字と逆V字、押し目上げと戻り売り、
  急騰のあとと急落のあと）。`probe_readout.py` の 5 種とは形が違う。**どれも相場の観測ではない**
- 最後の終値をそろえる。文字（ペア名・気配の数字）が 6 種で同じになり、文字の差が混ざらない
- 1 組の 2 枚は、足の大きさが同じで、並びと色（上昇足・下落足）だけが違う
- 順序: 無地で 2 回ならしてから、6 種を `--cycles` 巡見せる。**巡ごとに順序を入れ替える**
  （種は固定）。状態を引き継ぐので、順序が固定だと「直前に見た画像」の差が画像の差に混ざる
- 嗅覚の回路の点火は「自然」だけ（`--screen` と同じ条件）。この条件のシミュレーションは乱数を使わない

当てる検定（どれも並べ替え検定。種を固定）:

- 画像による差: 6 種の札を入れ替えて、群間平方和（F 値と同順）を比べる。`--screen` と同じ
- 上げと下げの差: 1 組の中だけで札を入れ替え、「上げ 3 枚の平均 − 下げ 3 枚の平均」の
  絶対値を比べる。どちらを上げと呼ぶか（V字・押し目上げ・急騰のあと）は設計者の読み替えである
- 確かめる型の数だけ検定を並べるので、Holm の方法で補正したあとの数も出す

使いかた:

    python3 scripts/validate_readout.py --config arms/fly-hs1.yaml \\
        --discovery var/bench/readout-hs1.json

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

import probe_readout as probe  # noqa: E402  同じ scripts/ にある

# 最後の終値。6 種でそろえ、文字（気配の数字）を同じにする。
END_CLOSE = 15_000_000
# 合成のチャート。終値の動き（円）を 8 本ぶん。相場の観測ではない。
CHARTS = {
    "V字": (-50_000,) * 4 + (50_000,) * 4,
    "逆V字": (50_000,) * 4 + (-50_000,) * 4,
    "押し目上げ": (60_000, 60_000, -30_000, -30_000, 60_000, 60_000, -30_000, 60_000),
    "戻り売り": (-60_000, -60_000, 30_000, 30_000, -60_000, -60_000, 30_000, -60_000),
    "急騰のあと": (150_000,) + (0,) * 7,
    "急落のあと": (-150_000,) + (0,) * 7,
}
# 上下に映した組。左を「上げ」と呼ぶ。設計者の読み替えであって、ハエの区別ではない。
PAIRS = (("V字", "逆V字"), ("押し目上げ", "戻り売り"), ("急騰のあと", "急落のあと"))
BLANK = probe.BLANK
WARMUP = probe.WARMUP
ORDER_SEED = 20261012
PERMUTATION_TRIALS = 20_000
PERMUTATION_SEED = 20261012
# 視葉の中の型。画像による差が出て当たり前なので、表では分けて数える。
OPTIC_LOBE = ("ol_sensory", "ol_intrinsic")


def chart_candles(moves: tuple[int, ...], now_ms: int) -> tuple[vision.Candle, ...]:
    """終値が END_CLOSE で終わる足を作る。ひげは実体の上下 1 万円（`probe_readout` と同じ）。"""
    step = 15 * 60_000
    close = END_CLOSE - sum(moves)
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


def images(config: Config) -> dict[str, tuple[vision.Image, ...]]:
    """画像ごとに、1 観測のあいだに見せる枚を順に並べる（静止画なら 1 枚）。"""
    now_ms = 1_789_430_400_000
    out = {BLANK: (probe.blank_image(config),)}
    for name, moves in CHARTS.items():
        candles = chart_candles(moves, now_ms)
        if len(candles) != config.vision_lookback_candles:
            raise SystemExit(f"合成のチャートは {config.vision_lookback_candles} 本で作る（いま {len(candles)} 本）")
        out[name] = vision.render_frames(config, candles, Decimal(END_CLOSE - 500), Decimal(END_CLOSE + 500),
                                         config.pair)
    return out


def sequence(cycles: int, seed: int = ORDER_SEED) -> list[str]:
    """無地 × WARMUP のあと、6 種を巡ごとに入れ替えた順で並べる。同じ種なら同じ順。"""
    rng = random.Random(seed)
    out = [BLANK] * WARMUP
    for _ in range(cycles):
        order = list(CHARTS)
        rng.shuffle(order)
        out += order
    return out


def image_test(values, labels, n_groups: int, trials: int, seed: int):
    """画像の札を入れ替えて群間平方和を比べる。p は (超えた回数 + 1) / (試行 + 1)。

    values は 型 × 観測。全体の平方和は入れ替えで変わらないので、群間平方和は F 値と同順。
    """
    import numpy as np

    one_hot = np.eye(n_groups)

    def between(order):
        groups = one_hot[order]
        sums = values @ groups
        return (sums ** 2 / groups.sum(axis=0)).sum(axis=1) - values.sum(axis=1) ** 2 / values.shape[1]

    observed = between(labels)
    rng = np.random.default_rng(seed)
    extreme = np.zeros(len(values))
    for _ in range(trials):
        extreme += between(rng.permutation(labels)) >= observed - 1e-9
    return (extreme + 1) / (trials + 1)


def direction_test(values, labels, pairs: list[tuple[int, int]], trials: int, seed: int):
    """上げ − 下げの平均の差と、組の中だけで札を入れ替えた並べ替え検定（両側）の p。

    labels は観測ごとの画像の番号、pairs は (上げの番号, 下げの番号)。組に入らない観測は使わない。
    """
    import numpy as np

    labels = np.asarray(labels)
    up = np.zeros(len(labels), dtype=bool)
    blocks = []
    for up_label, down_label in pairs:
        up |= labels == up_label
        blocks.append(np.flatnonzero((labels == up_label) | (labels == down_label)))
    used = np.concatenate(blocks)

    def weights(mask):
        w = np.zeros(len(labels))
        w[used] = np.where(mask[used], 1.0 / mask[used].sum(), -1.0 / (~mask[used]).sum())
        return w

    observed = values @ weights(up)
    rng = np.random.default_rng(seed)
    extreme = np.zeros(len(values))
    for _ in range(trials):
        shuffled = up.copy()
        for block in blocks:
            shuffled[block] = rng.permutation(up[block])
        extreme += np.abs(values @ weights(shuffled)) >= np.abs(observed) - 1e-9
    return observed, (extreme + 1) / (trials + 1)


def holm(p_values, alpha: float = 0.05) -> list[bool]:
    """Holm の方法。小さい順に alpha / (残りの数) と比べ、超えたところで止める。"""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    reject = [False] * len(p_values)
    for rank, i in enumerate(order):
        if p_values[i] > alpha / (len(p_values) - rank):
            break
        reject[i] = True
    return reject


def preregistered(discovery: dict, threshold: float) -> list[dict]:
    """確かめる型を、発見の JSON だけから決める（走らせる前）。"""
    screen = discovery.get("screen") or {}
    found = screen.get("p_le_0.01")
    if found is None:
        raise SystemExit("--discovery に screen.p_le_0.01 がありません。probe_readout.py --screen で作り直してください")
    return [t for t in found if t["p"] <= threshold]


def type_rates(network: connectome.Network, records, seconds: float):
    """型ごとの発火率（Hz / 細胞）。型 × 観測。"""
    import numpy as np

    names, inverse = np.unique(network.cell_type, return_inverse=True)
    sizes = np.bincount(inverse, minlength=len(names))
    rates = np.zeros((len(names), len(records)))
    for j, r in enumerate(records):
        rates[:, j] = np.bincount(inverse, weights=r["counts"], minlength=len(names))
    return names, rates / sizes[:, None] / seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--discovery", type=Path, required=True,
                        help="型を選んだ probe_readout.py --screen の JSON（同じ設定で作ったもの）")
    parser.add_argument("--discovery-p", type=float, default=0.001,
                        help="確かめる型の基準。発見の p がこれ以下の型（既定 0.001）")
    parser.add_argument("--config", type=Path, default=None, help="使う設定（既定は agent.yaml）")
    parser.add_argument("--cycles", type=int, default=8, help="6 種を何巡見せるか（既定 8）")
    parser.add_argument("--order-seed", type=int, default=ORDER_SEED, help="巡ごとの順序を決める種")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.cycles < 2:
        parser.error("--cycles は 2 以上（揺れを測るため）")

    import numpy as np

    discovery_bytes = args.discovery.read_bytes()
    discovery = json.loads(discovery_bytes)
    chosen = preregistered(discovery, args.discovery_p)

    config = load(args.config)
    params = lif.Params.from_settings(config.simulation)
    found_hs = (discovery.get("photoreceptor_input") or {}).get("half_saturation")
    if found_hs != params.input_half_saturation:
        print(f"発見の半飽和（{found_hs}）と設定（{params.input_half_saturation}）が違います。"
              "同じ網の同じ入力で確かめ直す", file=sys.stderr)
        return 1
    # 動きの設定も同じでなければ、発見と同じ入力で確かめたことにならない（無い JSON は静止画で測ったもの）。
    found_motion = discovery.get("vision_motion") or {"mode": "none", "frames": 1}
    if found_motion != probe.motion_record(config):
        print(f"発見の動き（{found_motion}）と設定（{probe.motion_record(config)}）が違います", file=sys.stderr)
        return 1
    duration_ms = config.simulation.neural_time_sec * 1000.0
    seconds = duration_ms / 1000.0
    spec = connectome.spec_from_config(config)
    if discovery.get("network_key") != spec.key():
        print(f"発見の網（{discovery.get('network_key')}）と設定の網（{spec.key()}）が違います", file=sys.stderr)
        return 1
    try:
        network, _ = connectome.load_or_compile(config, spec)
        column_map = retina.load_column_map(REPO_ROOT / config.retina_column_map_path)
    except (connectome.ConnectomeError, retina.RetinaError) as exc:
        print(f"準備できません: {exc}", file=sys.stderr)
        return 1

    print(f"確かめる型: 発見の p ≦ {args.discovery_p:g} の {len(chosen)} 型（{args.discovery}）")
    print("  " + "・".join(t["type"] for t in chosen))

    pictures = images(config)
    lights = probe.to_lights(config, network, column_map, pictures)
    order = sequence(args.cycles, args.order_seed)
    print(f"網: 細胞 {network.n_neurons:,} / 結合 {network.n_edges:,}（{spec.key()}）")
    print(f"光受容細胞への入力: {params.input_max_mv:g} × L / ({params.input_half_saturation:g} + L) mV")
    print(f"動き: {config.vision_motion}（1 観測 {probe.motion_record(config)['frames']} 枚）")
    print(f"順序: 無地 × {WARMUP} → 6 種 × {args.cycles} 巡（巡ごとに入れ替える。種 {args.order_seed}）")
    records = probe.run(network, params, lights, order, "natural", duration_ms, seed=0)

    shown = records[WARMUP:]
    charts = list(CHARTS)
    labels = np.array([charts.index(r["image"]) for r in shown])
    pair_index = [(charts.index(u), charts.index(d)) for u, d in PAIRS]
    names, rates = type_rates(network, shown, seconds)
    active = rates.std(axis=1) > 0
    image_p = np.full(len(names), np.nan)
    image_p[active] = image_test(rates[active], labels, len(charts), PERMUTATION_TRIALS, PERMUTATION_SEED)
    diff = np.full(len(names), np.nan)
    direction_p = np.full(len(names), np.nan)
    diff[active], direction_p[active] = direction_test(rates[active], labels, pair_index,
                                                       PERMUTATION_TRIALS, PERMUTATION_SEED)
    superclass = probe.type_superclass(config)
    position = {str(n): i for i, n in enumerate(names)}

    def describe(i: int) -> dict:
        by_chart = {c: float(rates[i, labels == k].mean()) for k, c in enumerate(charts)}
        return {
            "mean_hz": float(rates[i].mean()),
            "image_p": None if np.isnan(image_p[i]) else float(image_p[i]),
            "up_minus_down_hz": None if np.isnan(diff[i]) else float(diff[i]),
            "direction_p": None if np.isnan(direction_p[i]) else float(direction_p[i]),
            "pair_diffs_hz": {f"{u}−{d}": by_chart[u] - by_chart[d] for u, d in PAIRS},
            "by_chart_hz": by_chart,
        }

    tested = []
    for t in chosen:
        i = position.get(t["type"])
        entry = {"type": t["type"], "superclass": t["superclass"], "discovery_p": t["p"]}
        entry.update(describe(i) if i is not None else {"mean_hz": None, "image_p": None, "direction_p": None})
        tested.append(entry)

    def family(rows: list[dict], key: str) -> dict:
        ps = [r[key] if r[key] is not None else 1.0 for r in rows]
        reject = holm(ps)
        for r, flag in zip(rows, reject):
            r[f"{key}_holm"] = flag
        return {"tested": len(rows), "p_le_0.05": sum(p <= 0.05 for p in ps),
                "expected_by_chance": 0.05 * len(rows), "holm_0.05": sum(reject)}

    beyond = [r for r in tested if r["superclass"] not in OPTIC_LOBE]
    summary = {
        "all": {"image": family(tested, "image_p"), "direction": family(tested, "direction_p")},
    }
    beyond_rows = [dict(r) for r in beyond]
    summary["beyond_optic_lobe"] = {"image": family(beyond_rows, "image_p"),
                                    "direction": family(beyond_rows, "direction_p")}

    # 読み出しの候補（群）。発見の JSON とは独立に、同じ検定を当てる。
    group_map = probe.groups(network)
    group_names = [n for n, g in group_map.items() if g["candidate"] and not n.startswith("  ")]
    group_rates = np.array([[probe.rates(r["counts"], group_map[n], seconds)["all"] for r in shown]
                            for n in group_names])
    g_active = group_rates.std(axis=1) > 0
    g_image = np.full(len(group_names), np.nan)
    g_dir = np.full(len(group_names), np.nan)
    g_diff = np.full(len(group_names), np.nan)
    if g_active.any():
        g_image[g_active] = image_test(group_rates[g_active], labels, len(charts), PERMUTATION_TRIALS,
                                       PERMUTATION_SEED)
        g_diff[g_active], g_dir[g_active] = direction_test(group_rates[g_active], labels, pair_index,
                                                           PERMUTATION_TRIALS, PERMUTATION_SEED)
    candidates = {
        n: {"cells": int(len(group_map[n]["index"])), "mean_hz": float(group_rates[k].mean()),
            "image_p": None if np.isnan(g_image[k]) else float(g_image[k]),
            "up_minus_down_hz": None if np.isnan(g_diff[k]) else float(g_diff[k]),
            "direction_p": None if np.isnan(g_dir[k]) else float(g_dir[k])}
        for k, n in enumerate(group_names)
    }

    # 別の画像の組での全型の数（偶然との比較のため）。
    by_superclass: dict[str, dict[str, int]] = {}
    for i in np.flatnonzero(active):
        key = str(superclass.get(names[i], "untyped") if names[i] else "untyped")
        row = by_superclass.setdefault(key, {"active": 0, "image_p_le_0.001": 0, "direction_p_le_0.001": 0})
        row["active"] += 1
        row["image_p_le_0.001"] += int(image_p[i] <= 0.001)
        row["direction_p_le_0.001"] += int(direction_p[i] <= 0.001)

    def fmt(value, width=7, digits=3):
        return " " * (width - 1) + "-" if value is None else f"{value:{width}.{digits}f}"

    print()
    print("確かめた型のうち p ≦ 0.05 だった数（Holm で補正したあとの数）。偶然でも 5% は出る")
    for scope, label in (("all", "全部"), ("beyond_optic_lobe", "視葉の外")):
        img, drc = summary[scope]["image"], summary[scope]["direction"]
        print(f"  {label:6s} {img['tested']:3d} 型: 画像による差 {img['p_le_0.05']:3d}（Holm {img['holm_0.05']:3d}）"
              f" / 上げと下げの差 {drc['p_le_0.05']:3d}（Holm {drc['holm_0.05']:3d}）"
              f" / 偶然の目安 {img['expected_by_chance']:.1f}")
    print()
    print(f"{'型':14s}{'superclass':20s}{'発見 p':>8s}{'平均 Hz':>9s}{'画像 p':>8s}{'上げ−下げ':>10s}{'p':>8s}  組ごとの差（Hz）")
    for r in tested:
        if r["superclass"] in OPTIC_LOBE:
            continue
        pairs = " ".join(f"{v:+6.2f}" for v in r.get("pair_diffs_hz", {}).values())
        print(f"{r['type']:14s}{r['superclass']:20s}{fmt(r['discovery_p'], 8, 4)}{fmt(r['mean_hz'], 9, 2)}"
              f"{fmt(r['image_p'], 8, 4)}{fmt(r.get('up_minus_down_hz'), 10, 2)}{fmt(r['direction_p'], 8, 4)}  {pairs}")
    print()
    print("読み出しの候補（群）")
    for n, c in candidates.items():
        print(f"  {n:10s}{c['cells']:5d}{fmt(c['mean_hz'], 9, 2)}{fmt(c['image_p'], 8, 4)}"
              f"{fmt(c['up_minus_down_hz'], 10, 2)}{fmt(c['direction_p'], 8, 4)}")
    print()
    trials = int(active.sum())
    print(f"別の画像の組での全型（発火した型 {trials:,}）。偶然でも p ≦ 0.001 は約 {trials * 0.001:.0f} 型出る")
    print(f"  {'superclass':22s}{'発火':>7s}{'画像':>7s}{'上げ下げ':>9s}")
    for key, row in sorted(by_superclass.items(), key=lambda kv: -kv[1]["active"]):
        print(f"  {key:22s}{row['active']:7d}{row['image_p_le_0.001']:7d}{row['direction_p_le_0.001']:9d}")

    started = datetime.now(timezone.utc)
    report = {
        "started_at": started.isoformat(),
        "config": str(args.config) if args.config else "agent.yaml",
        "network_key": spec.key(),
        "params_key": params.key(),
        "photoreceptor_input": {"max_mv": params.input_max_mv, "half_saturation": params.input_half_saturation},
        "discovery": {"path": str(args.discovery), "sha256": hashlib.sha256(discovery_bytes).hexdigest(),
                      "threshold_p": args.discovery_p, "types": [t["type"] for t in chosen]},
        "condition": "natural",
        "sequence": order,
        "order_seed": args.order_seed,
        "pairs": [{"up": u, "down": d} for u, d in PAIRS],
        "duration_ms": duration_ms,
        "permutation": {"trials": PERMUTATION_TRIALS, "seed": PERMUTATION_SEED,
                        "image": "between-group sum of squares", "direction": "|up − down|, within pairs"},
        "vision_motion": probe.motion_record(config),
        "images": {name: [img.sha256 for img in frames] for name, frames in pictures.items()},
        "column_map_sha256": column_map.sha256,
        "wall_sec": [r["wall_sec"] for r in records],
        "total_spikes": [r["total_spikes"] for r in records],
        "summary": summary,
        "tested": tested,
        "candidates": candidates,
        "screen": {"active_types": trials, "by_superclass": by_superclass,
                   "image_p_le_0.001": [str(names[i]) for i in np.flatnonzero(active) if image_p[i] <= 0.001],
                   "direction_p_le_0.001": [str(names[i]) for i in np.flatnonzero(active)
                                            if direction_p[i] <= 0.001]},
        "mean_wall_sec": statistics.fmean(r["wall_sec"] for r in records),
    }
    out = args.out or REPO_ROOT / "var" / "bench" / f"validate-{started:%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"結果: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
