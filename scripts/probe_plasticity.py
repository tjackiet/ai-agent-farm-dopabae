#!/usr/bin/env python3
"""学習の場所（KC→MBON07 / MBON11）を変えたとき、読み出し（DNp20 の左右・DNpe017）が変わるかを測る。

Phase 5 では、決済が確定した回の報酬・嫌悪で KC→MBON07 / MBON11 の重みを変える
（Stonkfly と同じ場所。抑圧のみ）。ところが方向を決める読み出しは、右 DNp20 − 左 DNp20 と
DNpe017 の関門である（`fly.readout`）。**学習の場所の変化が読み出しまで届かなければ、
学習しても方向は変わらない。** Phase 5 の前に、それを確かめる。学習そのものは入れない。
**結論は書かない。数字だけを出す。**

測りかた:

- 網の上の距離: MBON07 / MBON11 から、読み出しの細胞まで何段で届くか（速い作用を持つ結合だけをたどる）
- 重みを人為的に変えて動かす（条件は `CONDITIONS`）。画像は `probe_readout.py` と同じ合成のチャート 5 種で、
  無地で 2 回ならしてから `--cycles` 巡。状態を引き継ぐ。嗅覚の回路の点火は「自然」（何も足さない）。
  この条件は乱数を使わない
- 観測ごとに、読み出しの規則（`fly.read`）で方向を出し、基準（変えない）と比べる。
  **網はわずかな差でも発火がずれる**（Phase 3「光受容細胞 1 個で、KC の発火が 100 倍変わる」）ので、
  ×0.999・×1.001・×0.998・×1.002 の条件を揺らぎの目安として並べる。学習の場所を変えた条件が、
  揺らぎの目安どうしのばらつきの外に出るかを見る

使いかた:

    python3 scripts/probe_plasticity.py --config arms/fly.yaml --out var/bench/plasticity-hs0.02.json
    python3 scripts/probe_plasticity.py --summarize var/bench/plasticity-hs0.02.json

結果は `var/bench/` に JSON で残る（Git 管理外）。**発注しない。取引所にも触れない。**
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dopabae import connectome, fly, lif, retina  # noqa: E402
from dopabae.config import REPO_ROOT, load  # noqa: E402

import probe_readout as probe  # noqa: E402  同じ scripts/ にある

# 学習の場所（Stonkfly と同じ。agent.yaml の fly.reward.plasticity_edges）。
PLASTIC_TYPES = ("MBON07", "MBON11")
KC_PREFIX = "KC"
MBON_PREFIX = "MBON"
# 条件: (説明, 何を変えるか)。重みは KC→（対象の MBON）の結合だけを掛ける。
CONDITIONS = {
    "base": ("変えない（基準）", {}),
    "tiny": ("KC→MBON07/11 を ×0.999（網の揺らぎの目安）", {"scale": 0.999}),
    "tiny2": ("KC→MBON07/11 を ×1.001（網の揺らぎの目安）", {"scale": 1.001}),
    "tiny3": ("KC→MBON07/11 を ×0.998（網の揺らぎの目安）", {"scale": 0.998}),
    "tiny4": ("KC→MBON07/11 を ×1.002（網の揺らぎの目安）", {"scale": 1.002}),
    "half": ("KC→MBON07/11 を ×0.5", {"scale": 0.5}),
    "zero": ("KC→MBON07/11 を ×0（抑圧のみの学習の行き着く先）", {"scale": 0.0}),
    "all_zero": ("KC→すべての MBON を ×0（キノコ体の出力の上限）", {"scale": 0.0, "all_mbon": True}),
    "drive": ("MBON07/11 に +20 mV の一定電流（届くかの陽性対照）", {"current_mv": 20.0}),
    "silence": ("MBON07/11 に −500 mV の一定電流（出力を止める）", {"current_mv": -500.0}),
    # Stonkfly の規則（stonkfly/neural/rule.py）は時間の順序で弱めも強めもし、元の 0.1〜2.0 倍に収める。
    # PAM11 は KC→MBON07 だけ、PPL101 は KC→MBON11 だけを変える（circuit.py の区画の対応）。
    # 学習が届きうる端を、区画ごとに測る。
    "m07_low": ("KC→MBON07 だけ ×0.1（ごほうびの区画の下限）", {"scale_types": {"MBON07": 0.1}}),
    "m07_high": ("KC→MBON07 だけ ×2.0（ごほうびの区画の上限）", {"scale_types": {"MBON07": 2.0}}),
    "m11_low": ("KC→MBON11 だけ ×0.1（罰の区画の下限）", {"scale_types": {"MBON11": 0.1}}),
    "m11_high": ("KC→MBON11 だけ ×2.0（罰の区画の上限）", {"scale_types": {"MBON11": 2.0}}),
}
RECORD = ("MBON07", "MBON11", "lLN1_bc")


def scaled(network: connectome.Network, pre: object, post: object, factor: float) -> tuple[connectome.Network, int]:
    """`pre` の細胞から `post` の細胞への結合だけ、シナプス数に `factor` を掛けた網と、変えた結合の数を返す。元の網は変えない。"""
    import numpy as np

    n = network.n_neurons
    is_pre = np.zeros(n, dtype=bool)
    is_pre[pre] = True
    is_post = np.zeros(n, dtype=bool)
    is_post[post] = True
    edge_pre = np.repeat(np.arange(n, dtype=np.int32), np.diff(network.indptr))
    mask = is_pre[edge_pre] & is_post[network.post]
    counts = network.signed_count.astype(np.float32)
    counts[mask] *= np.float32(factor)
    meta = {**network.meta, "key": f"{network.meta.get('key')}-x{factor:g}-{int(mask.sum())}"}
    return dataclasses.replace(network, signed_count=counts, meta=meta), int(mask.sum())


def hops(network: connectome.Network, sources: object, targets: object, limit: int = 8) -> int | None:
    """`sources` から `targets` のどれかへ、速い作用を持つ結合をたどって何段で届くか。届かなければ None。

    調節性の物質だけを出す細胞（`modulatory`）の発火は行き先の膜電位を動かさないので、そこから先はたどらない。
    """
    import numpy as np

    fast = ~np.asarray(network.modulatory, dtype=bool)
    targets = np.asarray(targets)
    seen = np.zeros(network.n_neurons, dtype=bool)
    frontier = np.unique(np.asarray(sources))
    seen[frontier] = True
    for depth in range(1, limit + 1):
        frontier = frontier[fast[frontier]]
        if not frontier.size:
            return None
        starts = network.indptr[frontier]
        lengths = network.indptr[frontier + 1] - starts
        total = int(lengths.sum())
        if not total:
            return None
        edges = np.repeat(starts - (np.cumsum(lengths) - lengths), lengths) + np.arange(total)
        reached = np.unique(network.post[edges])
        if np.isin(targets, reached).any():
            return depth
        frontier = reached[~seen[reached]]
        seen[frontier] = True
    return None


def run_condition(config, network, params, lights, sequence, duration_ms, spec: dict):
    """1 つの条件で網を動かし、観測ごとの方向と発火を返す。"""
    import numpy as np

    kc = network.type_index(prefix=KC_PREFIX)
    plastic = network.type_index(PLASTIC_TYPES)
    targets = network.type_index(prefix=MBON_PREFIX) if spec.get("all_mbon") else plastic
    net, edges = (network, 0)
    if "scale" in spec:
        net, edges = scaled(network, kc, targets, spec["scale"])
    for name, factor in spec.get("scale_types", {}).items():
        net, changed = scaled(net, kc, network.type_index((name,)), factor)
        edges += changed
    stimulus = lif.Stimulus(index=plastic, current_mv=spec["current_mv"]) if "current_mv" in spec else None
    seconds = duration_ms / 1000.0
    groups = {name: net.type_index((name,)) for name in RECORD}
    state = None
    out = []
    for step, name in enumerate(sequence):
        result = lif.simulate(net, params, duration_ms, light=lights[name][0], stimulus=stimulus, state=state, seed=step)
        state = result.state
        counts = result.spike_counts
        reading = fly.read(config, net, counts, seconds)
        out.append({
            "image": name,
            "direction": reading["value"],
            "axis_hz": reading["axis_hz"],
            "approach_hz": reading["approach_hz"],
            "avoid_hz": reading["avoid_hz"],
            "gate_spikes": reading["gate_spikes"],
            "kc_hz": float(counts[kc].mean()) / seconds,
            **{f"{k}_hz": float(counts[v].mean()) / seconds for k, v in groups.items()},
            "total_spikes": int(counts.sum(dtype="int64")),
            "wall_sec": result.wall_sec,
        })
        print(f"    {step + 1:2d} {name:4s} {reading['value']:8s} 差 {reading['axis_hz']:+6.1f} 関門 {reading['gate_spikes']:3d} "
              f"MBON07 {out[-1]['MBON07_hz']:6.1f} MBON11 {out[-1]['MBON11_hz']:6.1f} {result.wall_sec:5.1f} 秒", flush=True)
    return out, edges


def summarize(report: dict) -> None:
    """条件ごとに、基準との違いを並べる（ならしの観測は除く）。"""
    base = report["conditions"].get("base")
    print(f"\n{report['config']}（半飽和 {report['half_saturation']:g}）。ならし {probe.WARMUP} 回を除く {report['observations']} 観測")
    print(f"{'条件':10s}{'APPROACH':>9s}{'AVOID':>6s}{'NONE':>6s}{'差の平均':>9s}{'関門 開':>8s}"
          f"{'MBON07':>8s}{'MBON11':>8s}{'KC':>6s}{'基準と同じ方向':>14s}{'|差の差|':>9s}")
    for name, cond in report["conditions"].items():
        obs = cond["observations"][probe.WARMUP:]
        counts = Counter(o["direction"] for o in obs)
        same = diffs = None
        if base is not None:
            ref = base["observations"][probe.WARMUP:]
            same = sum(a["direction"] == b["direction"] for a, b in zip(obs, ref)) / len(obs)
            diffs = statistics.fmean(abs(a["axis_hz"] - b["axis_hz"]) for a, b in zip(obs, ref))
        print(f"{name:10s}{counts.get('APPROACH', 0):9d}{counts.get('AVOID', 0):6d}{counts.get('NONE', 0):6d}"
              f"{statistics.fmean(o['axis_hz'] for o in obs):+9.2f}"
              f"{sum((o['gate_spikes'] or 0) >= 1 for o in obs) / len(obs):8.0%}"
              f"{statistics.fmean(o['MBON07_hz'] for o in obs):8.1f}{statistics.fmean(o['MBON11_hz'] for o in obs):8.1f}"
              f"{statistics.fmean(o['kc_hz'] for o in obs):6.2f}"
              f"{'' if same is None else f'{same:.0%}':>14s}{'' if diffs is None else f'{diffs:.2f}':>9s}")
    print("距離（段）:", report["hops"])


NULL_PREFIX = "tiny"
JUDGED = ("APPROACH", "AVOID", "NONE", "gate_open_pct", "axis_mean_hz")


def condition_stats(observations: list[dict], base: list[dict]) -> dict:
    """ならしを除いた観測の数字。基準と同じ方向だった割合も出す。"""
    counts = Counter(o["direction"] for o in observations)
    n = len(observations)
    return {
        "APPROACH": counts.get("APPROACH", 0),
        "AVOID": counts.get("AVOID", 0),
        "NONE": counts.get("NONE", 0),
        "gate_open_pct": round(100 * sum((o["gate_spikes"] or 0) >= 1 for o in observations) / n),
        "axis_mean_hz": round(statistics.fmean(o["axis_hz"] for o in observations), 2),
        "same_as_base_pct": round(100 * sum(a["direction"] == b["direction"] for a, b in zip(observations, base)) / n),
        "MBON07_hz": round(statistics.fmean(o["MBON07_hz"] for o in observations), 1),
        "MBON11_hz": round(statistics.fmean(o["MBON11_hz"] for o in observations), 1),
    }


def compare_to_null(reports: list[dict]) -> dict:
    """複数の結果をまとめ、揺らぎの目安（基準と tiny*）の幅の外に出た数字を挙げる。

    基準は結果どうしで同じでなければならない（乱数を使わないので、計算時間を除けば一致する）。
    幅は最小〜最大で、標本は少ない。外に出たかどうかは目安である。
    """
    strip = lambda obs: [{k: v for k, v in o.items() if k != "wall_sec"} for o in obs]  # noqa: E731
    bases = [strip(r["conditions"]["base"]["observations"]) for r in reports if "base" in r["conditions"]]
    if not bases or any(b != bases[0] for b in bases):
        raise SystemExit("基準（base）が無いか、結果どうしで一致しません")
    base = bases[0][probe.WARMUP:]
    merged = {}
    for report in reports:
        for name, cond in report["conditions"].items():
            merged.setdefault(name, condition_stats(cond["observations"][probe.WARMUP:], base))
    nulls = [k for k in merged if k == "base" or k.startswith(NULL_PREFIX)]
    span = {x: (min(merged[k][x] for k in nulls), max(merged[k][x] for k in nulls)) for x in JUDGED}
    tiny_same = [merged[k]["same_as_base_pct"] for k in nulls if k != "base"]
    outside = {k: [x for x in JUDGED if not span[x][0] <= v[x] <= span[x][1]]
               for k, v in merged.items() if k not in nulls}
    return {"conditions": merged, "null": nulls, "null_span": span,
            "null_same_as_base_pct": (min(tiny_same), max(tiny_same)) if tiny_same else None, "outside": outside}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="使う設定（腕）。既定は agent.yaml")
    parser.add_argument("--conditions", default=",".join(CONDITIONS), help="回す条件（カンマ区切り）")
    parser.add_argument("--cycles", type=int, default=4, help="5 種を何巡見せるか（既定 4）")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--summarize", nargs="+", type=Path, default=None, help="結果の JSON を読んで比べるだけ")
    args = parser.parse_args(argv)
    if args.summarize:
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.summarize]
        for report in reports:
            summarize(report)
        if len(reports) > 1:
            result = compare_to_null(reports)
            print("\n揺らぎの目安:", ", ".join(result["null"]), "の幅（最小〜最大）:", result["null_span"],
                  "／基準と同じ方向（tiny*）:", result["null_same_as_base_pct"])
            for name, keys in result["outside"].items():
                stats = result["conditions"][name]
                print(f"  {name}: 幅の外 {keys or 'なし'}／基準と同じ方向 {stats['same_as_base_pct']}%")
        return 0
    names = [n.strip() for n in args.conditions.split(",") if n.strip()]
    unknown = [n for n in names if n not in CONDITIONS]
    if unknown:
        parser.error(f"知らない条件です: {unknown}（{', '.join(CONDITIONS)}）")

    config = load(args.config)
    params = lif.Params.from_settings(config.simulation)
    duration_ms = config.simulation.neural_time_sec * 1000.0
    spec = connectome.spec_from_config(config)
    network, _ = connectome.load_or_compile(config, spec)
    column_map = retina.load_column_map(REPO_ROOT / config.retina_column_map_path)
    pictures = probe.images(config)
    lights = probe.to_lights(config, network, column_map, pictures)
    sequence = [probe.BLANK] * probe.WARMUP + list(probe.CHARTS) * args.cycles

    plastic = network.type_index(PLASTIC_TYPES)
    readout = {
        "DNp20:R": fly.select(network, ("DNp20:R",)),
        "DNp20:L": fly.select(network, ("DNp20:L",)),
        "DNpe017": fly.select(network, ("DNpe017",)),
    }
    distance = {name: hops(network, plastic, cells) for name, cells in readout.items()}
    print(f"MBON07 / MBON11 から読み出しの細胞まで（段）: {distance}")

    conditions = {}
    for name in names:
        label, cond = CONDITIONS[name]
        print(f"  {name}: {label}")
        observations, edges = run_condition(config, network, params, lights, sequence, duration_ms, cond)
        conditions[name] = {"label": label, "spec": cond, "edges_changed": edges, "observations": observations}

    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config) if args.config else "agent.yaml",
        "half_saturation": params.input_half_saturation,
        "network_key": spec.key(),
        "params_key": params.key(),
        "sequence": sequence,
        "observations": len(sequence) - probe.WARMUP,
        "hops": distance,
        "readout": {"approach": config.readout_approach, "avoid": config.readout_avoid,
                    "thresholds_hz": [config.readout_approach_threshold_hz, config.readout_avoid_threshold_hz],
                    "gate": config.readout_gate, "gate_min_spikes": config.readout_gate_min_spikes},
        "conditions": conditions,
    }
    out = args.out or REPO_ROOT / "var" / "bench" / f"plasticity-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summarize(report)
    print(f"\n結果: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
