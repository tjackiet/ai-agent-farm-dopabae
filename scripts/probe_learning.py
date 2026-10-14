#!/usr/bin/env python3
"""学習（Phase 5）を入れた網に報酬・嫌悪の刺激を与え、つながりの強さと MBON の発火がどう動くかを測る。

`fly.decide` をそのまま通す（刺激の入れかた・規則・保存は本番と同じ）。網の状態と学習の状態は
一時ディレクトリに置き、腕の状態には触らない。**発注しない。取引所にも触れない。結論は書かない。**

学習の値のうち人間が決めていないもの（報酬の閾値、記憶の時定数、いつ学ぶか）は、引数で仮に与える。
刺激は決済の損益からではなく、`--pulses` の観測で直接与える（規則の応答を見るため）。

画像は `probe_readout.py` と同じ合成のチャート。無地で 2 回ならしてから、5 種を順に見せる。

使いかた:

    python3 scripts/probe_learning.py --config arms/fly.yaml --stimulus reward --pulses 4,6,8 \\
        --learn-when stimulus_only --observations 14 --out var/bench/learning-hs0.02-reward.json

結果は `var/bench/` に JSON で残る（Git 管理外）。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dopabae import connectome, fly, retina  # noqa: E402
from dopabae.config import LEARN_WHEN, REPO_ROOT, load  # noqa: E402

import probe_readout as probe  # noqa: E402  同じ scripts/ にある

RECORD = ("lLN1_bc", "MBON07", "MBON11", "PAM11", "PPL101")


def sequence(pictures: dict, observations: int) -> list[str]:
    charts = [name for name in pictures if name != probe.BLANK]
    return [probe.BLANK if i < probe.WARMUP else charts[(i - probe.WARMUP) % len(charts)] for i in range(observations)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="学習を入れた網に刺激を与えて、重みと発火の動きを測る")
    parser.add_argument("--config", default=str(REPO_ROOT / "agent.yaml"))
    parser.add_argument("--stimulus", choices=("reward", "aversive"), required=True)
    parser.add_argument("--pulses", required=True, help="刺激を与える観測の番号（0 から。カンマ区切り）")
    parser.add_argument("--observations", type=int, default=14)
    parser.add_argument("--learn-when", choices=LEARN_WHEN, required=True)
    parser.add_argument("--memory-decay-sec", type=float, default=1800.0, help="記憶の時定数（神経時間。仮の値）")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    pulses = {int(x) for x in args.pulses.split(",") if x}
    cfg = load(args.config)
    cfg = dataclasses.replace(
        cfg,
        direction_source="fly",
        readout_record=RECORD,
        reward=dataclasses.replace(
            cfg.reward,
            enabled=True,
            threshold_jpy=cfg.reward.threshold_jpy or 1.0,  # 刺激は直接与えるので使わない
            memory_decay_sec=args.memory_decay_sec,
            learn_when=args.learn_when,
            weight_snapshot_every_n=10**9,
        ),
    )
    network, _ = connectome.load_or_compile(cfg, connectome.spec_from_config(cfg))
    column_map = retina.load_column_map(REPO_ROOT / cfg.retina_column_map_path)
    pictures = probe.images(cfg)
    order = sequence(pictures, args.observations)
    root = Path(tempfile.mkdtemp(prefix="probe-learning-"))

    rows = []
    for step, name in enumerate(order):
        stimulus = args.stimulus if step in pulses else "none"
        result = fly.decide(cfg, pictures[name], stimulus=stimulus, network=network, column_map=column_map, repo_root=root)
        obs = result.observation
        learning = obs["learning"]
        row = {
            "step": step,
            "image": name,
            "stimulus": stimulus,
            "direction": result.value,
            "axis_hz": obs["axis_hz"],
            "gate_spikes": obs["gate_spikes"],
            **{f"{k}_hz": v for k, v in obs["record_hz"].items()},
            "plastic": learning["plastic"],
            "changed_edges": learning["changed_edges"],
            "by_mbon": learning["by_mbon"],
            "wall_sec": obs["wall_sec"],
        }
        rows.append(row)
        effs = "  ".join(f"{k} {v['mean_efficacy']:.4f}（{v['min_efficacy']:.3f}〜{v['max_efficacy']:.3f}）" for k, v in learning["by_mbon"].items())
        print(f"{step:2d} {name:4s} {stimulus:8s} {result.value:8s} 差 {obs['axis_hz']:+6.1f} 関門 {obs['gate_spikes']:3d} "
              f"MBON07 {row['MBON07_hz']:5.1f} MBON11 {row['MBON11_hz']:5.1f} PAM11 {row['PAM11_hz']:5.1f} PPL101 {row['PPL101_hz']:5.1f}  {effs}",
              flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": args.config,
            "half_saturation": cfg.simulation.input_half_saturation,
            "stimulus": args.stimulus,
            "pulses": sorted(pulses),
            "learn_when": args.learn_when,
            "memory_decay_sec": args.memory_decay_sec,
            "network_key": network.meta.get("key"),
            "observations": rows,
        }
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
