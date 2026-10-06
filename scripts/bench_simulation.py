#!/usr/bin/env python3
"""全脳の LIF シミュレーションの計算時間とメモリを測る（前提 1・2・7）。

`docs/IMPLEMENTATION_PLAN.md` 3 節の未確認の前提のうち、
「全脳・1 観測（神経時間 0.5 秒）の計算時間」と「定期実行の環境で回るか」を、
**この計算機で実際に回して**測る。結論は書かない。数字だけを出す。

測るもの:

- 網の準備: flat-connectome からの変換（初回だけ）と、キャッシュからの読み込み
- 観測ごと（`fly.simulation.neural_time_per_observation_sec`）のシミュレーション。
  状態を引き継いで `--observations` 回続ける（定期実行と同じ）
- プロセスの最大メモリ

入力は `agent.yaml` の値どおり（Stonkfly と同じ。`docs/CONNECTOME_SURVEY.md` 1.1）。

- 画像: 決定的に作った合成の 15 分足を `vision.py` で描き、`retina.py` で光受容細胞へ
  落とし、明るさから作った電流にする
- 背景の活動: ラミナへの一定電流（`fly.simulation.lamina`）
- 負荷試験（`--background-hz`）: 全細胞へ同じ発火率のポアソン入力を足す。
  網がもっと活動したときの計算量を見るためで、**モデルの選択ではない**

使いかた:

    python3 scripts/bench_simulation.py                       # 閾値なし・負荷なし・2 観測
    python3 scripts/bench_simulation.py --threshold none --threshold 3 --threshold 5 \\
        --background-hz 0 --background-hz 20

結果は `var/bench/` に JSON で残る（Git 管理外）。
**発注しない。取引所にも触れない。** 配線図データだけを使う。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dopabae import connectome, lif, retina, vision  # noqa: E402
from dopabae.config import REPO_ROOT, Config, load  # noqa: E402

# 負荷試験のポアソン入力 1 発の重み（シナプス何個ぶんか）。Shiu ら (2024) の 250。
DEFAULT_BACKGROUND_WEIGHT_SYNAPSES = 250.0

# 観測ごとに数える細胞の群れ。網の下流まで届いたかを見るため。値の解釈はしない。
GROUPS = {
    "photoreceptors": {"names": ("R1-R6", "R8p", "R8y")},
    "lamina": {"from_config": "lamina"},
    "kc": {"prefix": "KC"},
    "mbon": {"prefix": "MBON"},
}


def synthetic_candles(count: int, now_ms: int) -> tuple[vision.Candle, ...]:
    """決定的な合成の足。上げと下げが交互に混ざる。相場の観測ではない。"""
    step = 15 * 60_000
    candles = []
    close = 14_700_000
    for i in range(count):
        drift = 30_000 if i % 3 else -45_000
        open_ = close
        close = open_ + drift
        candles.append(
            vision.Candle(
                timestamp_ms=now_ms - (count - 1 - i) * step,
                open=Decimal(open_),
                high=Decimal(max(open_, close) + 12_000),
                low=Decimal(min(open_, close) - 12_000),
                close=Decimal(close),
            )
        )
    return tuple(candles)


def image_light(config: Config, network: connectome.Network) -> tuple[lif.Light, dict]:
    """合成の画像から光受容細胞の明るさを作る。網に無い細胞は数えて落とす。"""
    now_ms = 1_789_430_400_000  # 固定。描く絵を毎回同じにする
    candles = synthetic_candles(config.vision_lookback_candles, now_ms)
    last = candles[-1].close
    image = vision.render(config, candles, last - 500, last + 500, config.pair)
    column_map = retina.load_column_map(REPO_ROOT / config.retina_column_map_path)
    activations = retina.activations(config, image, column_map)

    light, missing = lif.light_from_values(network, activations.currents)
    info = {
        "image_sha256": image.sha256,
        "column_map_sha256": column_map.sha256,
        "retina": activations.as_dict(),
        "receptors_in_network": int(len(light.index)),
        "receptors_not_in_network": missing,
        "mean_light": float(light.value.mean()) if len(light.index) else 0.0,
    }
    return light, info


def background_drive(network: connectome.Network, hz: float, weight_mv: float) -> lif.Drive | None:
    """全細胞へ同じ発火率のポアソン入力（負荷試験）。"""
    import numpy as np

    if hz <= 0:
        return None
    return lif.Drive(
        index=np.arange(network.n_neurons),
        rate_hz=np.full(network.n_neurons, hz, dtype=np.float64),
        weight_mv=weight_mv,
    )


def group_indices(config: Config, network: connectome.Network) -> dict:
    out = {}
    for name, rule in GROUPS.items():
        if rule.get("from_config") == "lamina":
            out[name] = network.type_index(tuple(config.simulation.lamina_types))
        else:
            out[name] = network.type_index(rule.get("names", ()), rule.get("prefix"))
    return out


def peak_rss_mb() -> float:
    """プロセスの最大メモリ。Linux は KB、macOS はバイトで返る。"""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def machine() -> dict:
    import numpy as np

    try:
        memory_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        memory_gb = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "memory_gb": None if memory_gb is None else round(memory_gb, 1),
    }


def parse_threshold(text: str) -> int | None:
    if text.lower() in ("none", "null", "0"):
        return None
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("閾値は 1 以上か none")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold", type=parse_threshold, action="append",
                        help="シナプス数の閾値（none / 整数）。複数指定可。既定は agent.yaml の値")
    parser.add_argument("--background-hz", type=float, action="append",
                        help="全細胞へのポアソン入力（負荷試験）。複数指定可。既定は 0 だけ")
    parser.add_argument("--background-weight-synapses", type=float,
                        default=DEFAULT_BACKGROUND_WEIGHT_SYNAPSES)
    parser.add_argument("--observations", type=int, default=2,
                        help="状態を引き継いで続ける観測の回数（既定 2）")
    parser.add_argument("--duration-ms", type=float, default=None,
                        help="1 観測の神経時間。既定は fly.simulation.neural_time_per_observation_sec")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--out", type=Path, default=None, help="結果の JSON（既定は var/bench/）")
    args = parser.parse_args(argv)
    if args.observations < 1:
        parser.error("--observations は 1 以上")

    config = load()
    thresholds = args.threshold or [config.connectome_synapse_threshold]
    backgrounds = args.background_hz or [0.0]
    duration_ms = args.duration_ms or config.simulation.neural_time_sec * 1000.0
    params = lif.Params.from_settings(config.simulation)
    background_weight_mv = args.background_weight_synapses * params.weight_per_synapse_mv

    started_at = datetime.now(timezone.utc)
    report: dict = {
        "started_at": started_at.isoformat(),
        "machine": machine(),
        "duration_ms": duration_ms,
        "observations": args.observations,
        "params": asdict(params),
        "background": {"weight_mv": background_weight_mv, "seed": args.seed},
        "runs": [],
    }
    print(f"計算機: {report['machine']}")
    print(f"神経時間: 1 観測 {duration_ms:g} ms / 刻み {params.dt_ms} ms"
          f"（{params.steps_for(duration_ms)} 刻み）× {args.observations} 観測")

    for threshold in thresholds:
        spec = connectome.spec_from_config(config, threshold)
        try:
            began = time.perf_counter()
            network, built = connectome.load_or_compile(config, spec)
            prepared = time.perf_counter() - began
            began = time.perf_counter()
            connectome.load(connectome.compiled_path(config, spec), spec)
            reload_sec = time.perf_counter() - began
            light, image_info = image_light(config, network)
        except (connectome.ConnectomeError, retina.RetinaError, vision.VisionError) as exc:
            print(f"準備できません: {exc}", file=sys.stderr)
            return 1
        groups = group_indices(config, network)

        label = "なし" if threshold is None else f"{threshold} 以上"
        print()
        print(f"閾値 {label}: 細胞 {network.n_neurons:,} / 結合 {network.n_edges:,} / "
              f"シナプス {network.meta['n_synapses']:,}")
        print(f"  網の準備 {prepared:.1f} 秒（{'変換した' if built else 'キャッシュ'}）"
              f" / キャッシュの読み込み {reload_sec:.2f} 秒")
        print(f"  画像の入力: 受容細胞 {image_info['receptors_in_network']:,}"
              f"（平均の明るさ {image_info['mean_light']:.3f}）")

        for background in backgrounds:
            drive = background_drive(network, background, background_weight_mv)
            state = None
            for observation in range(args.observations):
                result = lif.simulate(network, params, duration_ms, light=light, drive=drive,
                                      state=state, seed=args.seed + observation)
                state = result.state
                mean_rate = result.total_spikes / network.n_neurons / (result.duration_ms / 1000.0)
                by_group = {k: int(result.spike_counts[v].sum()) for k, v in groups.items()}
                print(f"  背景 {background:g} Hz・観測 {observation + 1}: {result.wall_sec:.1f} 秒"
                      f" / 発火 {result.total_spikes:,}（平均 {mean_rate:.1f} Hz）"
                      f" / 発火した細胞 {result.active_neurons:,} / KC {by_group['kc']:,}")
                report["runs"].append({
                    "synapse_threshold": threshold,
                    "network": {k: v for k, v in network.meta.items() if k != "spec"},
                    "compiled": built,
                    "prepare_sec": round(prepared, 3),
                    "reload_sec": round(reload_sec, 3),
                    "image": image_info,
                    "background_hz": background,
                    "observation": observation + 1,
                    "wall_sec": round(result.wall_sec, 3),
                    "total_spikes": result.total_spikes,
                    "input_spikes": result.input_spikes,
                    "active_neurons": result.active_neurons,
                    "mean_rate_hz": round(mean_rate, 3),
                    "spikes_by_group": by_group,
                })

    report["peak_rss_mb"] = round(peak_rss_mb(), 1)
    report["max_runtime_sec"] = config.max_runtime_sec
    print()
    print(f"最大メモリ: {report['peak_rss_mb']:,.0f} MB（変換した回はそのぶん大きい）")
    print(f"1 回の判断の上限（runtime.max_runtime_sec）: {config.max_runtime_sec} 秒")

    out = args.out or REPO_ROOT / "var" / "bench" / f"simulation-{started_at:%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"結果: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
