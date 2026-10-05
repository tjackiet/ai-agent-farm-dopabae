#!/usr/bin/env python3
"""全脳の LIF シミュレーションの計算時間とメモリを測る（前提 1・2・7）。

`docs/IMPLEMENTATION_PLAN.md` 3 節の未確認の前提のうち、
「全脳・1 観測（神経時間 0.5 秒）の計算時間」と「定期実行の環境で回るか」を、
**この計算機で実際に回して**測る。結論は書かない。数字だけを出す。

測るもの:

- 網の準備: flat-connectome からの変換（初回だけ）と、キャッシュからの読み込み
- 1 観測ぶん（`fly.simulation.neural_time_per_observation_sec`）のシミュレーション
- プロセスの最大メモリ

入力は 2 種類。

- 画像: 決定的に作った合成の 15 分足を `vision.py` で描き、`retina.py` で光受容細胞へ
  落とし、明るさ × `--input-max-hz` のポアソン発火にする（Shiu ら 2024 の 150 Hz）
- 背景（`--background-hz`）: 全細胞へ同じ発火率のポアソン入力を足す。**負荷試験である。**
  網が活動したときの計算量を見るためで、モデルの選択ではない

使いかた:

    python3 scripts/bench_simulation.py                       # 閾値なし・画像だけ
    python3 scripts/bench_simulation.py --threshold none --threshold 3 --threshold 5 \\
        --background-hz 0 --background-hz 5 --background-hz 20

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

# Shiu ら (2024) の入力。150 Hz のポアソン発火を、シナプス 250 個ぶんの重みで与える。
# 画像の明るさ（0〜1）から発火率への直しかたは Phase 3 で決める（いまは計測のための仮置き）。
DEFAULT_INPUT_MAX_HZ = 150.0
DEFAULT_INPUT_WEIGHT_SYNAPSES = 250.0


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


def image_drive(config: Config, network: connectome.Network, max_hz: float, weight_mv: float):
    """合成の画像から光受容細胞への入力を作る。網に無い細胞は数えて落とす。"""
    import numpy as np

    now_ms = 1_789_430_400_000  # 固定。描く絵を毎回同じにする
    candles = synthetic_candles(config.vision_lookback_candles, now_ms)
    last = candles[-1].close
    image = vision.render(config, candles, last - 500, last + 500, config.pair)
    column_map = retina.load_column_map(REPO_ROOT / config.retina_column_map_path)
    activations = retina.activations(config, image, column_map)

    body_ids = np.array(sorted(activations.currents), dtype=np.int64)
    values = np.array([activations.currents[b] for b in body_ids], dtype=np.float64)
    index = network.index_of(body_ids)
    inside = index >= 0
    drive = lif.Drive(index=index[inside], rate_hz=values[inside] * max_hz, weight_mv=weight_mv)
    info = {
        "image_sha256": image.sha256,
        "column_map_sha256": column_map.sha256,
        "receptors": int(len(body_ids)),
        "receptors_in_network": int(inside.sum()),
        "mean_rate_hz": float(drive.rate_hz.mean()) if inside.any() else 0.0,
    }
    return drive, info


def with_background(network: connectome.Network, drive: lif.Drive, background_hz: float) -> lif.Drive:
    """全細胞へ同じ発火率を足す（負荷試験）。画像の入力を受ける細胞は足し合わせる。"""
    import numpy as np

    if background_hz <= 0:
        return drive
    rate = np.full(network.n_neurons, background_hz, dtype=np.float64)
    np.add.at(rate, drive.index, drive.rate_hz)
    return lif.Drive(index=np.arange(network.n_neurons), rate_hz=rate, weight_mv=drive.weight_mv)


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
    parser.add_argument("--duration-ms", type=float, default=None,
                        help="神経時間。既定は fly.simulation.neural_time_per_observation_sec")
    parser.add_argument("--input-max-hz", type=float, default=DEFAULT_INPUT_MAX_HZ)
    parser.add_argument("--input-weight-synapses", type=float, default=DEFAULT_INPUT_WEIGHT_SYNAPSES)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--out", type=Path, default=None, help="結果の JSON（既定は var/bench/）")
    args = parser.parse_args(argv)

    config = load()
    thresholds = args.threshold or [config.connectome_synapse_threshold]
    backgrounds = args.background_hz or [0.0]
    duration_ms = args.duration_ms or config.simulation.neural_time_sec * 1000.0
    params = lif.Params.from_settings(config.simulation)
    weight_mv = args.input_weight_synapses * params.weight_per_synapse_mv

    started_at = datetime.now(timezone.utc)
    report: dict = {
        "started_at": started_at.isoformat(),
        "machine": machine(),
        "duration_ms": duration_ms,
        "params": asdict(params),
        "input": {"max_hz": args.input_max_hz, "weight_mv": weight_mv, "seed": args.seed},
        "runs": [],
    }
    print(f"計算機: {report['machine']}")
    print(f"神経時間: {duration_ms:g} ms / 刻み {params.dt_ms} ms（{params.steps_for(duration_ms)} 刻み）")

    for threshold in thresholds:
        spec = connectome.spec_from_config(config, threshold)
        try:
            began = time.perf_counter()
            network, built = connectome.load_or_compile(config, spec)
            prepared = time.perf_counter() - began
            began = time.perf_counter()
            connectome.load(connectome.compiled_path(config, spec), spec)
            reload_sec = time.perf_counter() - began
            drive, image_info = image_drive(config, network, args.input_max_hz, weight_mv)
        except (connectome.ConnectomeError, retina.RetinaError, vision.VisionError) as exc:
            print(f"準備できません: {exc}", file=sys.stderr)
            return 1

        label = "なし" if threshold is None else f"{threshold} 以上"
        print()
        print(f"閾値 {label}: 細胞 {network.n_neurons:,} / 結合 {network.n_edges:,} / "
              f"シナプス {network.meta['n_synapses']:,}")
        print(f"  網の準備 {prepared:.1f} 秒（{'変換した' if built else 'キャッシュ'}）"
              f" / キャッシュの読み込み {reload_sec:.2f} 秒")
        print(f"  画像の入力: 受容細胞 {image_info['receptors_in_network']:,} / {image_info['receptors']:,}"
              f"（平均 {image_info['mean_rate_hz']:.1f} Hz）")

        for background in backgrounds:
            result = lif.simulate(network, params, duration_ms,
                                  with_background(network, drive, background), seed=args.seed)
            mean_rate = result.total_spikes / network.n_neurons / (result.duration_ms / 1000.0)
            print(f"  背景 {background:g} Hz: {result.wall_sec:.1f} 秒 / 発火 {result.total_spikes:,}"
                  f"（平均 {mean_rate:.1f} Hz）/ 発火した細胞 {result.active_neurons:,}")
            report["runs"].append({
                "synapse_threshold": threshold,
                "network": {k: v for k, v in network.meta.items() if k != "spec"},
                "compiled": built,
                "prepare_sec": round(prepared, 3),
                "reload_sec": round(reload_sec, 3),
                "image": image_info,
                "background_hz": background,
                "wall_sec": round(result.wall_sec, 3),
                "total_spikes": result.total_spikes,
                "input_spikes": result.input_spikes,
                "active_neurons": result.active_neurons,
                "mean_rate_hz": round(mean_rate, 3),
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
