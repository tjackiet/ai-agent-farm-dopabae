"""ハエ版の方向の出どころ（Phase 3）。

画像（1 観測のあいだに見せる枚）を光受容細胞の明るさに直し、配線図の網を LIF で
神経時間だけ動かし、読み出しの細胞群の発火から方向を決める。

- **網は売買を知らない。** 渡すのは画像だけで、残高・損益・保有量は渡さない
- 読み出しは設計者が決めた読み替え（`fly.readout`）。2026-10-12 に人間が Stonkfly と同じもの
  （右 DNp20 − 左 DNp20、DNpe017 が条件）に決めた。「ハエの脳に売買を判断する細胞がある」
  という意味ではない
- 網の状態は観測をまたいで引き継ぐ（`fly.simulation.carry_state`）。状態のファイルが無ければ
  静止状態から始め、あっても今の網・値と合わなければ**読まずに失敗にする**（別の脳の状態を
  引き継がない。消すかどうかは人間が決める）
- 失敗（網・座標表・状態が読めない、画像が無い、シミュレーションの失敗）はすべて例外にする。
  呼び出し側（`direction.resolve`）が方向の失敗として HOLD にする

判断ログに残すのは観測した値（発火数・発火率・差・関門の発火数）と、どの規則を当てたかだけ。
解釈は書かない（CLAUDE.md）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from . import connectome, lif, retina
from .config import REPO_ROOT, Config
from .direction import APPROACH, AVOID, NONE, Direction
from .vision import Image


class FlyError(Exception):
    """ハエ版が方向を返せない。呼び出し側が HOLD にする。"""


def select(network: connectome.Network, selectors: Sequence[str]) -> Any:
    """「型名」「型名:L」「型名:R」の並びから、網の細胞番号を返す。1 個も無ければ FlyError。"""
    np = connectome._numpy()
    found = []
    for item in selectors:
        name, _, side = item.partition(":")
        index = network.type_index((name,))
        if side:
            index = index[network.side[index] == side]
        if not len(index):
            raise FlyError(f"読み出しの細胞群が網にありません: {item}")
        found.append(index)
    return np.unique(np.concatenate(found))


def read(config: Config, network: connectome.Network, counts: Any, seconds: float) -> dict:
    """発火数から、接近と回避の指標・差・関門・方向を出す。値は観測から計算するだけ。"""
    approach = select(network, config.readout_approach)
    avoid = select(network, config.readout_avoid)
    approach_hz = float(counts[approach].mean()) / seconds
    avoid_hz = float(counts[avoid].mean()) / seconds
    axis_hz = approach_hz - avoid_hz
    gate_spikes = None
    if config.readout_gate:
        gate_spikes = int(counts[select(network, config.readout_gate)].sum())
    up, down = config.readout_approach_threshold_hz, config.readout_avoid_threshold_hz
    if gate_spikes is not None and gate_spikes < config.readout_gate_min_spikes:
        value, rule = NONE, f"関門 {gate_spikes} 回 < {config.readout_gate_min_spikes} 回"
    elif axis_hz >= up:
        value, rule = APPROACH, f"差 {axis_hz:+.2f} Hz ≥ {up:+g} Hz"
    elif axis_hz <= down:
        value, rule = AVOID, f"差 {axis_hz:+.2f} Hz ≤ {down:+g} Hz"
    else:
        value, rule = NONE, f"{down:+g} Hz < 差 {axis_hz:+.2f} Hz < {up:+g} Hz"
    # 判断には使わない。発火率を記録するだけ（点火の有無などを、評価で分けて見るため）。
    # 網に無い型は null（観測していない値を作らない）。記録が取れないだけで方向は止めない。
    record_hz = None
    if config.readout_record:
        record_hz = {}
        for item in config.readout_record:
            try:
                record_hz[item] = float(counts[select(network, (item,))].mean()) / seconds
            except FlyError:
                record_hz[item] = None
    return {
        "value": value,
        "rule": rule,
        "record_hz": record_hz,
        "approach_hz": approach_hz,
        "avoid_hz": avoid_hz,
        "axis_hz": axis_hz,
        "gate_spikes": gate_spikes,
        "cells": {
            "approach": [int(b) for b in network.body_ids[approach]],
            "avoid": [int(b) for b in network.body_ids[avoid]],
            "gate": [int(b) for b in network.body_ids[select(network, config.readout_gate)]] if config.readout_gate else None,
        },
    }


def decide(
    config: Config,
    frames: Sequence[Image] | None,
    *,
    network: connectome.Network | None = None,
    column_map: retina.ColumnMap | None = None,
    repo_root: Path | None = None,
) -> Direction:
    """画像の枚を見せて網を動かし、方向を返す。`network` と `column_map` は差し替え口（テスト用）。"""
    if not frames:
        raise FlyError("ハエに見せる画像がありません")
    if config.readout_approach is None or config.readout_avoid is None:
        raise FlyError("fly.readout が決まっていません")
    root = repo_root if repo_root is not None else REPO_ROOT
    spec = connectome.spec_from_config(config)
    if network is None:
        network, _ = connectome.load_or_compile(config, spec)
    if column_map is None:
        column_map = retina.load_column_map(root / config.retina_column_map_path)
    params = lif.Params.from_settings(config.simulation)

    lights = []
    outside = 0
    for image in frames:
        light, outside = lif.light_from_values(network, retina.activations(config, image, column_map).currents)
        if not len(light.index):
            raise FlyError("光受容細胞に値が 1 つも入りませんでした")
        lights.append(light)

    sim = config.simulation
    state_path = root / sim.checkpoint_path
    state = None
    state_origin = "rest"
    if sim.carry_state and state_path.exists():
        state = lif.load_state(state_path, network, params)  # 合わなければ SimulationError（HOLD）
        state_origin = "carried"

    duration_ms = sim.neural_time_sec * 1000.0
    counts = None
    wall = 0.0
    for light in lights:
        result = lif.simulate(network, params, duration_ms / len(lights), light=light, state=state)
        state = result.state
        counts = result.spike_counts if counts is None else counts + result.spike_counts
        wall += result.wall_sec
    state_sha256 = lif.save_state(state, state_path, network, params) if sim.carry_state else None

    reading = read(config, network, counts, duration_ms / 1000.0)
    observation = {
        "approach_hz": reading["approach_hz"],
        "avoid_hz": reading["avoid_hz"],
        "axis_hz": reading["axis_hz"],
        "gate_spikes": reading["gate_spikes"],
        "rule": reading["rule"],
        "record_hz": reading["record_hz"],
        "cells": reading["cells"],
        "total_spikes": int(counts.sum(dtype="int64")),
        "active_neurons": int((counts > 0).sum()),
        "frames": len(lights),
        "photoreceptors_lit": int(len(lights[-1].index)),
        "photoreceptors_outside_network": int(outside),  # 値はあるが網に無い細胞（値を作らずに落とした数）
        "neural_time_ms": duration_ms,
        "wall_sec": round(wall, 3),
        "state_from": state_origin,
        "state_sha256": state_sha256,
        "network_key": spec.key(),
        "params_key": params.key(),
        "column_map_sha256": column_map.sha256,
    }
    reason = (
        f"接近の指標 {reading['approach_hz']:.2f} Hz − 回避の指標 {reading['avoid_hz']:.2f} Hz"
        f"（関門 {reading['gate_spikes']} 回）→ {reading['rule']}"
    )
    return Direction(source="fly", value=reading["value"], reason=reason, observation=observation)
