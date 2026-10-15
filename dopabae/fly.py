"""ハエ版の方向の出どころ（Phase 3）。

画像（1 観測のあいだに見せる枚）を光受容細胞の明るさに直し、配線図の網を LIF で
神経時間だけ動かし、読み出しの細胞群の発火から方向を決める。

- **網は売買を知らない。** 渡すのは画像だけで、残高・損益・保有量は渡さない
- 読み出しは設計者が決めた読み替え（`fly.readout`）。2026-10-12 に人間が Stonkfly と同じもの
  （右 DNp20 − 左 DNp20、DNpe017 が条件）に決めた。2026-10-14 に、学習の分を答えに入れるため
  MBON の項（`fly.readout.mbon`。B2）を足せるようにした（入れるかは腕ごと）。「ハエの脳に売買を
  判断する細胞がある」という意味ではない
- 網の状態は観測をまたいで引き継ぐ（`fly.simulation.carry_state`）。状態のファイルが無ければ
  静止状態から始め、あっても今の網・値と合わなければ**読まずに失敗にする**（別の脳の状態を
  引き継がない。消すかどうかは人間が決める）
- 失敗（網・座標表・状態が読めない、画像が無い、シミュレーションの失敗）はすべて例外にする。
  呼び出し側（`direction.resolve`）が方向の失敗として HOLD にする
- 学習（`fly.reward.enabled`。Phase 5）を入れたときだけ、KC→MBON の重みが観測ごとに変わる
  （`dopabae/plasticity.py`）。報酬・嫌悪は呼び出し側が決済の損益から決め（`dopabae/reward.py`）、
  ここには「報酬 / 嫌悪 / なし」だけが届く。観測の頭から `pulse_ms` のあいだ、ドーパミン細胞に電流を流す

判断ログに残すのは観測した値（発火数・発火率・差・関門の発火数）と、どの規則を当てたかだけ。
解釈は書かない（CLAUDE.md）。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Sequence

from . import connectome, lif, plasticity, retina
from . import reward as reward_module
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
    """発火数から、接近と回避の指標・差・関門・方向を出す。値は観測から計算するだけ。

    差は 右 DNp20 − 左 DNp20。`fly.readout.mbon` を入れた腕は、これに MBON の項を足す（B2）。
    """
    approach = select(network, config.readout_approach)
    avoid = select(network, config.readout_avoid)
    approach_hz = float(counts[approach].mean()) / seconds
    avoid_hz = float(counts[avoid].mean()) / seconds
    axis_hz = approach_hz - avoid_hz
    # MBON の項（fly.readout.mbon。B2）。入れた腕だけ、差に足す。学習で動いた分が答えに届くようにするため。
    mbon = None
    mbon_cells = None
    if config.readout_mbon is not None:
        term = config.readout_mbon
        up_cells, down_cells = select(network, (term.approach,)), select(network, (term.avoid,))
        up_hz = float(counts[up_cells].mean()) / seconds
        down_hz = float(counts[down_cells].mean()) / seconds
        term_hz = term.term_hz(up_hz, down_hz)
        mbon = {
            "hz": {term.approach: up_hz, term.avoid: down_hz},
            "baseline_hz": {term.approach: term.approach_baseline_hz, term.avoid: term.avoid_baseline_hz},
            "weight": term.weight,
            "term_hz": term_hz,
        }
        mbon_cells = {
            term.approach: [int(b) for b in network.body_ids[up_cells]],
            term.avoid: [int(b) for b in network.body_ids[down_cells]],
        }
        axis_hz += term_hz
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
        "mbon": mbon,
        "gate_spikes": gate_spikes,
        "cells": {
            "approach": [int(b) for b in network.body_ids[approach]],
            "avoid": [int(b) for b in network.body_ids[avoid]],
            "gate": [int(b) for b in network.body_ids[select(network, config.readout_gate)]] if config.readout_gate else None,
            "mbon": mbon_cells,
        },
    }


def _np_isin(values: Any, targets: list[int]) -> Any:
    return connectome._numpy().isin(values, targets)


class _Learning:
    """1 観測ぶんの学習の段取り。学習を入れていなければ作らない。"""

    def __init__(self, config: Config, network: connectome.Network, root: Path, stimulus: str):
        settings = config.reward
        self.settings = settings
        self.params = plasticity.RuleParams.from_settings(settings)
        self.circuit = plasticity.identify(network, settings.compartments)
        self.path = root / settings.memory_path
        self.origin = "rest"
        if self.path.exists():
            # 合わなければ PlasticityError（HOLD）。ファイルは残す（消すかどうかは人間が決める）
            memory = plasticity.load_memory(self.path, network, self.circuit, self.params)
            self.origin = "carried"
        else:
            memory = plasticity.Memory.fresh(self.circuit)
        self.memory = memory
        # いつ・どこを学ぶかは fly.reward.rule.learn_when（人間が決める）。学ばない結合でも記憶は元へ戻っていく
        self.learning_mbon = self._learning_mbon(settings, stimulus)
        mbons = [mbon for _, mbon in settings.compartments]
        if len(self.learning_mbon) == len(mbons):
            plastic = True
        elif not self.learning_mbon:
            plastic = False
        else:
            plastic = _np_isin(self.circuit.edge_compartment, [mbons.index(m) for m in self.learning_mbon])
        self.plastic = plastic
        self.learner = plasticity.Learner(
            self.circuit,
            self.params,
            memory,
            plasticity.base_weight(network, self.circuit, config.simulation.weight_per_synapse_mv),
            plastic=self.plastic,
        )
        self.stimulus_name = stimulus
        self.pulse = None
        self.pulse_cells: list[int] = []
        if stimulus != reward_module.NONE:
            cell_type = settings.reward_cells if stimulus == reward_module.REWARD else settings.aversive_cells
            index = network.type_index((cell_type,))
            if not len(index):
                raise FlyError(f"刺激する細胞が網にありません: {cell_type}")
            self.pulse = lif.Stimulus(index=index, current_mv=settings.pulse_mv)
            self.pulse_cells = [int(b) for b in network.body_ids[index]]

    @staticmethod
    def _learning_mbon(settings, stimulus: str) -> list[str]:
        """この観測で学ぶ区画（MBON の型）。"""
        mbons = [mbon for _, mbon in settings.compartments]
        if settings.learn_when == "always":
            return mbons
        if stimulus == reward_module.NONE:
            return []
        if settings.learn_when == "stimulus_only":
            return mbons
        return [settings.reward_mbon if stimulus == reward_module.REWARD else settings.aversive_mbon]

    def finish(self, network: connectome.Network) -> dict:
        """観測の数を進めて保存し、決まった回数ごとに写しを取る。判断ログに残す値を返す。"""
        self.memory.observations += 1
        sha256 = plasticity.save_memory(self.memory, self.path, network, self.circuit, self.params)
        snapshot = None
        if self.memory.observations % self.settings.weight_snapshot_every_n == 0:
            target = self.path.parent / "snapshots" / f"memory-{self.memory.observations:06d}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.path, target)
            snapshot = target.name
        return {
            "stimulus": self.stimulus_name,
            "pulse_cells": self.pulse_cells,
            "pulse_mv": self.settings.pulse_mv if self.pulse is not None else None,
            "pulse_ms": self.settings.pulse_ms if self.pulse is not None else None,
            "memory_from": self.origin,
            "memory_sha256": sha256,
            "snapshot": snapshot,
            "rule_key": self.params.key(),
            "circuit_key": self.circuit.key(),
            "learning_mbon": self.learning_mbon,  # この観測で学んだ区画。空なら学んでいない
            **self.memory.summary(self.params, self.circuit),
        }


def decide(
    config: Config,
    frames: Sequence[Image] | None,
    *,
    stimulus: str = reward_module.NONE,
    network: connectome.Network | None = None,
    column_map: retina.ColumnMap | None = None,
    repo_root: Path | None = None,
) -> Direction:
    """画像の枚を見せて網を動かし、方向を返す。`network` と `column_map` は差し替え口（テスト用）。

    `stimulus` はドーパミン細胞への刺激（`reward.NONE` / `REWARD` / `AVERSIVE`）。学習を入れていないときに
    なし以外が来たら失敗にする（学習しない網に刺激を入れない）。
    """
    if not frames:
        raise FlyError("ハエに見せる画像がありません")
    if config.readout_approach is None or config.readout_avoid is None:
        raise FlyError("fly.readout が決まっていません")
    if stimulus not in (reward_module.NONE, reward_module.REWARD, reward_module.AVERSIVE):
        raise FlyError(f"扱えない刺激です: {stimulus}")
    if stimulus != reward_module.NONE and not config.reward.enabled:
        raise FlyError("学習を入れていない網に、報酬・嫌悪の刺激は入れません")
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

    learning = _Learning(config, network, root, stimulus) if config.reward.enabled else None
    learner = learning.learner if learning is not None else None

    duration_ms = sim.neural_time_sec * 1000.0
    frame_ms = duration_ms / len(lights)
    # 刺激は観測の頭から pulse_ms のあいだ（Stonkfly と同じ）。枚をまたぐときは枚ごとに分けて流す。
    pulse_left = config.reward.pulse_ms if learning is not None and learning.pulse is not None else 0.0
    counts = None
    wall = 0.0
    for light in lights:
        segments = []
        if pulse_left > 0:
            first = min(pulse_left, frame_ms)
            segments.append((first, learning.pulse))
            pulse_left -= first
            if frame_ms - first > 1e-9:
                segments.append((frame_ms - first, None))
        else:
            segments.append((frame_ms, None))
        for length, extra in segments:
            result = lif.simulate(network, params, length, light=light, stimulus=extra, state=state, learner=learner)
            state = result.state
            counts = result.spike_counts if counts is None else counts + result.spike_counts
            wall += result.wall_sec
    # 読み出しを先に済ませる（ここで失敗すれば何も保存しない）。次に学習の状態、最後に網の状態を保存する。
    # 学習の状態の保存に失敗すれば網の状態も保存せず、次の回は前の状態と前の記憶から続く。
    reading = read(config, network, counts, duration_ms / 1000.0)
    learning_record = learning.finish(network) if learning is not None else None
    state_sha256 = lif.save_state(state, state_path, network, params) if sim.carry_state else None
    observation = {
        "approach_hz": reading["approach_hz"],
        "avoid_hz": reading["avoid_hz"],
        "axis_hz": reading["axis_hz"],
        # MBON の項を入れていない腕は null（差は DNp20 の左右だけ）
        "mbon": reading["mbon"],
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
        # 学習を入れていない回は null（重みは配線図のまま）
        "learning": learning_record,
    }
    mbon_part = f" ＋ MBON の項 {reading['mbon']['term_hz']:+.2f} Hz" if reading["mbon"] else ""
    gate_part = "関門なし" if reading["gate_spikes"] is None else f"関門 {reading['gate_spikes']} 回"
    reason = (
        f"接近の指標 {reading['approach_hz']:.2f} Hz − 回避の指標 {reading['avoid_hz']:.2f} Hz{mbon_part}"
        f"（{gate_part}）→ {reading['rule']}"
    )
    return Direction(source="fly", value=reading["value"], reason=reason, observation=observation)
