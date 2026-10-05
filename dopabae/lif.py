"""LIF シミュレーション（Phase 3 の中核）。

Shiu ら (2024) のモデル（Stonkfly が再利用したもの。`docs/CONNECTOME_SURVEY.md` 5 章）を
numpy で書く。値は `agent.yaml` の `fly.simulation`（どれも候補値）。

    u = v - v_rest
    du/dt = (x - u) / τm          不応期のあいだは止める
    dx/dt = -x / τs               同上
    u > v_th - v_rest で発火 → u = v_reset - v_rest、x = 0、不応期に入る
    発火は遅延のあと、行き先の x に「符号つきシナプス数 × 1 個あたりの重み」を足す

積分は線形の厳密解（Brian2 の `method='linear'` と同じ式）。1 刻みの順序は Brian2 の
既定（状態の更新 → 閾値 → シナプス → リセット）に合わせた。
**Brian2 / Stonkfly と発火が一致するかは確かめていない。**

**ここは売買を知らない。** 受け取るのは網と、外から与える発火率だけである。
残高・損益・建玉は引数にも取らない。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from .config import SimulationSettings
from .connectome import ConnectomeError, Network, _numpy


class SimulationError(ConnectomeError):
    """シミュレーションを回せない。呼び出し側は方向の失敗として HOLD にする。"""


@dataclass(frozen=True)
class Params:
    dt_ms: float
    membrane_tau_ms: float
    synaptic_tau_ms: float
    resting_mv: float
    reset_mv: float
    threshold_mv: float
    refractory_ms: float
    delay_ms: float
    weight_per_synapse_mv: float

    @classmethod
    def from_settings(cls, settings: SimulationSettings) -> "Params":
        return cls(
            dt_ms=settings.dt_ms,
            membrane_tau_ms=settings.membrane_tau_ms,
            synaptic_tau_ms=settings.synaptic_tau_ms,
            resting_mv=settings.resting_mv,
            reset_mv=settings.reset_mv,
            threshold_mv=settings.threshold_mv,
            refractory_ms=settings.refractory_ms,
            delay_ms=settings.delay_ms,
            weight_per_synapse_mv=settings.weight_per_synapse_mv,
        )

    @property
    def delay_steps(self) -> int:
        return max(1, round(self.delay_ms / self.dt_ms))

    @property
    def refractory_steps(self) -> int:
        return round(self.refractory_ms / self.dt_ms)

    def steps_for(self, duration_ms: float) -> int:
        return round(duration_ms / self.dt_ms)

    def coefficients(self) -> tuple[float, float, float]:
        """1 刻みの厳密解の係数。u' = a·u + c·x、x' = b·x。"""
        a = math.exp(-self.dt_ms / self.membrane_tau_ms)
        b = math.exp(-self.dt_ms / self.synaptic_tau_ms)
        ts, tm = self.synaptic_tau_ms, self.membrane_tau_ms
        c = ts / (ts - tm) * (b - a)
        return a, b, c


@dataclass(frozen=True)
class Drive:
    """外からの入力。細胞ごとにポアソン発火を与え、1 発ごとに x へ `weight_mv` を足す。"""

    index: Any  # numpy.ndarray[int64]。網の細胞番号
    rate_hz: Any  # numpy.ndarray[float64]
    weight_mv: float


@dataclass(frozen=True)
class Result:
    """観測した値だけを持つ。解釈は入れない。"""

    spike_counts: Any  # numpy.ndarray[int32]。細胞ごとの発火数
    steps: int
    duration_ms: float
    wall_sec: float
    input_spikes: int

    @property
    def total_spikes(self) -> int:
        return int(self.spike_counts.sum(dtype="int64"))

    @property
    def active_neurons(self) -> int:
        return int((self.spike_counts > 0).sum())


def _check_drive(network: Network, params: Params, drive: Drive) -> Any:
    np = _numpy()
    index = np.asarray(drive.index, dtype=np.int64)
    rate = np.asarray(drive.rate_hz, dtype=np.float64)
    if index.shape != rate.shape:
        raise SimulationError("入力の細胞と発火率の数が合いません")
    if len(index) and (index.min() < 0 or index.max() >= network.n_neurons):
        raise SimulationError("入力の細胞番号が網の外を指しています")
    if np.any(~np.isfinite(rate)) or np.any(rate < 0):
        raise SimulationError("入力の発火率は 0 以上の有限値です")
    prob = rate * params.dt_ms / 1000.0
    if np.any(prob > 1.0):
        raise SimulationError("入力の発火率が 1 刻みに 1 発を超えます")
    return index, prob


def simulate(
    network: Network,
    params: Params,
    duration_ms: float,
    drive: Drive | None = None,
    seed: int = 0,
) -> Result:
    """網を `duration_ms` だけ動かし、細胞ごとの発火数を返す。

    同じ網・値・入力・種からは同じ結果が出る。
    """
    np = _numpy()
    n = network.n_neurons
    steps = params.steps_for(duration_ms)
    if n == 0 or steps <= 0:
        raise SimulationError("網が空か、動かす時間が 0 です")

    a, b, c = (np.float32(v) for v in params.coefficients())
    u_th = np.float32(params.threshold_mv - params.resting_mv)
    u_reset = np.float32(params.reset_mv - params.resting_mv)
    delay = params.delay_steps
    refractory = params.refractory_steps
    slots = delay + 1

    indptr = network.indptr
    post = network.post
    weight = network.signed_count.astype(np.float32) * np.float32(params.weight_per_synapse_mv)

    u = np.zeros(n, dtype=np.float32)  # 静止電位から始める
    x = np.zeros(n, dtype=np.float32)
    ready_at = np.zeros(n, dtype=np.int64)  # この刻みから不応期が明ける
    pending = np.zeros((slots, n), dtype=np.float32)  # 遅延中の入力（輪状に使う）
    counts = np.zeros(n, dtype=np.int32)

    rng = np.random.default_rng(seed)
    drive_index = drive_prob = None
    drive_weight = np.float32(0.0)
    if drive is not None:
        drive_index, drive_prob = _check_drive(network, params, drive)
        drive_weight = np.float32(drive.weight_mv)
    input_spikes = 0

    started = time.perf_counter()
    for t in range(steps):
        # 1. 状態の更新（不応期の細胞は止める）
        active = ready_at <= t
        np.copyto(u, a * u + c * x, where=active)
        np.copyto(x, b * x, where=active)

        # 2. 閾値
        fired = np.flatnonzero(active & (u > u_th))

        # 3. シナプス: 遅延の明けた入力と、外からの入力
        due = pending[t % slots]
        x += due
        due.fill(0.0)
        if drive_index is not None and len(drive_index):
            hits = drive_index[rng.random(len(drive_index)) < drive_prob]
            if hits.size:
                np.add.at(x, hits, drive_weight)
                input_spikes += int(hits.size)
        if fired.size:
            starts = indptr[fired]
            lengths = indptr[fired + 1] - starts
            total = int(lengths.sum())
            if total:
                edges = np.repeat(starts - (np.cumsum(lengths) - lengths), lengths) + np.arange(total)
                np.add.at(pending[(t + delay) % slots], post[edges], weight[edges])

        # 4. リセット
        if fired.size:
            u[fired] = u_reset
            x[fired] = 0.0
            ready_at[fired] = t + refractory
            counts[fired] += 1
    wall = time.perf_counter() - started

    return Result(
        spike_counts=counts,
        steps=steps,
        duration_ms=steps * params.dt_ms,
        wall_sec=wall,
        input_spikes=input_spikes,
    )
