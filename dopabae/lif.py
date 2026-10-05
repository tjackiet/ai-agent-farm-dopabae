"""LIF シミュレーション（Phase 3 の中核）。

Shiu ら (2024) のモデルを Stonkfly が拡張したもの（`docs/CONNECTOME_SURVEY.md` 1.1・5 章）を
numpy で書く。値は `agent.yaml` の `fly.simulation`（どれも候補値）。

    u = v - v_rest（細胞ごと。KC だけ静止電位が違う）
    du/dt = (I + x - u - w) / τm        不応期のあいだは止める
    dx/dt = -x / τs                     同上
    dw/dt = -w / τa                     KC の適応。不応期中も減る
    v > v_th で発火 → u = v_reset - v_rest、x = 0、不応期に入る。KC は w に適応ぶんを足す
    発火は遅延のあと、行き先の x に「符号つきシナプス数 × 1 個あたりの重み」を足す

- I は一定電流。ラミナには背景の電流を、光受容細胞には明るさから作った電流を流す
- 調節性の細胞（ドーパミンなど）の発火は、行き先の x を動かさない（結合は残す）
- 不応期中の細胞へ届いた入力は捨てる（`refractory_input: keep` なら足す）

積分は線形の厳密解。1 刻みの順序は「状態の更新 → 閾値 → シナプス → リセット」
（Brian2 の既定、Stonkfly のカーネルと同じ）。Stonkfly は事象駆動で計算を飛ばすが、
厳密解なので実数の上では同じ値になる（浮動小数点の丸めは違いうる）。

**ここは売買を知らない。** 受け取るのは網と、光受容細胞に見せる明るさだけである。
残高・損益・建玉は引数にも取らない。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import SimulationSettings
from .connectome import ConnectomeError, Network, _numpy

# 状態のファイルの形式の版。形を変えたら上げる。
STATE_VERSION = 1
# KC の型名の頭（Stonkfly と同じ）。
KC_PREFIX = "KC"


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
    refractory_input: str = "drop"
    # KC。kc_resting_mv が None なら他の細胞と同じに扱う。
    kc_resting_mv: float | None = None
    kc_adaptation_mv: float = 0.0
    kc_adaptation_tau_ms: float = 200.0
    # 背景の活動（ラミナへの一定電流）。
    lamina_types: tuple[str, ...] = ()
    lamina_bias_mv: float = 0.0
    # 光受容細胞への入力。
    input_max_mv: float = 30.0
    input_half_saturation: float = 0.02
    input_smoothing_ms: float = 10.0
    input_chunk_ms: float = 10.0

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
            refractory_input=settings.refractory_input,
            kc_resting_mv=settings.kc_resting_mv,
            kc_adaptation_mv=settings.kc_adaptation_mv,
            kc_adaptation_tau_ms=settings.kc_adaptation_tau_ms,
            lamina_types=tuple(settings.lamina_types),
            lamina_bias_mv=settings.lamina_bias_mv,
            input_max_mv=settings.input_max_mv,
            input_half_saturation=settings.input_half_saturation,
            input_smoothing_ms=settings.input_smoothing_ms,
            input_chunk_ms=settings.input_chunk_ms,
        )

    @property
    def delay_steps(self) -> int:
        return max(1, round(self.delay_ms / self.dt_ms))

    @property
    def refractory_steps(self) -> int:
        return round(self.refractory_ms / self.dt_ms)

    @property
    def chunk_steps(self) -> int:
        return max(1, round(self.input_chunk_ms / self.dt_ms))

    def steps_for(self, duration_ms: float) -> int:
        return round(duration_ms / self.dt_ms)

    def coefficients(self) -> tuple[float, float, float]:
        """1 刻みの厳密解の係数。u' = a·u + c·x、x' = b·x（電流と適応を除く）。"""
        a = math.exp(-self.dt_ms / self.membrane_tau_ms)
        b = math.exp(-self.dt_ms / self.synaptic_tau_ms)
        ts, tm = self.synaptic_tau_ms, self.membrane_tau_ms
        c = ts / (ts - tm) * (b - a)
        return a, b, c

    def adaptation_coefficients(self) -> tuple[float, float]:
        """KC の適応の係数。u' = … − d·w、w' = e·w。"""
        a = math.exp(-self.dt_ms / self.membrane_tau_ms)
        e = math.exp(-self.dt_ms / self.kc_adaptation_tau_ms)
        ta, tm = self.kc_adaptation_tau_ms, self.membrane_tau_ms
        return ta / (ta - tm) * (e - a), e

    def photoreceptor_current(self, light: Any) -> Any:
        """明るさ（0〜1）から光受容細胞へ流す電流（mV 相当）。飽和する。"""
        return self.input_max_mv * light / (self.input_half_saturation + light)

    def key(self) -> str:
        packed = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(packed.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Light:
    """光受容細胞に見せる明るさ（0〜1）。網の細胞番号と値。"""

    index: Any  # numpy.ndarray[int64]
    value: Any  # numpy.ndarray[float]


@dataclass(frozen=True)
class Drive:
    """外からのポアソン入力。1 発ごとに x へ `weight_mv` を足す。**負荷試験のためのもの。**"""

    index: Any  # numpy.ndarray[int64]
    rate_hz: Any  # numpy.ndarray[float64]
    weight_mv: float


@dataclass
class State:
    """網の状態。観測をまたいで引き継ぐ（`fly.simulation.carry_state`）。"""

    clock: int  # これまでに進めた刻みの数
    u: Any  # 静止電位からの差（mV）
    x: Any  # シナプス入力（mV）
    adaptation: Any  # KC の適応（mV）。KC 以外は 0
    ready_at: Any  # この刻みから不応期が明ける
    pending: Any  # 遅延中の入力。pending[clock % slots] がその刻みに届く
    light: Any  # 平滑化した明るさ。光受容細胞以外は 0

    def copy(self) -> "State":
        return State(
            clock=self.clock,
            u=self.u.copy(),
            x=self.x.copy(),
            adaptation=self.adaptation.copy(),
            ready_at=self.ready_at.copy(),
            pending=self.pending.copy(),
            light=self.light.copy(),
        )


@dataclass(frozen=True)
class Result:
    """観測した値だけを持つ。解釈は入れない。"""

    spike_counts: Any  # numpy.ndarray[int32]。細胞ごとの発火数
    steps: int
    duration_ms: float
    wall_sec: float
    input_spikes: int
    state: State

    @property
    def total_spikes(self) -> int:
        return int(self.spike_counts.sum(dtype="int64"))

    @property
    def active_neurons(self) -> int:
        return int((self.spike_counts > 0).sum())


def initial_state(network: Network, params: Params) -> State:
    """静止状態。全細胞が自分の静止電位にいて、入力も適応も無い。"""
    np = _numpy()
    n = network.n_neurons
    return State(
        clock=0,
        u=np.zeros(n, dtype=np.float32),
        x=np.zeros(n, dtype=np.float32),
        adaptation=np.zeros(n, dtype=np.float32),
        ready_at=np.zeros(n, dtype=np.int64),
        pending=np.zeros((params.delay_steps + 1, n), dtype=np.float32),
        light=np.zeros(n, dtype=np.float32),
    )


def _identity(network: Network, params: Params) -> dict:
    """状態がどの網・どの値のものか。引き継ぐときに照らし合わせる。"""
    return {
        "version": STATE_VERSION,
        "network_key": network.meta.get("key"),
        "params_key": params.key(),
        "n_neurons": network.n_neurons,
    }


def save_state(state: State, path: Path, network: Network, params: Params) -> str:
    """状態を保存し、ファイルの SHA-256 を返す。一時ファイルに書いてから移す。"""
    np = _numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("wb") as handle:
        np.savez(
            handle,
            u=state.u,
            x=state.x,
            adaptation=state.adaptation,
            ready_at=state.ready_at,
            pending=state.pending,
            light=state.light,
            meta=np.array(json.dumps({**_identity(network, params), "clock": int(state.clock)})),
        )
    part.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_state(path: Path, network: Network, params: Params) -> State:
    """保存した状態を読む。網か値が違えば読まない（別の脳の状態を引き継がない）。"""
    np = _numpy()
    if not path.exists():
        raise SimulationError(f"状態のファイルがありません: {path}")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        for key, expected in _identity(network, params).items():
            if meta.get(key) != expected:
                raise SimulationError(f"状態のファイルの {key} が今の網・値と合いません: {path}")
        state = State(
            clock=int(meta["clock"]),
            u=data["u"],
            x=data["x"],
            adaptation=data["adaptation"],
            ready_at=data["ready_at"],
            pending=data["pending"],
            light=data["light"],
        )
    for name in ("u", "x", "adaptation", "pending", "light"):
        if not np.isfinite(getattr(state, name)).all():
            raise SimulationError(f"状態のファイルの {name} に有限でない値があります: {path}")
    return state


def _check_light(network: Network, light: Light) -> tuple[Any, Any]:
    np = _numpy()
    index = np.asarray(light.index, dtype=np.int64)
    value = np.asarray(light.value, dtype=np.float32)
    if index.shape != value.shape:
        raise SimulationError("明るさの細胞と値の数が合いません")
    if len(index) and (index.min() < 0 or index.max() >= network.n_neurons):
        raise SimulationError("明るさの細胞番号が網の外を指しています")
    if len(np.unique(index)) != len(index):
        raise SimulationError("明るさの細胞番号が重複しています")
    if not np.isfinite(value).all():
        raise SimulationError("明るさに有限でない値があります")
    return index, np.clip(value, 0.0, 1.0)


def _check_drive(network: Network, params: Params, drive: Drive) -> tuple[Any, Any]:
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
    *,
    light: Light | None = None,
    drive: Drive | None = None,
    state: State | None = None,
    seed: int = 0,
) -> Result:
    """網を `duration_ms` だけ動かし、細胞ごとの発火数と、動かしたあとの状態を返す。

    `state` を渡せばそこから続け、渡さなければ静止状態から始める。渡した状態は書き換えない。
    同じ網・値・入力・状態・種からは同じ結果が出る（乱数を使うのは `drive` だけ）。
    """
    np = _numpy()
    n = network.n_neurons
    steps = params.steps_for(duration_ms)
    if n == 0 or steps <= 0:
        raise SimulationError("網が空か、動かす時間が 0 です")
    if params.refractory_input not in ("drop", "keep"):
        raise SimulationError(f"扱えない refractory_input です: {params.refractory_input}")

    st = initial_state(network, params) if state is None else state.copy()
    if st.u.shape != (n,) or st.pending.shape != (params.delay_steps + 1, n):
        raise SimulationError("状態の大きさが網と合いません")

    a, b, c = (np.float32(v) for v in params.coefficients())
    one_minus_a = np.float32(1.0 - float(a))
    delay = params.delay_steps
    refractory = params.refractory_steps
    slots = delay + 1
    drop = params.refractory_input == "drop"

    # 細胞ごとの静止電位と、閾値までの距離。KC だけ静止電位が違う。
    rest = np.full(n, params.resting_mv, dtype=np.float32)
    kc = np.zeros(0, dtype=np.int64)
    if params.kc_resting_mv is not None:
        kc = network.type_index(prefix=KC_PREFIX)
        rest[kc] = params.kc_resting_mv
    u_th = (np.float32(params.threshold_mv) - rest).astype(np.float32)
    u_reset = np.float32(params.reset_mv - params.resting_mv)
    is_kc = np.zeros(n, dtype=bool)
    is_kc[kc] = True
    kc_d, kc_e = (np.float32(v) for v in params.adaptation_coefficients())
    kc_jump = np.float32(params.kc_adaptation_mv)

    # 一定電流。ラミナには背景の電流、光受容細胞には明るさから作った電流。
    current = np.zeros(n, dtype=np.float32)
    if params.lamina_types:
        current[network.type_index(params.lamina_types)] = params.lamina_bias_mv
    light_index = light_target = None
    if light is not None:
        light_index, light_target = _check_light(network, light)

    indptr = network.indptr
    post = network.post
    weight = network.signed_count.astype(np.float32) * np.float32(params.weight_per_synapse_mv)
    fast = ~np.asarray(network.modulatory, dtype=bool)
    counts = np.zeros(n, dtype=np.int32)

    rng = np.random.default_rng(seed)
    drive_index = drive_prob = None
    drive_weight = np.float32(0.0)
    if drive is not None:
        drive_index, drive_prob = _check_drive(network, params, drive)
        drive_weight = np.float32(drive.weight_mv)
    input_spikes = 0

    u, x, w = st.u, st.x, st.adaptation
    started = time.perf_counter()
    done = 0
    while done < steps:
        chunk = min(params.chunk_steps, steps - done)
        # 明るさを平滑化し、この区切りのあいだの電流を決める（Stonkfly と同じく区切りの頭で）。
        if light_index is not None and len(light_index):
            alpha = np.float32(1.0 - math.exp(-chunk * params.dt_ms / params.input_smoothing_ms))
            smoothed = st.light[light_index]
            smoothed += alpha * (light_target - smoothed)
            st.light[light_index] = smoothed
            current[light_index] = params.photoreceptor_current(smoothed)
        drive_term = one_minus_a * current

        for _ in range(chunk):
            t = st.clock
            # 1. 状態の更新（不応期の細胞は止める。適応は止めずに減らす）
            active = st.ready_at <= t
            new_u = a * u + c * x + drive_term
            if kc.size:
                new_u[kc] -= kc_d * w[kc]
            np.copyto(u, new_u, where=active)
            np.copyto(x, b * x, where=active)
            if kc.size:
                w[kc] *= kc_e

            # 2. 閾値
            fired = np.flatnonzero(active & (u > u_th))

            # 3. シナプス: 遅延の明けた入力と、外からの入力
            due = st.pending[t % slots]
            if drop:
                due *= active
            x += due
            due.fill(0.0)
            if drive_index is not None and len(drive_index):
                hits = drive_index[rng.random(len(drive_index)) < drive_prob]
                if drop:
                    hits = hits[active[hits]]
                if hits.size:
                    np.add.at(x, hits, drive_weight)
                    input_spikes += int(hits.size)
            senders = fired[fast[fired]]
            if senders.size:
                starts = indptr[senders]
                lengths = indptr[senders + 1] - starts
                total = int(lengths.sum())
                if total:
                    edges = np.repeat(starts - (np.cumsum(lengths) - lengths), lengths) + np.arange(total)
                    np.add.at(st.pending[(t + delay) % slots], post[edges], weight[edges])

            # 4. リセット（各細胞の静止電位へ）。KC は適応を足す
            if fired.size:
                u[fired] = u_reset
                x[fired] = 0.0
                st.ready_at[fired] = t + refractory
                counts[fired] += 1
                if kc.size:
                    w[fired[is_kc[fired]]] += kc_jump
            st.clock += 1
        done += chunk
    wall = time.perf_counter() - started

    return Result(
        spike_counts=counts,
        steps=steps,
        duration_ms=steps * params.dt_ms,
        wall_sec=wall,
        input_spikes=input_spikes,
        state=st,
    )
