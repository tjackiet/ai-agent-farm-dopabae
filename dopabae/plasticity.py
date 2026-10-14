"""学習（Phase 5）。KC→MBON のつながりの強さを、ドーパミン細胞と KC の発火の順序で変える。

規則は Stonkfly の `stonkfly/neural/rule.py`（Huang, Luo ら 2024 の式 3.2〜3.5 を、既存の結合ごとに
中心化して当てたもの）と同じ。2026-10-13 にソースで確かめた（`docs/IMPLEMENTATION_PLAN.md` Phase 5）。
10 ms 以下の区切り h ごとに、

    K, D   : KC とドーパミン細胞（DAN）の発火率の痕跡（時定数 trace_sec の一次遅れ）
    drive  = η [ r_K ⊙ (Gᵀ D_mid) − (Gᵀ r_D) ⊙ K_mid ]      結合ごと
    τ_U du/dt = −u + τ_U drive                              記憶（時定数 memory_decay_sec で元へ戻る）
    τ_W dw/dt = u − w                                       効き目（filter_sec の一次遅れ）
    重み = 元の重み × (1 + w)、u と w は [min_fraction − 1, max_fraction − 1] に収める

- **ドーパミンの前に発火していた KC の結合は弱まり、ドーパミンのあとに発火した KC の結合は強まる。**
  弱めるだけの規則ではない
- 発火率が一定なら drive は 0 になる（中心化）。変えるのは発火の順序の偏りだけである
- G は区画の対応（PAM11 → KC→MBON07、PPL101 → KC→MBON11）と、DAN から MBON への接触の割合
  （Stonkfly の `circuit.py` と同じ）。ほかの区画の結合は変えない
- 時間はすべて神経時間（1 観測 0.5 秒）。15 分おきの運用では、神経時間 1 秒が実時間の 30 分にあたる

**ここは売買を知らない。** 受け取るのは発火数だけで、価格・損益は受け取らない。
報酬・嫌悪は、決定的なコードが決済の損益から決めて、DAN への刺激として渡す（`dopabae/reward.py`）。
学習で変わるのはシナプスの重みだけで、閾値や読み出しの細胞は変えない（CLAUDE.md）。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import RewardSettings
from .connectome import ConnectomeError, Network, _numpy

# 記憶のファイルの形式の版。形を変えたら上げる。
MEMORY_VERSION = 1
KC_PREFIX = "KC"
# Stonkfly と同じく、1 回の区切りは 10 ms 以下（rule.py の検査と同じ）。
MAX_BIN_SEC = 0.010


class PlasticityError(ConnectomeError):
    """学習を回せない。呼び出し側は方向の失敗として HOLD にする。"""


@dataclass(frozen=True)
class RuleParams:
    """規則の値（`fly.reward.rule`）。ハエは書き換えない。"""

    eta: float
    trace_sec: float
    filter_sec: float
    memory_decay_sec: float
    min_fraction: float
    max_fraction: float
    dan_baseline_hz: float

    @classmethod
    def from_settings(cls, settings: RewardSettings) -> "RuleParams":
        if settings.memory_decay_sec is None:
            raise PlasticityError("fly.reward.rule.memory_decay_sec が決まっていません（人間が決める）")
        return cls(
            eta=settings.eta,
            trace_sec=settings.trace_sec,
            filter_sec=settings.filter_sec,
            memory_decay_sec=settings.memory_decay_sec,
            min_fraction=settings.min_fraction,
            max_fraction=settings.max_fraction,
            dan_baseline_hz=settings.dan_baseline_hz,
        )

    @property
    def bounds(self) -> tuple[float, float]:
        return self.min_fraction - 1.0, self.max_fraction - 1.0

    def key(self) -> str:
        packed = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(packed.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Circuit:
    """学習の場所。網の上の位置と、DAN から結合への割り振り（G）。"""

    edges: Any  # 学習する結合の、網の結合の並び（CSR）での位置
    edge_kc: Any  # 結合ごとに、kc_cells の何番目の KC から出ているか
    edge_mbon: Any  # 結合ごとの行き先（網の細胞番号）
    edge_compartment: Any  # 結合ごとに、compartments の何番目の区画か
    kc_cells: Any  # 学習する結合を出している KC（網の細胞番号）
    dan_cells: Any  # 報酬の細胞、嫌悪の細胞の順（網の細胞番号）
    gain: Any  # (DAN の数, 結合の数)。各結合の列は、その区画の DAN にだけ値を持ち、和は 1
    compartments: tuple[tuple[str, str], ...]  # (DAN の型, MBON の型)

    @property
    def cells(self) -> Any:
        """シミュレーションで発火数を数える細胞（KC、DAN の順）。"""
        return _numpy().concatenate([self.kc_cells, self.dan_cells])

    def key(self) -> str:
        digest = hashlib.sha256()
        for array in (self.edges, self.kc_cells, self.dan_cells, self.gain):
            digest.update(_numpy().ascontiguousarray(array).tobytes())
        digest.update(json.dumps(self.compartments).encode("utf-8"))
        return digest.hexdigest()[:16]


def identify(network: Network, compartments: tuple[tuple[str, str], ...]) -> Circuit:
    """学習の場所を網から探す。Stonkfly の `circuit.py` の `identify()` と同じ数えかた。

    - 学習する結合: KC から、各区画の MBON への結合すべて（シナプス数の下限は置かない）
    - G[d, e] = c(d, t) / Σ_{d' ∈ 区画} c(d', t)。t は結合 e の行き先、c は d から t へのシナプス数の絶対値の和
    """
    np = _numpy()
    n = network.n_neurons
    kc = network.type_index(prefix=KC_PREFIX)
    if not len(kc):
        raise PlasticityError("網に KC がありません")
    is_kc = np.zeros(n, dtype=bool)
    is_kc[kc] = True
    edge_pre = np.repeat(np.arange(n, dtype=np.int64), np.diff(network.indptr))

    dan_groups = []
    mbon_groups = []
    for dan_type, mbon_type in compartments:
        dans = network.type_index((dan_type,))
        mbons = network.type_index((mbon_type,))
        if not len(dans) or not len(mbons):
            raise PlasticityError(f"網に {dan_type} か {mbon_type} がありません")
        dan_groups.append(dans)
        mbon_groups.append(mbons)
    dan_cells = np.concatenate(dan_groups).astype(np.int64)

    is_target = np.zeros(n, dtype=bool)
    for mbons in mbon_groups:
        is_target[mbons] = True
    edges = np.flatnonzero(is_kc[edge_pre] & is_target[network.post]).astype(np.int64)
    if not len(edges):
        raise PlasticityError("KC から学習の場所の MBON への結合がありません")
    if (network.signed_count[edges] <= 0).any():
        raise PlasticityError("KC→MBON に正でない結合があります（興奮の符号を保てません）")
    edge_mbon = network.post[edges].astype(np.int64)
    kc_cells, edge_kc = np.unique(edge_pre[edges], return_inverse=True)
    edge_compartment = np.zeros(len(edges), dtype=np.int64)
    for number, mbons in enumerate(mbon_groups):
        edge_compartment[np.isin(edge_mbon, mbons)] = number

    gain = np.zeros((len(dan_cells), len(edges)), dtype=np.float64)
    offset = 0
    for dans, mbons, (dan_type, mbon_type) in zip(dan_groups, mbon_groups, compartments):
        for target in mbons:
            contacts = np.array([_contact(network, int(d), int(target)) for d in dans], dtype=np.float64)
            total = contacts.sum()
            if total <= 0:
                raise PlasticityError(f"{dan_type} から {mbon_type}（{int(network.body_ids[target])}）への接触がありません")
            columns = edge_mbon == target
            gain[offset:offset + len(dans), columns] = (contacts / total)[:, None]
        offset += len(dans)
    return Circuit(
        edges=edges,
        edge_kc=edge_kc.astype(np.int64),
        edge_mbon=edge_mbon,
        edge_compartment=edge_compartment,
        kc_cells=kc_cells.astype(np.int64),
        dan_cells=dan_cells,
        gain=gain,
        compartments=tuple(compartments),
    )


def _contact(network: Network, pre: int, post: int) -> float:
    """pre から post へのシナプス数の絶対値の和。"""
    start, stop = int(network.indptr[pre]), int(network.indptr[pre + 1])
    targets = network.post[start:stop]
    return float(abs(network.signed_count[start:stop][targets == post]).sum())


@dataclass
class Memory:
    """学習の状態。観測をまたいで引き継ぐ。最初はすべて 0（元の重み）。"""

    u: Any  # 結合ごとの記憶（元からのずれ）
    w: Any  # 結合ごとの効き目（元からのずれ）。重みに掛かるのはこちら
    kc_trace: Any  # KC ごとの痕跡（Hz）
    dan_trace: Any  # DAN ごとの痕跡（Hz）
    observations: int = 0

    @classmethod
    def fresh(cls, circuit: Circuit) -> "Memory":
        np = _numpy()
        return cls(
            u=np.zeros(len(circuit.edges)),
            w=np.zeros(len(circuit.edges)),
            kc_trace=np.zeros(len(circuit.kc_cells)),
            dan_trace=np.zeros(len(circuit.dan_cells)),
        )

    def efficacy(self) -> Any:
        return 1.0 + self.w

    def summary(self, params: RuleParams, circuit: Circuit | None = None) -> dict:
        """観測した値だけ。解釈は書かない。`circuit` を渡すと、区画（MBON の型）ごとの値も出す。"""
        np = _numpy()
        eff = self.efficacy()
        low, high = params.bounds
        by_mbon = None
        if circuit is not None:
            by_mbon = {}
            for number, (_, mbon_type) in enumerate(circuit.compartments):
                part = eff[circuit.edge_compartment == number]
                by_mbon[mbon_type] = {
                    "edges": int(len(part)),
                    "mean_efficacy": round(float(part.mean()), 6) if len(part) else None,
                    "min_efficacy": round(float(part.min()), 6) if len(part) else None,
                    "max_efficacy": round(float(part.max()), 6) if len(part) else None,
                }
        return {
            "by_mbon": by_mbon,
            "plastic_edges": int(len(eff)),
            "changed_edges": int((np.abs(self.w) > 1e-9).sum()),
            "mean_efficacy": round(float(eff.mean()), 6),
            "min_efficacy": round(float(eff.min()), 6),
            "max_efficacy": round(float(eff.max()), 6),
            "at_lower_bound": int((self.w <= low + 1e-12).sum()),
            "at_upper_bound": int((self.w >= high - 1e-12).sum()),
            "observations": int(self.observations),
            "w_sha256": hashlib.sha256(np.ascontiguousarray(self.w).tobytes()).hexdigest(),
        }


def advance(
    memory: Memory, circuit: Circuit, params: RuleParams, kc_rate: Any, dan_rate: Any, h: float, *, plastic: bool = True
) -> None:
    """区切り 1 つぶん、痕跡と記憶を進める。式は Stonkfly の `rule.py` の `advance()` と同じ。

    `plastic` が False なら drive を 0 にする（痕跡は進み、記憶は元へ戻っていく）。
    """
    np = _numpy()
    if not 0.0 < h <= MAX_BIN_SEC + 1e-12:
        raise PlasticityError(f"学習の区切りは 0 より長く 10 ms 以下です: {h * 1000:g} ms")
    a_k = math.exp(-h / params.trace_sec)
    a_d = math.exp(-h / params.trace_sec)
    half_k, half_d = math.sqrt(a_k), math.sqrt(a_d)
    k_mid = memory.kc_trace * half_k + kc_rate * (1.0 - half_k)
    d_mid = memory.dan_trace * half_d + dan_rate * (1.0 - half_d)
    memory.kc_trace = memory.kc_trace * a_k + kc_rate * (1.0 - a_k)
    memory.dan_trace = memory.dan_trace * a_d + dan_rate * (1.0 - a_d)

    # DAN の痕跡と発火率を、結合ごとに割り振る。
    dan_trace_edge = circuit.gain.T @ d_mid
    dan_rate_edge = circuit.gain.T @ dan_rate
    drive = params.eta * (kc_rate[circuit.edge_kc] * dan_trace_edge - dan_rate_edge * k_mid[circuit.edge_kc])
    if not plastic:
        drive = np.zeros_like(drive)

    tau_u, tau_w = params.memory_decay_sec, params.filter_sec
    e_u = math.exp(-h / tau_u)
    e_w = math.exp(-h / tau_w)
    c = tau_u / (tau_u - tau_w) * (e_u - e_w)
    u_old = memory.u
    memory.u = u_old * e_u + drive * tau_u * (1.0 - e_u)
    memory.w = memory.w * e_w + u_old * c + drive * tau_u * ((1.0 - e_w) - c)
    low, high = params.bounds
    np.clip(memory.u, low, high, out=memory.u)
    np.clip(memory.w, low, high, out=memory.w)


class Learner:
    """`lif.simulate` に渡す差し込み口。区切りごとに発火数を受け取り、重みを書き換える。"""

    def __init__(self, circuit: Circuit, params: RuleParams, memory: Memory, base_weight: Any, plastic: bool = True):
        self.circuit = circuit
        self.params = params
        self.memory = memory
        # False なら痕跡と記憶の減衰だけ進め、drive を 0 にする（刺激の無い観測では学ばない設定のため）
        self.plastic = plastic
        self.base = base_weight  # 学習する結合の元の重み（mV）。網の符号つきシナプス数 × 1 個あたりの重み
        self.cells = circuit.cells
        self._n_kc = len(circuit.kc_cells)

    def begin(self, weight: Any) -> None:
        """いまの効き目を重みに入れる。シミュレーションの頭で 1 回呼ぶ。"""
        weight[self.circuit.edges] = self.base * self.memory.efficacy()

    def step(self, spikes: Any, seconds: float, weight: Any) -> None:
        """区切りのあいだの発火数（`cells` の順）から発火率を出し、規則を進めて重みを書き換える。"""
        rates = spikes / seconds
        kc_rate = rates[: self._n_kc]
        dan_rate = rates[self._n_kc:] - self.params.dan_baseline_hz
        advance(self.memory, self.circuit, self.params, kc_rate, dan_rate, seconds, plastic=self.plastic)
        weight[self.circuit.edges] = self.base * self.memory.efficacy()


def base_weight(network: Network, circuit: Circuit, weight_per_synapse_mv: float) -> Any:
    np = _numpy()
    return network.signed_count[circuit.edges].astype(np.float64) * weight_per_synapse_mv


def _identity(network: Network, circuit: Circuit, params: RuleParams) -> dict:
    return {
        "version": MEMORY_VERSION,
        "network_key": network.meta.get("key"),
        "circuit_key": circuit.key(),
        "rule_key": params.key(),
    }


def save_memory(memory: Memory, path: Path, network: Network, circuit: Circuit, params: RuleParams) -> str:
    """学習の状態を保存し、ファイルの SHA-256 を返す。一時ファイルに書いてから移す。"""
    np = _numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("wb") as handle:
        np.savez(
            handle,
            u=memory.u,
            w=memory.w,
            kc_trace=memory.kc_trace,
            dan_trace=memory.dan_trace,
            meta=np.array(json.dumps({**_identity(network, circuit, params), "observations": int(memory.observations)})),
        )
    part.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_memory(path: Path, network: Network, circuit: Circuit, params: RuleParams) -> Memory:
    """保存した学習の状態を読む。網・学習の場所・規則の値が違えば読まない（別の記憶を引き継がない）。"""
    np = _numpy()
    if not path.exists():
        raise PlasticityError(f"学習の状態のファイルがありません: {path}")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        for key, expected in _identity(network, circuit, params).items():
            if meta.get(key) != expected:
                raise PlasticityError(f"学習の状態のファイルの {key} が今の網・規則と合いません: {path}")
        memory = Memory(
            u=data["u"].astype(np.float64),
            w=data["w"].astype(np.float64),
            kc_trace=data["kc_trace"].astype(np.float64),
            dan_trace=data["dan_trace"].astype(np.float64),
            observations=int(meta["observations"]),
        )
    sizes = {"u": len(circuit.edges), "w": len(circuit.edges), "kc_trace": len(circuit.kc_cells), "dan_trace": len(circuit.dan_cells)}
    for name, size in sizes.items():
        array = getattr(memory, name)
        if array.shape != (size,) or not np.isfinite(array).all():
            raise PlasticityError(f"学習の状態のファイルの {name} の形か値が正しくありません: {path}")
    return memory
