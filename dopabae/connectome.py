"""配線図を計算用の配列にする（Phase 3 の下ごしらえ）。

MaleCNS v1.0 の flat-connectome（`scripts/fetch_connectome.py` で取ったもの）から、
LIF シミュレーションが使う網を作る。網は「細胞の並び」と、発火元ごとに並べた
結合（CSR: `indptr` / `post` / `signed_count`）でできている。

**ここは売買を知らない。** 扱うのは細胞・結合・符号だけである。

作った網は `fly.connectome.local_dir` の `compiled/` にキャッシュする（Git 管理外）。
ファイル名は入力の指紋と作りかたの指紋から決まるので、どちらかが変われば別の
ファイルになる。定期実行は 1 周ごとに新しいプロセスなので、毎回 1 GB の結合を
読み直さずに済むようにするためである。

変換には pyarrow が、網の読み込みとシミュレーションには numpy が要る
（`requirements-connectome.txt`）。どちらもこのモジュールの中でだけ読み込む。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, Config

# キャッシュの形式の版。形を変えたら上げる（古いキャッシュを読まないため）。
COMPILED_VERSION = 2

# 符号を正にする結合（発火元の型名の頭、行き先の型名）。R8 → aMe12 は Xiao ら 2023 に基づく
# Stonkfly の仮定（docs/CONNECTOME_SURVEY.md 1.1）。
R8_AME12 = ("R8", "aMe12")


class ConnectomeError(Exception):
    """網を作れない・読めない（入力が無い・指紋が合わない・形が想定と違う）。"""


def _numpy():
    try:
        import numpy
    except ImportError as exc:
        raise ConnectomeError(
            "numpy が要ります: python3 -m pip install -r requirements-connectome.txt"
        ) from exc
    return numpy


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_input(config: Config, name: str) -> Path:
    """`agent.yaml` の指紋と一致する入力だけを返す。"""
    expected = config.connectome_files[name]
    path = REPO_ROOT / config.connectome_local_dir / expected.path
    if not path.exists():
        raise ConnectomeError(
            f"{name} がありません: {path}\n先に python3 scripts/fetch_connectome.py を実行する"
        )
    if path.stat().st_size != expected.size_bytes or sha256_of(path) != expected.sha256:
        raise ConnectomeError(f"{name} の指紋が agent.yaml と合いません: {path}")
    return path


@dataclass(frozen=True)
class Spec:
    """網の作りかた。どれも設計者の選択であって、データの性質ではない。"""

    neuron_policy: str
    synapse_threshold: int | None
    transmitter_column: str
    inhibitory_transmitters: tuple[str, ...]
    modulatory_transmitters: tuple[str, ...]
    # 符号を正にする結合（発火元の型名の頭、行き先の型名）。
    excitatory_overrides: tuple[tuple[str, str], ...]
    # 入力ファイルの SHA-256。中身が変われば別の網になる。
    inputs_sha256: tuple[tuple[str, str], ...]

    def key(self) -> str:
        packed = json.dumps(
            {"version": COMPILED_VERSION, **asdict(self)}, sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(packed.encode("utf-8")).hexdigest()[:16]


def spec_from_config(config: Config, synapse_threshold: int | None | str = "config") -> Spec:
    """`agent.yaml` から作りかたを組む。閾値だけは計測のために差し替えられる。"""
    threshold = config.connectome_synapse_threshold if synapse_threshold == "config" else synapse_threshold
    if threshold is not None and (not isinstance(threshold, int) or threshold < 1):
        raise ConnectomeError(f"シナプス数の閾値は 1 以上の整数か None です: {threshold!r}")
    sim = config.simulation
    return Spec(
        neuron_policy=sim.neuron_policy,
        synapse_threshold=threshold,
        transmitter_column=sim.transmitter_column,
        inhibitory_transmitters=sim.inhibitory_transmitters,
        modulatory_transmitters=sim.modulatory_transmitters,
        excitatory_overrides=(R8_AME12,) if sim.r8_to_ame12_excitatory else (),
        inputs_sha256=tuple(sorted((k, v.sha256) for k, v in config.connectome_files.items())),
    )


@dataclass(frozen=True)
class Network:
    """網。`body_ids` は昇順で、細胞の番号はこの並びの位置である。

    `indptr[i]:indptr[i + 1]` が細胞 i から出る結合で、行き先が `post`、
    符号つきのシナプス数が `signed_count`（抑制なら負）。
    `modulatory` が真の細胞の発火は、行き先の膜電位を動かさない（結合は残す）。
    """

    body_ids: Any  # numpy.ndarray[int64]
    indptr: Any  # numpy.ndarray[int64]
    post: Any  # numpy.ndarray[int32]
    signed_count: Any  # numpy.ndarray[int32]
    cell_type: Any  # numpy.ndarray[str]。型の無い細胞は ""
    side: Any  # numpy.ndarray[str]。"L" / "R" / ""
    modulatory: Any  # numpy.ndarray[bool]
    meta: dict

    @property
    def n_neurons(self) -> int:
        return int(len(self.body_ids))

    @property
    def n_edges(self) -> int:
        return int(len(self.post))

    def type_index(self, names: tuple[str, ...] = (), prefix: str | None = None) -> Any:
        """型名が `names` のどれか、または `prefix` で始まる細胞の番号。"""
        np = _numpy()
        mask = np.isin(self.cell_type, list(names)) if names else np.zeros(self.n_neurons, dtype=bool)
        if prefix:
            mask |= np.char.startswith(self.cell_type.astype(str), prefix)
        return np.flatnonzero(mask)

    def index_of(self, body_ids: Any) -> Any:
        """body_id の並びを細胞番号へ。網に無いものは -1。"""
        np = _numpy()
        wanted = np.asarray(body_ids, dtype=np.int64)
        pos = np.searchsorted(self.body_ids, wanted)
        pos = np.clip(pos, 0, max(self.n_neurons - 1, 0))
        found = self.n_neurons > 0
        hit = (self.body_ids[pos] == wanted) if found else np.zeros(len(wanted), dtype=bool)
        return np.where(hit, pos, -1)


def sign_of(transmitter: object, inhibitory: tuple[str, ...]) -> int:
    """抑制として扱う物質なら -1、それ以外（不明・欠損を含む）は +1（Stonkfly と同じ）。"""
    return -1 if isinstance(transmitter, str) and transmitter in inhibitory else 1


def build_csr(n: int, pre: Any, post: Any, signed: Any) -> tuple[Any, Any, Any]:
    """発火元ごとに結合を並べる。同じ発火元の中では元の順を保つ（決定的にするため）。"""
    np = _numpy()
    pre = np.asarray(pre, dtype=np.int64)
    order = np.argsort(pre, kind="stable")
    counts = np.bincount(pre, minlength=n) if len(pre) else np.zeros(n, dtype=np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return (
        indptr,
        np.asarray(post, dtype=np.int32)[order],
        np.asarray(signed, dtype=np.int32)[order],
    )


def compile_network(config: Config, spec: Spec) -> Network:
    """flat-connectome から網を作る。pyarrow はここでだけ使う。"""
    np = _numpy()
    try:
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
        import pyarrow.feather as feather
    except ImportError as exc:
        raise ConnectomeError(
            "pyarrow が要ります: python3 -m pip install -r requirements-connectome.txt"
        ) from exc

    annotations = verified_input(config, "annotations")
    weights = verified_input(config, "weights")
    transmitters = verified_input(config, "neurotransmitters")

    table = feather.read_table(
        annotations, columns=["bodyId", "type", "status", "superclass", "instance"]
    )
    if spec.neuron_policy != "assigned_superclass":  # pragma: no cover - 設定で弾いてある
        raise ConnectomeError(f"扱えない neuron_policy です: {spec.neuron_policy}")
    # superclass が付いていて、status が Glia でないもの（Stonkfly と同じ）。
    superclass = pc.fill_null(table["superclass"], "")
    keep = pc.and_(
        pc.not_equal(superclass, ""),
        pc.invert(pc.fill_null(pc.equal(table["status"], "Glia"), False)),
    )
    kept = table.filter(keep).sort_by("bodyId")
    body_ids = kept["bodyId"].to_numpy()
    if len(body_ids) == 0:
        raise ConnectomeError("網に入れる細胞が 1 つもありません。注釈の形を確かめる")
    if len(np.unique(body_ids)) != len(body_ids):
        raise ConnectomeError("注釈の bodyId が重複しています")
    cell_type = np.array([t or "" for t in kept["type"].to_pylist()], dtype=str)
    side = np.array(
        [(i[-1] if isinstance(i, str) and i[-2:] in ("_L", "_R") else "") for i in kept["instance"].to_pylist()],
        dtype=str,
    )

    # 符号と調節性は発火元の伝達物質で決める。表に無い・不明な細胞は興奮（sign_of と同じ規則）。
    nt = feather.read_table(transmitters, columns=["body", spec.transmitter_column])
    nt_ids = nt["body"].to_numpy()
    column = nt[spec.transmitter_column]
    inhibitory = pc.fill_null(
        pc.is_in(column, value_set=pa_strings(spec.inhibitory_transmitters)), False
    ).to_numpy(zero_copy_only=False)
    modulators = pc.fill_null(
        pc.is_in(column, value_set=pa_strings(spec.modulatory_transmitters)), False
    ).to_numpy(zero_copy_only=False)
    sign = np.ones(len(body_ids), dtype=np.int32)
    modulatory = np.zeros(len(body_ids), dtype=bool)
    pos = np.clip(np.searchsorted(body_ids, nt_ids), 0, len(body_ids) - 1)
    hit = body_ids[pos] == nt_ids
    sign[pos[hit & inhibitory]] = -1
    modulatory[pos[hit & modulators]] = True

    # 1 億 5 千万行を丸ごと読まない。網の細胞どうしの結合だけを取り出す。
    id_set = pa_ints(body_ids)
    expression = ds.field("body_pre").isin(id_set) & ds.field("body_post").isin(id_set)
    if spec.synapse_threshold is not None:
        expression = expression & (ds.field("weight") >= spec.synapse_threshold)
    edges = ds.dataset(weights, format="ipc").to_table(
        columns=["body_pre", "body_post", "weight"], filter=expression
    )
    pre = np.searchsorted(body_ids, edges["body_pre"].to_numpy())
    post = np.searchsorted(body_ids, edges["body_post"].to_numpy())
    count = edges["weight"].to_numpy()
    del edges
    if len(count) and int(count.max()) > np.iinfo(np.int32).max:
        raise ConnectomeError("シナプス数が int32 に収まりません。形が想定と違う")
    signed = count.astype(np.int32) * sign[pre]
    overridden = 0
    for pre_prefix, post_type in spec.excitatory_overrides:
        # 細胞ごとの印を先に作る（結合ごとに型名の配列を作ると数 GB になる）。
        from_mask = np.char.startswith(cell_type, pre_prefix)
        to_mask = cell_type == post_type
        flip = from_mask[pre] & to_mask[post]
        overridden += int(flip.sum())
        signed[flip] = np.abs(signed[flip])
    indptr, post_sorted, signed_sorted = build_csr(len(body_ids), pre, post, signed)

    meta = {
        "version": COMPILED_VERSION,
        "key": spec.key(),
        "spec": asdict(spec),
        "n_neurons": int(len(body_ids)),
        "n_edges": int(len(post_sorted)),
        "n_synapses": int(np.abs(signed_sorted).sum(dtype=np.int64)),
        "n_inhibitory_neurons": int((sign < 0).sum()),
        "n_modulatory_neurons": int(modulatory.sum()),
        "n_overridden_edges": overridden,
    }
    return Network(body_ids, indptr, post_sorted, signed_sorted, cell_type, side, modulatory, meta)


def pa_strings(values: tuple[str, ...]):
    import pyarrow as pa

    return pa.array(list(values), type=pa.string())


def pa_ints(values: Any):
    import pyarrow as pa

    return pa.array(values, type=pa.int64())


def compiled_path(config: Config, spec: Spec) -> Path:
    return REPO_ROOT / config.connectome_local_dir / "compiled" / f"network-{spec.key()}.npz"


def save(network: Network, path: Path) -> None:
    """一時ファイルに書いてから移す。途中で落ちても壊れたキャッシュを残さない。"""
    np = _numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("wb") as handle:
        np.savez(
            handle,
            body_ids=network.body_ids,
            indptr=network.indptr,
            post=network.post,
            signed_count=network.signed_count,
            cell_type=network.cell_type,
            side=network.side,
            modulatory=network.modulatory,
            meta=np.array(json.dumps(network.meta, ensure_ascii=False)),
        )
    part.replace(path)


def load(path: Path, spec: Spec | None = None) -> Network:
    """キャッシュを読む。作りかたの指紋が違えば読まない。"""
    np = _numpy()
    if not path.exists():
        raise ConnectomeError(f"網のキャッシュがありません: {path}")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("version") != COMPILED_VERSION:
            raise ConnectomeError(f"網のキャッシュの版が違います: {path}")
        if spec is not None and meta.get("key") != spec.key():
            raise ConnectomeError(f"網のキャッシュの作りかたが agent.yaml と違います: {path}")
        return Network(
            body_ids=data["body_ids"],
            indptr=data["indptr"],
            post=data["post"],
            signed_count=data["signed_count"],
            cell_type=data["cell_type"],
            side=data["side"],
            modulatory=data["modulatory"],
            meta=meta,
        )


def load_or_compile(config: Config, spec: Spec) -> tuple[Network, bool]:
    """キャッシュがあれば読み、無ければ作って保存する。2 つめは作ったかどうか。"""
    path = compiled_path(config, spec)
    if path.exists():
        return load(path, spec), False
    network = compile_network(config, spec)
    save(network, path)
    return network, True
