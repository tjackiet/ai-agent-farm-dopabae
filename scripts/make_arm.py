#!/usr/bin/env python3
"""評価の腕（arm）の設定を作る。

腕とは、**方向の出どころだけが違う設定**のこと。設計メモ 4 章の評価 3・4 は
腕どうしの比較で、replay（過去の足での再実行）ではない。ハエ版の腕は、網への
入力のしかた（光受容細胞への入力の半飽和、1 観測のあいだに見せる動き、配色、
格子を画像へ置く上下の向き）と、読み出しの関門の有無・MBON の項（B2）も変えられる。
檻とリスクの値は変えない。`bitbank paper` の
約定判定を自前で真似ると、真似が本物とずれた分だけ評価が嘘になるためである。

腕ごとに次を分ける。分けないと、同じペーパー口座と同じ判断ログを奪い合う。

- ペーパー口座の状態（`cli.state_path`）
- 判断ログ（`memory.decisions.path`）と画像（`memory.vision.path`）
- スナップショットと実績（`agent.status_output` / `agent.performance_output`）

使いかた:

    python3 scripts/make_arm.py cage-only  --source always_approach
    python3 scripts/make_arm.py random     --source random --seed 20260915
    python3 scripts/make_arm.py fly        --source fly
    python3 scripts/make_arm.py fly-hs0.2  --source fly --half-saturation 0.2
    python3 scripts/make_arm.py fly-grow   --source fly --motion grow_last
    python3 scripts/make_arm.py fly-mono   --source fly --palette up=118,118,118 --palette down=118,118,118
    python3 scripts/make_arm.py fly-flipy  --source fly --flip-y false
    python3 scripts/make_arm.py fly-hs1-nogate --source fly --half-saturation 1.0 --no-gate
    python3 scripts/make_arm.py fly-b2     --source fly --mbon-baseline MBON11=17.7 --mbon-baseline MBON07=1.14

作った設定で回す:

    BITBANK_PAPER_STATE_PATH=var/arms/cage-only/paper-state.json \
      bitbank paper init --jpy=1000000
    python3 -m dopabae.run --config arms/cage-only.yaml --dry-run

比べる:

    python3 -m dopabae.evaluate --config arms/fly.yaml --config arms/random.yaml

**このスクリプトは発注しない。** 設定ファイルを書くだけである。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ARMS_DIR = REPO_ROOT / "arms"
SOURCES = ("always_approach", "random", "fly")
PALETTE_KEYS = ("background", "text", "up", "down", "wick", "bid", "ask")


def parse_palette(items: list[str]) -> dict[str, list[int]]:
    """`KEY=R,G,B` の並びを辞書にする。読めなければ ValueError。"""
    out: dict[str, list[int]] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or key not in PALETTE_KEYS:
            raise ValueError(f"--palette は {'|'.join(PALETTE_KEYS)}=R,G,B です: {item!r}")
        parts = value.split(",")
        if len(parts) != 3 or not all(p.strip().isdigit() and 0 <= int(p) <= 255 for p in parts):
            raise ValueError(f"--palette の色は 0〜255 の 3 つの整数です: {item!r}")
        out[key] = [int(p) for p in parts]
    return out


def parse_mbon_baselines(items: list[str]) -> dict[str, float]:
    """`TYPE=HZ` の並びを辞書にする。読めなければ ValueError。"""
    out: dict[str, float] = {}
    for item in items:
        name, sep, value = item.partition("=")
        try:
            hz = float(value)
        except ValueError:
            hz = None
        if not sep or not name.strip() or hz is None or not hz > 0:
            raise ValueError(f"--mbon-baseline は 型名=正の Hz です: {item!r}")
        out[name.strip()] = hz
    return out


def build(
    raw: dict,
    name: str,
    source: str,
    seed: int | None,
    weights: dict | None,
    half_saturation: float | None = None,
    motion: str | None = None,
    palette: dict[str, list[int]] | None = None,
    flip_y: bool | None = None,
    no_gate: bool = False,
    mbon_baselines: dict[str, float] | None = None,
) -> dict:
    """正本の agent.yaml から、腕ぶんの設定を組み立てる。

    変えるのは方向の出どころ（ハエ版なら光受容細胞への入力の半飽和・見せる動き・配色・
    格子の上下の向き・読み出しの関門の有無・MBON の項も）と置き場所だけ。
    檻とリスクの値は触らない。腕ごとに戦略が違うと、比べているものが方向の差でなくなる。
    """
    arm = json.loads(json.dumps(raw))  # 深い複製。元の辞書を壊さない
    base = f"var/arms/{name}"

    arm["agent"]["id"] = f"{raw['agent']['id']}-{name}"
    arm["agent"]["status_output"] = f"{base}/status.yaml"
    arm["agent"]["performance_output"] = f"{base}/performance.yaml"
    arm["cli"]["state_path"] = f"{base}/paper-state.json"
    # ロックは腕ごとに分ける。分けないと腕が互いの実行を止める。
    arm["runtime"]["lock_path"] = f"{base}/run.lock"
    arm["memory"]["root"] = f"{base}/memory/"
    arm["memory"]["decisions"]["path"] = f"{base}/memory/decisions/{{date}}.jsonl"
    arm["memory"]["daily"]["path"] = f"{base}/memory/daily/{{date}}.md"
    arm["memory"]["lessons"]["path"] = f"{base}/memory/lessons.md"
    arm["memory"]["vision"]["path"] = f"{base}/memory/vision/{{date}}/{{run_id}}.png"
    # 網の状態も腕ごとに分ける。分けないとハエ版の腕どうしが状態を上書きし合う。
    arm["fly"]["simulation"]["checkpoint_path"] = f"{base}/fly/state.npz"
    # 学習の状態と報酬の台帳も腕ごと（Phase 5）。分けないと、腕どうしが記憶と決済を取り合う。
    arm["fly"]["reward"]["memory_path"] = f"{base}/fly/memory.npz"
    arm["fly"]["reward"]["ledger_path"] = f"{base}/fly/reward-ledger.json"

    arm["direction"]["source"] = source
    arm["direction"]["random_seed"] = seed
    arm["direction"]["random_weights"] = weights
    if half_saturation is not None:
        arm["fly"]["simulation"]["photoreceptor_input"]["half_saturation"] = half_saturation
    if motion is not None:
        arm["fly"]["vision"]["motion"]["mode"] = motion
    if palette:
        arm["fly"]["vision"]["palette"].update(palette)
    if flip_y is not None:
        arm["fly"]["retina"]["flip_y"] = flip_y
    if no_gate:
        arm["fly"]["readout"]["gate_neurons"] = None
    if mbon_baselines:
        mbon = arm["fly"]["readout"].get("mbon")
        if not mbon:
            raise ValueError("元の設定に fly.readout.mbon がありません")
        expected = {mbon["approach"], mbon["avoid"]}
        if set(mbon_baselines) != expected:
            raise ValueError(f"--mbon-baseline は {' と '.join(sorted(expected))} の両方を書きます: {sorted(mbon_baselines)}")
        mbon["enabled"] = True
        mbon["approach_baseline_hz"] = mbon_baselines[mbon["approach"]]
        mbon["avoid_baseline_hz"] = mbon_baselines[mbon["avoid"]]
    return arm


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="腕の名前（例: cage-only / random / fly）")
    parser.add_argument("--source", required=True, choices=SOURCES, help="方向の出どころ")
    parser.add_argument("--seed", type=int, default=None, help="source=random の乱数種")
    parser.add_argument(
        "--weights", default=None, metavar="JSON",
        help='source=random の三値の頻度。比べる相手の値を写す。'
             '例: \'{"APPROACH":0.5,"AVOID":0.2,"NONE":0.3}\'。'
             "python3 -m dopabae.evaluate の matched_random_weights が写すべき値",
    )
    parser.add_argument(
        "--half-saturation", type=float, default=None,
        help="source=fly の光受容細胞への入力の半飽和（fly.simulation.photoreceptor_input.half_saturation）。"
             "省けば正本と同じ",
    )
    parser.add_argument(
        "--motion", default=None, choices=("none", "grow_last"),
        help="source=fly の 1 観測のあいだに見せる動き（fly.vision.motion.mode）。省けば正本と同じ",
    )
    parser.add_argument(
        "--palette", action="append", default=[], metavar="KEY=R,G,B",
        help="source=fly の配色を差し替える（fly.vision.palette）。何度でも指定できる。"
             "例: --palette up=118,118,118 --palette down=118,118,118",
    )
    parser.add_argument(
        "--flip-y", default=None, choices=("true", "false"),
        help="source=fly の格子を画像へ置く上下の向き（fly.retina.flip_y）。省けば正本と同じ",
    )
    parser.add_argument(
        "--no-gate", action="store_true",
        help="source=fly の読み出しの関門（fly.readout.gate_neurons）を外す。差だけで方向を出す",
    )
    parser.add_argument(
        "--mbon-baseline", action="append", default=[], metavar="TYPE=HZ",
        help="source=fly の読み出しに MBON の項（fly.readout.mbon。B2）を入れ、その基準を置く。"
             "approach と avoid の両方を書く。例: --mbon-baseline MBON11=17.7 --mbon-baseline MBON07=1.14",
    )
    parser.add_argument("--config", default=None, help="元にする agent.yaml（既定はリポジトリ直下）")
    args = parser.parse_args(argv)
    if args.half_saturation is not None:
        if args.source != "fly":
            parser.error("--half-saturation は --source fly の腕だけで使う（対照群は画像を見ない）")
        if not args.half_saturation > 0:
            parser.error("--half-saturation は正の値")
    if args.motion is not None and args.source != "fly":
        parser.error("--motion は --source fly の腕だけで使う（対照群は画像を見ない）")
    if (args.palette or args.flip_y is not None) and args.source != "fly":
        parser.error("--palette と --flip-y は --source fly の腕だけで使う（対照群は画像を見ない）")
    if (args.no_gate or args.mbon_baseline) and args.source != "fly":
        parser.error("--no-gate と --mbon-baseline は --source fly の腕だけで使う（対照群は網を回さない）")
    try:
        palette = parse_palette(args.palette)
        mbon_baselines = parse_mbon_baselines(args.mbon_baseline)
    except ValueError as exc:
        parser.error(str(exc))
    flip_y = None if args.flip_y is None else args.flip_y == "true"

    weights = None
    if args.weights:
        try:
            weights = json.loads(args.weights)
        except json.JSONDecodeError as exc:
            print(f"--weights を JSON として読めません: {exc}", file=sys.stderr)
            return 1
        if not isinstance(weights, dict):
            print("--weights は辞書です", file=sys.stderr)
            return 1

    source_path = Path(args.config) if args.config else REPO_ROOT / "agent.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    try:
        arm = build(
            raw, args.name, args.source, args.seed, weights, args.half_saturation, args.motion, palette, flip_y,
            args.no_gate, mbon_baselines,
        )
    except ValueError as exc:
        parser.error(str(exc))

    ARMS_DIR.mkdir(parents=True, exist_ok=True)
    target = ARMS_DIR / f"{args.name}.yaml"
    parts = ["方向の出どころ"]
    if args.half_saturation is not None:
        parts.append(f"光受容細胞への入力の半飽和（{args.half_saturation:g}）")
    if args.motion is not None:
        parts.append(f"見せる動き（{args.motion}）")
    if palette:
        parts.append("配色（" + "・".join(f"{k}={','.join(map(str, v))}" for k, v in palette.items()) + "）")
    if flip_y is not None:
        parts.append(f"格子の上下の向き（flip_y: {args.flip_y}）")
    if args.no_gate:
        parts.append("読み出しの関門（外した）")
    if mbon_baselines:
        parts.append("MBON の項（基準 " + "・".join(f"{k} {v:g} Hz" for k, v in mbon_baselines.items()) + "）")
    changed = "・".join(parts) + ("と置き場所" if len(parts) == 1 else "・置き場所")
    header = (
        f"# 評価の腕「{args.name}」（生成物。手で編集せず scripts/make_arm.py で作り直す）\n"
        f"# 元: {source_path.name} / 方向の出どころ: {args.source}\n"
        "#\n"
        f"# 変えたのは{changed}だけです。檻とリスクの値は正本と同じで、\n"
        "# 腕の差が方向の差だけになるようにしています。\n"
    )
    body = yaml.safe_dump(arm, allow_unicode=True, sort_keys=False)
    target.write_text(f"{header}\n{body}", encoding="utf-8")
    print(f"{target.relative_to(REPO_ROOT)} を書きました（発注はしていません）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
