#!/usr/bin/env python3
"""評価の腕（arm）の設定を作る。

腕とは、**方向の出どころだけが違う設定**のこと。設計メモ 4 章の評価 3・4 は
腕どうしの比較で、replay（過去の足での再実行）ではない。`bitbank paper` の
約定判定を自前で真似ると、真似が本物とずれた分だけ評価が嘘になるためである。

腕ごとに次を分ける。分けないと、同じペーパー口座と同じ判断ログを奪い合う。

- ペーパー口座の状態（`cli.state_path`）
- 判断ログ（`memory.decisions.path`）と画像（`memory.vision.path`）
- スナップショットと実績（`agent.status_output` / `agent.performance_output`）

使いかた:

    python3 scripts/make_arm.py cage-only  --source always_approach
    python3 scripts/make_arm.py random     --source random --seed 20260915
    python3 scripts/make_arm.py fly        --source fly

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


def build(raw: dict, name: str, source: str, seed: int | None, weights: dict | None) -> dict:
    """正本の agent.yaml から、腕ぶんの設定を組み立てる。

    変えるのは方向の出どころと置き場所だけ。檻とリスクの値は触らない。
    腕ごとに戦略が違うと、比べているものが方向の差でなくなる。
    """
    arm = json.loads(json.dumps(raw))  # 深い複製。元の辞書を壊さない
    base = f"var/arms/{name}"

    arm["agent"]["id"] = f"{raw['agent']['id']}-{name}"
    arm["agent"]["status_output"] = f"{base}/status.yaml"
    arm["agent"]["performance_output"] = f"{base}/performance.yaml"
    arm["cli"]["state_path"] = f"{base}/paper-state.json"
    arm["memory"]["root"] = f"{base}/memory/"
    arm["memory"]["decisions"]["path"] = f"{base}/memory/decisions/{{date}}.jsonl"
    arm["memory"]["daily"]["path"] = f"{base}/memory/daily/{{date}}.md"
    arm["memory"]["lessons"]["path"] = f"{base}/memory/lessons.md"
    arm["memory"]["weights"]["path"] = f"{base}/memory/weights/{{date}}.npz"
    arm["memory"]["vision"]["path"] = f"{base}/memory/vision/{{date}}/{{run_id}}.png"

    arm["direction"]["source"] = source
    arm["direction"]["random_seed"] = seed
    arm["direction"]["random_weights"] = weights
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
    parser.add_argument("--config", default=None, help="元にする agent.yaml（既定はリポジトリ直下）")
    args = parser.parse_args(argv)

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
    arm = build(raw, args.name, args.source, args.seed, weights)

    ARMS_DIR.mkdir(parents=True, exist_ok=True)
    target = ARMS_DIR / f"{args.name}.yaml"
    header = (
        f"# 評価の腕「{args.name}」（生成物。手で編集せず scripts/make_arm.py で作り直す）\n"
        f"# 元: {source_path.name} / 方向の出どころ: {args.source}\n"
        "#\n"
        "# 変えたのは方向の出どころと置き場所だけです。檻とリスクの値は正本と同じで、\n"
        "# 腕の差が方向の差だけになるようにしています。\n"
    )
    body = yaml.safe_dump(arm, allow_unicode=True, sort_keys=False)
    target.write_text(f"{header}\n{body}", encoding="utf-8")
    print(f"{target.relative_to(REPO_ROOT)} を書きました（発注はしていません）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
