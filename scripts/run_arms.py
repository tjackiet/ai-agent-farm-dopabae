#!/usr/bin/env python3
"""腕をまとめて1回ぶん回す。定期実行（launchd / cron）から呼ぶ。

腕とは、**方向の出どころだけが違う設定**のこと（`scripts/make_arm.py`）。
評価（`docs/IMPLEMENTATION_PLAN.md` Phase 4）は腕どうしの比較で行うので、
同じ相場を同時に見せる必要がある。順番に回すと、遅い腕のぶんだけ後ろの腕が
ずれた相場を見ることになるため、**並行して起動する**。

各腕の実行は `dopabae.run` 側のロックで守られている。前の回がまだ走っていれば
その腕はこの回を飛ばす（`dopabae/lock.py`）。

このスクリプトは**発注しない。** 各腕の `agent.yaml` に従うだけで、
`runtime.dry_run` が true なら注文は組み立てられるだけである。

使いかた:

    python3 scripts/run_arms.py                  # arms/*.yaml を全部
    python3 scripts/run_arms.py --config arms/fly.yaml   # 腕を指定
    python3 scripts/run_arms.py --timeout-sec 300

`arms/` が空なら、リポジトリ直下の `agent.yaml` を1つの腕として回す。

**終了コードは常に 0。** 腕が1つ落ちても定期実行そのものは失敗ではない。
落ちたことは標準出力の要約と各腕の `run.err.log` に残る。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARMS_DIR = REPO_ROOT / "arms"
# 15分間隔に対する既定の上限。これを超えた腕は打ち切る。打ち切られた腕は
# プロセスが死ぬのでロックも外れ、次の回はやり直せる。
DEFAULT_TIMEOUT_SEC = 600


@dataclass(frozen=True)
class Result:
    """腕1つぶんの結果。判断の中身は各腕の判断ログに残る。"""

    name: str
    status: str  # ok | skipped | failed | timeout
    seconds: float
    action: str | None = None
    detail: str | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "seconds": round(self.seconds, 1),
            "action": self.action,
            "detail": self.detail,
        }


def discover(configs: list[str] | None, arms_dir: Path = ARMS_DIR) -> list[Path]:
    """回す腕の設定を決める。

    明示された設定があればそれだけ。無ければ `arms/*.yaml` を全部。
    それも無ければリポジトリ直下の `agent.yaml` を1つの腕として扱う。
    """
    if configs:
        return [Path(c) for c in configs]
    found = sorted(arms_dir.glob("*.yaml")) if arms_dir.is_dir() else []
    return found or [REPO_ROOT / "agent.yaml"]


def log_paths(config_path: Path) -> tuple[Path, Path]:
    """腕ごとのログの置き場所。`arms/fly.yaml` なら `var/arms/fly/`。"""
    name = config_path.stem
    base = REPO_ROOT / "var" / "arms" / name if config_path.parent.name == "arms" else REPO_ROOT / "var"
    return base / "run.log", base / "run.err.log"


def summarize(line: str) -> tuple[str, str | None, str | None]:
    """`dopabae.run` の1行 JSON から、状態・行動・補足を読む。"""
    try:
        payload = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return "failed", None, "出力を JSON として読めません"
    if not isinstance(payload, dict):
        return "failed", None, "出力が辞書ではありません"
    if payload.get("skipped"):
        return "skipped", None, str(payload.get("reason", ""))[:200]
    action = payload.get("action")
    detail = payload.get("error") or payload.get("cage")
    return "ok", (str(action) if action else None), (str(detail)[:200] if detail else None)


def run_one(config_path: Path, timeout_sec: int, python: str) -> Result:
    """腕を1つ回す。落ちても例外を投げない。定期実行を止めないため。"""
    name = config_path.stem
    started = time.monotonic()
    out_path, err_path = log_paths(config_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [python, "-m", "dopabae.run", "--config", str(config_path)]
    try:
        proc = subprocess.run(  # noqa: S603 - 引数は自前で組み立てている
            argv, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout_sec, check=False
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started
        with err_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{time.strftime('%FT%T%z')}] {int(elapsed)} 秒で打ち切った\n")
        return Result(name, "timeout", elapsed, detail=f"{timeout_sec} 秒で打ち切った")
    except OSError as exc:
        return Result(name, "failed", time.monotonic() - started, detail=str(exc)[:200])

    elapsed = time.monotonic() - started
    stamp = time.strftime("%FT%T%z")
    if proc.stdout:
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{proc.stdout.rstrip()}\n")
    if proc.stderr:
        with err_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {proc.stderr.rstrip()}\n")
    if proc.returncode != 0:
        return Result(name, "failed", elapsed, detail=f"終了コード {proc.returncode}")
    last = (proc.stdout or "").strip().splitlines()
    if not last:
        return Result(name, "failed", elapsed, detail="出力が空です")
    status, action, detail = summarize(last[-1])
    return Result(name, status, elapsed, action=action, detail=detail)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="腕をまとめて1回ぶん回す（発注は各腕の agent.yaml に従う）",
    )
    parser.add_argument("--config", action="append", default=None, metavar="PATH", help="回す腕の agent.yaml")
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC, help="腕1つあたりの上限")
    parser.add_argument("--python", default=sys.executable, help="腕を回す Python")
    args = parser.parse_args(argv)

    targets = discover(args.config)
    missing = [p for p in targets if not p.is_file()]
    if missing:
        print(json.dumps({"error": f"設定が見つかりません: {[str(p) for p in missing]}"}, ensure_ascii=False))
        return 1

    # 同じ相場を見せるため、順番ではなく並行して起動する。
    with ThreadPoolExecutor(max_workers=max(1, len(targets))) as pool:
        results = list(pool.map(lambda p: run_one(p, args.timeout_sec, args.python), targets))

    print(json.dumps(
        {"at": time.strftime("%FT%T%z"), "arms": [r.as_dict() for r in results]},
        ensure_ascii=False,
    ))
    # 腕が落ちても定期実行そのものは失敗ではない。
    return 0


if __name__ == "__main__":
    sys.exit(main())
