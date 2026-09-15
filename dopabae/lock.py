"""同じ腕の二重起動を防ぐ。

15分ごとの定期実行で、前の回がまだ走っているうちに次の回が始まると、
**同じペーパー口座を 2 つの実行が同時に触る。** `paper tick` が二重に走り、
同じ判断で 2 回発注しうる。ハエのシミュレーション（Phase 3）が入ると
1 回の実行が長くなるので、この危険は現実のものになる。

`flock` で守る。プロセスが死ねば OS が自動で外すので、異常終了しても
ロックが残らない。取れなかった回は**何もせずに終わる**（CLAUDE.md
「判断できないときは HOLD」。実行しないことによる機会損失は許容し、
二重発注は許容しない）。

腕ごとにロックのファイルを分ける（`runtime.lock_path`）。分けないと、
腕どうしが互いをブロックしてしまう。
"""

from __future__ import annotations

import errno
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:  # POSIX のみ。macOS と Linux はどちらも持つ。
    import fcntl
except ImportError:  # pragma: no cover - 本リポジトリの対象外の環境
    fcntl = None  # type: ignore[assignment]


class AlreadyRunning(Exception):
    """同じ腕の実行がまだ走っている。"""


@contextmanager
def hold(path: Path | str) -> Iterator[bool]:
    """ロックを取る。取れなければ `AlreadyRunning`。

    `yield` する真偽値は「ロックが実際に効いているか」。`fcntl` が無い環境では
    False を返す。**守れていないことを黙って隠さない。**
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if fcntl is None:
        yield False
        return

    # "a+" で開く。既存の内容を切り詰めないので、ロックを取れなかった側が
    # 相手の pid を読める。
    handle = target.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            handle.seek(0)
            holder = handle.read().strip() or "不明"
            raise AlreadyRunning(
                f"同じ腕の実行がまだ走っている（pid {holder}、ロック {target}）"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        yield True
    finally:
        # 閉じれば flock も外れる。明示的な解放は要らない。
        handle.close()
