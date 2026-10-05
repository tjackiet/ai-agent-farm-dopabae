#!/usr/bin/env python3
"""配線図データを取得する。

配線図データはリポジトリに含めない（`docs/CONNECTOME_SURVEY.md` 3.6）。
取得元（`fly.connectome.download_base`）と指紋（`fly.connectome.files`）だけを
`agent.yaml` に置き、中身はこのスクリプトで `fly.connectome.local_dir`（Git 管理外）へ取る。

**指紋が合わないファイルは置かない。** 取得したものの SHA-256 と大きさが
`agent.yaml` と一致したときだけ、所定の名前へ移す。

**既にあるファイルは上書きしない。** 指紋が合えば何もしない。合わなければ止まり、
中身を確かめて消すのは人間に任せる。

使いかた:

    python3 scripts/fetch_connectome.py              # 全部（約 1 GB）
    python3 scripts/fetch_connectome.py annotations  # 名前を指定

`gs://` は公開バケットの HTTPS（`https://storage.googleapis.com/`）へ読み替える。
ログインも `gsutil` も要らない。依存は標準ライブラリだけである。
"""

from __future__ import annotations

import hashlib
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dopabae.config import REPO_ROOT, ConnectomeFile, load  # noqa: E402
from dopabae.connectome import sha256_of  # noqa: E402

GCS_HTTPS = "https://storage.googleapis.com/"
CHUNK = 1 << 20


class FetchError(Exception):
    """取得できない、または取得したものの指紋が合わない。"""


def https_url(download_base: str, path: str) -> str:
    """`gs://bucket/dir/` を公開バケットの HTTPS に読み替えて、相対パスをつなぐ。"""
    base = download_base if download_base.endswith("/") else download_base + "/"
    if base.startswith("gs://"):
        base = GCS_HTTPS + base[len("gs://") :]
    if not base.startswith("https://"):
        raise FetchError(f"扱えない取得元です: {download_base}")
    return base + urllib.parse.quote(path)


def matches(path: Path, expected: ConnectomeFile) -> bool:
    """大きさと SHA-256 が `agent.yaml` と一致するか。"""
    return path.stat().st_size == expected.size_bytes and sha256_of(path) == expected.sha256


def download(url: str, dest: Path, expected: ConnectomeFile) -> None:
    """一時ファイルへ取り、指紋が合ったときだけ `dest` へ移す。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as out:
            for block in iter(lambda: response.read(CHUNK), b""):
                out.write(block)
                digest.update(block)
                size += len(block)
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise FetchError(f"取得できません: {url}: {exc}") from exc

    if size != expected.size_bytes or digest.hexdigest() != expected.sha256:
        part.unlink(missing_ok=True)
        raise FetchError(
            f"取得したものの指紋が agent.yaml と合いません: {dest.name}"
            f"（大きさ {size} / 期待 {expected.size_bytes}）"
        )
    part.replace(dest)


def fetch(names: list[str] | None = None) -> int:
    config = load()
    local_dir = REPO_ROOT / config.connectome_local_dir
    wanted = names or list(config.connectome_files)
    unknown = [n for n in wanted if n not in config.connectome_files]
    if unknown:
        print(f"agent.yaml の fly.connectome.files に無い名前です: {unknown}", file=sys.stderr)
        return 1

    for name in wanted:
        expected = config.connectome_files[name]
        dest = local_dir / expected.path
        if dest.exists():
            if matches(dest, expected):
                print(f"{name}: 指紋が合っています。取得しません（{dest}）")
                continue
            print(
                f"{name}: 既にあるファイルの指紋が agent.yaml と合いません: {dest}\n"
                "上書きしません。中身を確かめてから消してください",
                file=sys.stderr,
            )
            return 1
        url = https_url(config.connectome_download_base, expected.path)
        print(f"{name}: {url} から取得します（{expected.size_bytes:,} バイト）")
        try:
            download(url, dest, expected)
        except FetchError as exc:
            print(f"{name}: {exc}", file=sys.stderr)
            return 1
        print(f"{name}: 取得しました。SHA-256 {expected.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(fetch(sys.argv[1:]))
