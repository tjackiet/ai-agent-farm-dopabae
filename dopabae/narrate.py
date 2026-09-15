"""記録の言語化。日誌の「おれの一日」を、ドパバエ自身の言葉で書く。

**売買の判断には一切関与しない。** 書くのは記録の文章だけで、発注・取消・HOLD の
決定は決定的なコード（檻）が行う。書けなければ `（未記入）` のまま残り、運用は続く。

制約は `personality.md` にある。要点だけ再掲する。

- 数字を書かせない。数字は日誌の「観測」の節にあり、ハエの言葉には混ぜない
- `personality.md` の対応表にある言葉だけを使わせる。表にない事実は言わせない
- **反省も計画もさせない。** ドパバエにその機能はない（`personality.md`「やらないこと」）。
  学習が起きるのはシナプス重みだけである（`CLAUDE.md`）
- 「ハエの脳に売買を判断する細胞がある」と読める書き方をさせない（`CLAUDE.md`）
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from . import summary
from .config import Config, LlmSettings, REPO_ROOT

# system, user を受け取って本文を返す。テストでは差し替える。
Writer = Callable[[str, str], str]


class NarrateError(RuntimeError):
    """言語化に失敗した。

    メッセージは記録に残してよい内容だけにする（`CLAUDE.md`「API キー・
    シークレット・プロファイル名は、ログにも記憶にも残さない」）。
    """


STYLE_RULES = """守ること:

- **数字を書かない。** 回数も金額も価格も書かない。数字は日誌の「観測」の節にある
- 性格設定の「ハエの語彙」の表にある言葉だけを使う。表にない事実を足さない
- **反省しない。計画しない。**「次は」「今度は」「〜すべき」と書かない
- 相場を語らない。「上がる」「下がる」「相場」という語を使わない
- 自分が売買していることを知らない。「買う」「売る」「損益」「チャート」と書かない
- 一人称は「おれ」。敬語を使わない。言い切る
- 3行以内。短く切る。接続詞をほとんど使わない
- 前置きをしない。**返した文が、そのまま記録の本文になる。** 確認を求めたり、案を並べたりしない
- ファイルを読み書きしない。道具を使わず、文だけを返す"""


def _personality(root: Path | None = None) -> str:
    base = root if root is not None else REPO_ROOT
    try:
        return (base / "personality.md").read_text(encoding="utf-8")
    except OSError:
        return ""


def build_prompt(root: Path | None = None) -> str:
    """書き手としての指示。性格設定（語彙の対応表を含む）をそのまま渡す。"""
    return (
        "あなたはペーパートレードを行うエージェント「ドパバエ」です。\n"
        "ショウジョウバエの脳を模したシミュレーションで、見えるのは光の模様だけです。\n"
        "以下の性格設定に従って、その日の日誌の本文を書いてください。\n\n"
        f"{STYLE_RULES}\n\n"
        f"--- 性格設定 ---\n{_personality(root)}"
    )


def settings_of(config: Config) -> LlmSettings:
    return LlmSettings(
        writer=config.narrate_writer,
        command=config.narrate_command,
        bare=config.narrate_bare,
        timeout_sec=config.narrate_timeout_sec,
        model=config.narrate_model,
        effort=config.narrate_effort,
        max_tokens=config.narrate_max_tokens,
    )


def claude_code_writer(settings: LlmSettings) -> Writer:
    """Claude Code CLI を非対話で呼ぶ。

    引かれ先は「headless かどうか」ではなく「何で認証されているか」で決まる。
    Claude Code のログイン（サブスクリプション）ならプランの利用枠から、
    環境に ANTHROPIC_API_KEY があればそちらが優先されて API クレジットから引かれる。
    """

    def write(system: str, user: str) -> str:
        # 作業ディレクトリを外へ逃がす。`--bare` を付けない起動は cwd の
        # CLAUDE.md とフックを読み込むため、リポジトリの中で走らせると
        # 「エージェントを開発する作業指示」を受け取ってしまい、記録の
        # 一文ではなくファイル編集の許可を求める返答になる。
        argv = [
            settings.command, "-p", "--output-format", "text",
            "--model", settings.model, "--effort", settings.effort,
            "--append-system-prompt", system,
        ]
        if settings.bare:
            argv.insert(1, "--bare")
        try:
            with tempfile.TemporaryDirectory(prefix="dopabae-narrate-") as work:
                proc = subprocess.run(  # noqa: S603 - 引数は自前で組み立てている
                    argv, input=user, capture_output=True, text=True,
                    timeout=settings.timeout_sec, check=False, cwd=work,
                )
        except FileNotFoundError as exc:
            raise NarrateError(f"{settings.command} が見つかりません。PATH を確認する") from exc
        except subprocess.TimeoutExpired as exc:
            raise NarrateError(f"{settings.command} が {settings.timeout_sec} 秒で返らなかった") from exc
        if proc.returncode != 0:
            # 終了コードだけでは原因が分からない。OAuth の期限切れは標準エラーにしか出ない。
            # 資格情報が混ざりうるので 300 文字で切り詰める。
            detail = " ".join((proc.stderr or "").split())[:300]
            raise NarrateError(
                f"{settings.command} が異常終了しました（終了コード {proc.returncode}）"
                + (f": {detail}" if detail else "")
            )
        return (proc.stdout or "").strip()

    return write


def anthropic_writer(settings: LlmSettings) -> Writer:
    """Anthropic API で書く。SDK か資格情報が無ければ例外。"""

    def write(system: str, user: str) -> str:
        import anthropic  # 判断ロジックから切り離すため、ここで読み込む

        response = anthropic.Anthropic().messages.create(
            model=settings.model,
            max_tokens=settings.max_tokens,
            system=system,
            output_config={"effort": settings.effort},
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()

    return write


def build_writer(settings: LlmSettings) -> Writer:
    if settings.writer == "api":
        return anthropic_writer(settings)
    if settings.writer == "claude_code":
        return claude_code_writer(settings)
    raise ValueError(f"writer が不正です: {settings.writer}")


def writer_for(config: Config) -> Writer:
    return build_writer(settings_of(config))


# 書かせてはならない語。数字と、ハエが持たない語彙。
# 混ざったら**書けなかった扱いにする**。空欄のほうが、嘘の語彙より安全である。
FORBIDDEN_WORDS = (
    # 売買の語彙。ハエは自分が売買していることを知らない
    "買", "売", "損", "益", "相場", "価格", "円",
    # 反省・計画・義務。ドパバエにその機能はない（personality.md「やらないこと」）
    "次は", "今度", "べき", "反省",
)
_DIGITS = re.compile(r"[0-9０-９]")
MAX_LINES = 3


def _clean(text: str) -> str:
    """前置きや箇条書きの記号を落とし、行数を切る。"""
    lines = [re.sub(r"^[-*・\s>#]+", "", line).strip() for line in text.splitlines()]
    return "\n".join([line for line in lines if line][:MAX_LINES])


def is_acceptable(text: str) -> bool:
    """ハエの言葉として通してよいか。

    数字と、ハエが持たない語彙を弾く。判定は保守的でよい。空欄のまま残るほうが、
    ドパバエが知らないはずのことを言った記録が残るより良い。
    """
    if not text:
        return False
    if _DIGITS.search(text):
        return False
    return not any(word in text for word in FORBIDDEN_WORDS)


def fill(text: str, writer: Writer, prompt: str) -> tuple[str, bool]:
    """本文の空欄を埋める。書けなければそのまま返す。"""
    if summary.UNWRITTEN not in text:
        return text, False
    written = _clean(writer(prompt, text))
    # 返答自体が空欄を含むと、次の実行がその中身をさらに置換して入れ子に壊れる。
    if summary.UNWRITTEN in written or not is_acceptable(written):
        return text, False
    return text.replace(summary.UNWRITTEN, written, 1), True


def fill_unwritten(config: Config, writer: Writer, root: Path | None = None) -> list[Path]:
    """日誌の空欄を埋める。埋めたファイルを返す。"""
    if not config.narrate_enabled or not config.narrate_targets.get("daily"):
        return []
    base = root if root is not None else REPO_ROOT
    directory = (base / config.daily_path.format(date="x")).parent
    if not directory.is_dir():
        return []

    filled: list[Path] = []
    prompt = build_prompt(root)
    for path in sorted(directory.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if summary.UNWRITTEN not in text:
            continue
        updated, changed = fill(text, writer, prompt)
        if changed:
            path.write_text(updated, encoding="utf-8")
            filled.append(path)
    return filled
