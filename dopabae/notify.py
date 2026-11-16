"""動きがあったときの通知（Discord）。

`scripts/run_arms.py` が、全部の腕を回し終えたあとに 1 回だけ呼ぶ。腕ごとに文面を作り、
1 通にまとめて送る。腕のプロセス（`dopabae.run`）はこのモジュールを使わない。
**通知は判断に影響させない。** 腕の発注と記録が済んでから送り、送れなくても何も変えない。

送るもの（`agent.yaml` の `notify`）:

| きっかけ | 中身 |
| --- | --- |
| 約定 | 腕の `paper tick` が返した約定（観測値）。`[腕] 約定 買い 価格 × 数量` |
| 失敗の連続 | 腕の回が続けて失敗したとき。閾値ちょうどと、その後は決まった回数ごと |
| 振り返り | 決めた時刻をまたいだ最初の回。腕ごとに 1 行と、ドパバエの一言 |

発注・取消は送らない。檻は方向が続くあいだ毎回指値を置き直すので、15 分ごとに
何本ぶんも通知が出てしまう。

Webhook の URL は環境変数からのみ読む。リポジトリにも判断ログにも書かない
（CLAUDE.md「API キー・シークレット・プロファイル名は、ログにも記憶にも残さない」）。
送れなかった理由にも URL を入れない。

ナンピノニクスの `notify.py` の送りかた（User-Agent、1900 字、例外を投げない）を写した。
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from . import timeutil
from . import config as config_module, narrate as narrate_module, performance as performance_module
from .config import REPO_ROOT, ConfigError
from .direction import AVOID, NONE, APPROACH

Poster = Callable[[str, str, int], None]

# 名乗らないと Discord の前段にいる Cloudflare が弾く（403 / error code 1010）。
# urllib の既定は `Python-urllib/3.x` で、これが署名として拒否される（ナンピノニクスで確かめた）。
USER_AGENT = "Dopabae/1.0 (paper-trading agent)"
# Discord の本文の上限は 2000 字。余白を残して切る。
MAX_CHARS = 1900
# 腕の回が「失敗」だったとみなす run_arms の状態。skipped は失敗に数えない（前の回が走っているだけ）。
FAILED_STATUSES = ("failed", "timeout")
_SLOT = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


@dataclass(frozen=True)
class Settings:
    """`agent.yaml` の `notify`。既定値は持たない。欠けていれば ConfigError。"""

    enabled: bool
    webhook_env: str
    timeout_sec: int
    fill: bool
    error: bool
    error_streak: int
    streak_repeat_every: int
    report_at: tuple[str, ...]
    state_path: str


def _need(raw: Any, key: str) -> Any:
    if not isinstance(raw, dict) or key not in raw:
        raise ConfigError(f"agent.yaml の notify.{key} がありません")
    return raw[key]


def load_settings(raw: Mapping[str, Any]) -> Settings:
    """`agent.yaml` 全体から `notify` を読む。腕の設定（arms/*.yaml）は読まない。"""
    section = raw.get("notify") if isinstance(raw, Mapping) else None
    if not isinstance(section, dict):
        raise ConfigError("agent.yaml に notify がありません")
    events = _need(section, "events")
    if not isinstance(events, dict):
        raise ConfigError("agent.yaml の notify.events は辞書です")
    for key in ("fill", "error"):
        if not isinstance(events.get(key), bool):
            raise ConfigError(f"agent.yaml の notify.events.{key} は true / false です")
    slots = _need(section, "report_at")
    if not isinstance(slots, list) or not all(isinstance(s, str) and _SLOT.match(s) for s in slots):
        raise ConfigError('agent.yaml の notify.report_at は "HH:MM" の並びです')
    enabled = _need(section, "enabled")
    webhook_env = _need(section, "webhook_env")
    timeout = _need(section, "timeout_sec")
    streak = _need(section, "error_streak")
    repeat = _need(section, "streak_repeat_every")
    state_path = _need(section, "state_path")
    if not isinstance(enabled, bool):
        raise ConfigError("agent.yaml の notify.enabled は true / false です")
    if not isinstance(webhook_env, str) or not webhook_env.strip():
        raise ConfigError("agent.yaml の notify.webhook_env は環境変数の名前です")
    for name, value, low in (("timeout_sec", timeout, 1), ("error_streak", streak, 1), ("streak_repeat_every", repeat, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise ConfigError(f"agent.yaml の notify.{name} は {low} 以上の整数です")
    if not isinstance(state_path, str) or not state_path.strip():
        raise ConfigError("agent.yaml の notify.state_path はファイルの場所です")
    return Settings(
        enabled=enabled,
        webhook_env=webhook_env.strip(),
        timeout_sec=timeout,
        fill=events["fill"],
        error=events["error"],
        error_streak=streak,
        streak_repeat_every=repeat,
        report_at=tuple(slots),
        state_path=state_path,
    )


# --- 前回の様子（run_arms の単位で 1 つ持つ） ---


@dataclass(frozen=True)
class Previous:
    """前回の通知の様子。差分をとるためだけに使い、判断には使わない。"""

    at: datetime | None
    failures: dict[str, int] = field(default_factory=dict)
    # 前回の振り返りを送った時刻。一言の材料は、ここから今回までの様子にする。
    reported_at: datetime | None = None


def _read_time(value: Any, tz_name: str) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return timeutil.from_iso(value, tz_name)
    except ValueError:
        return None


def read_previous(path: Path, tz_name: str) -> Previous:
    if not path.is_file():
        return Previous(at=None)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return Previous(at=None)
    if not isinstance(document, dict):
        return Previous(at=None)
    raw = document.get("failures")
    failures = (
        {str(k): v for k, v in raw.items() if isinstance(v, int) and not isinstance(v, bool) and v >= 0}
        if isinstance(raw, dict)
        else {}
    )
    return Previous(
        at=_read_time(document.get("at"), tz_name),
        failures=failures,
        reported_at=_read_time(document.get("reported_at"), tz_name),
    )


def write_previous(
    path: Path, at: datetime, failures: Mapping[str, int], reported_at: datetime | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(
        {
            "at": timeutil.to_iso(at),
            "reported_at": timeutil.to_iso(reported_at) if reported_at is not None else None,
            "failures": dict(sorted(failures.items())),
        },
        allow_unicode=True, sort_keys=False,
    )
    path.write_text(
        "# 通知の前回の様子（scripts/run_arms.py が書く。Git 管理外。判断には使わない）\n" + body,
        encoding="utf-8",
    )


# --- この回の腕の結果 ---


@dataclass(frozen=True)
class ArmRound:
    """この回の腕 1 本ぶんの結果。`scripts/run_arms.py` の Result から作る。"""

    name: str
    status: str  # ok | skipped | failed | timeout
    fills: tuple[dict, ...] = ()
    error: str | None = None  # 腕の回の error（観測・発注の失敗）か、プロセスの失敗の理由

    @property
    def failed(self) -> bool:
        return self.status in FAILED_STATUSES or (self.status == "ok" and bool(self.error))


def next_failures(previous: Mapping[str, int], rounds: Sequence[ArmRound]) -> dict[str, int]:
    """腕ごとに、続けて失敗した回数を進める。skipped の腕は数えたまま据え置く。"""
    counts = {name: n for name, n in previous.items() if any(r.name == name for r in rounds)}
    for r in rounds:
        if r.status == "skipped":
            counts[r.name] = previous.get(r.name, 0)
        elif r.failed:
            counts[r.name] = previous.get(r.name, 0) + 1
        else:
            counts[r.name] = 0
    return counts


def streak_due(streak: int, threshold: int, repeat_every: int) -> bool:
    """閾値に達した回と、そのあとは `repeat_every` 回ごとにだけ真。

    閾値を超えている間ずっと送ると、15 分ごとの運用では 1 日 96 通になる
    （ナンピノニクスで 11 日続いた凍結は、およそ 1,000 通ぶんに相当した）。
    """
    if streak <= 0 or streak < threshold:
        return False
    if repeat_every <= 0:
        return streak == threshold
    return (streak - threshold) % repeat_every == 0


def _jpy(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{round(float(value)):,}"


def _btc(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def _pct(value: float | None, signed: bool = True) -> str:
    if value is None:
        return "—"
    return f"{value:+.2f}%" if signed else f"{value:.2f}%"


def fill_lines(name: str, fills: Sequence[dict]) -> list[str]:
    """約定を 1 件 1 行で。値は `paper tick` の応答そのまま。"""
    lines = []
    for fill in fills:
        side = {"buy": "買い", "sell": "売り"}.get(fill.get("side"), "?")
        price = fill.get("fillPrice")
        amount = fill.get("amount")
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (price, amount))
        notional = float(price) * float(amount) if numeric else None
        lines.append(f"[{name}] 約定 {side} {_jpy(price)} × {_btc(amount)}（{_jpy(notional)} JPY）")
    return lines


def build_messages(settings: Settings, rounds: Sequence[ArmRound], failures: Mapping[str, int]) -> list[str]:
    """この回に知らせることを組み立てる。何も無ければ空。"""
    messages: list[str] = []
    if settings.fill:
        for r in rounds:
            messages.extend(fill_lines(r.name, r.fills))
    if settings.error:
        for r in rounds:
            streak = failures.get(r.name, 0)
            if r.failed and streak_due(streak, settings.error_streak, settings.streak_repeat_every):
                messages.append(f"[{r.name}] {streak}回続けて失敗しています: {r.error or r.status}")
    return messages


# --- 振り返り ---


def crossed_report_times(settings: Settings, previous_at: datetime | None, now: datetime) -> list[str]:
    """前回の実行から今回までに、振り返りの時刻をまたいだか。

    止まっていて時刻を過ぎても、動いた最初の回で送る。日をまたいで止まっていた
    ときも拾えるよう、前回の日付から今日までの時刻を見る（最大 3 日ぶん）。
    前回が分からない（初めての回）ときは送らない。
    """
    if previous_at is None or previous_at >= now:
        return []
    crossed: list[str] = []
    day = max(previous_at.date(), (now - timedelta(days=2)).date())
    while day <= now.date():
        for slot in settings.report_at:
            hour, minute = (int(part) for part in slot.split(":"))
            target = now.replace(year=day.year, month=day.month, day=day.day,
                                 hour=hour, minute=minute, second=0, microsecond=0)
            if previous_at < target <= now:
                crossed.append(f"{day.isoformat()} {slot}")
        day += timedelta(days=1)
    return crossed


@dataclass(frozen=True)
class ArmSummary:
    """振り返りの 1 行ぶん。すべて判断ログの観測値から。取れない値は None（「—」と出す）。"""

    name: str
    equity_jpy: float | None
    total_pct: float | None
    change_24h_pct: float | None
    drawdown_pct: float | None
    fills_24h: int | None
    position_btc: float | None
    direction: str | None


def _at(record: dict, tz_name: str) -> datetime | None:
    run_id = record.get("run_id")
    if not isinstance(run_id, str):
        return None
    try:
        return timeutil.from_iso(run_id, tz_name)
    except ValueError:
        return None


def _equity(record: dict) -> float | None:
    account = record.get("account")
    value = account.get("equity_jpy") if isinstance(account, dict) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def summarize_arm(name: str, records: Sequence[dict], initial_jpy: float, now: datetime, tz_name: str) -> ArmSummary:
    """腕の判断ログから、振り返りの 1 行を作る。総資産は `paper assets` 由来の観測値で、組み立て直さない。"""
    dated = [(at, r) for r in records if (at := _at(r, tz_name)) is not None and at <= now]
    dated.sort(key=lambda pair: pair[0])
    points = [(at, eq) for at, r in dated if (eq := _equity(r)) is not None]
    equity = points[-1][1] if points else None
    total = (equity - initial_jpy) / initial_jpy * 100 if equity is not None and initial_jpy > 0 else None

    since = now - timedelta(hours=24)
    recent = [eq for at, eq in points if at >= since]
    change = (recent[-1] - recent[0]) / recent[0] * 100 if len(recent) >= 2 and recent[0] > 0 else None

    peak = max((eq for _, eq in points), default=None)
    drawdown = (peak - equity) / peak * 100 if peak and equity is not None else None

    # 約定を記録し始める前の回が 24 時間に入っていたら、数は分からない。数えない。
    window = [r for at, r in dated if at >= since]
    fills_24h = None
    if window and all(isinstance(r.get("fills"), list) for r in window):
        fills_24h = sum(len(r["fills"]) for r in window)

    last = dated[-1][1] if dated else {}
    position = last.get("position")
    amount = position.get("amount") if isinstance(position, dict) else None
    direction = last.get("direction")
    value = direction.get("value") if isinstance(direction, dict) else None
    return ArmSummary(
        name=name,
        equity_jpy=equity,
        total_pct=total,
        change_24h_pct=change,
        drawdown_pct=drawdown,
        fills_24h=fills_24h,
        position_btc=float(amount) if isinstance(amount, (int, float)) and not isinstance(amount, bool) else None,
        direction=value if value in (APPROACH, AVOID, NONE) else None,
    )


def report_line(summary: ArmSummary) -> str:
    position = (
        "建玉なし" if summary.position_btc == 0
        else f"建玉 {_btc(summary.position_btc)} BTC" if summary.position_btc is not None
        else "建玉 —"
    )
    fills = "—" if summary.fills_24h is None else str(summary.fills_24h)
    return (
        f"[{summary.name}] 総資産 {_jpy(summary.equity_jpy)} JPY ({_pct(summary.total_pct)}) / "
        f"24時間 {_pct(summary.change_24h_pct)} / 最高からの下落 {_pct(summary.drawdown_pct, signed=False)} / "
        f"24時間の約定 {fills}件 / {position} / 方向 {summary.direction or '—'}"
    )


def build_report(now: datetime, summaries: Sequence[ArmSummary]) -> str:
    lines = [f"**{timeutil.to_iso(now)} の振り返り**"]
    lines.extend(report_line(s) for s in summaries)
    lines.append("（総資産・建玉は各腕の判断ログの観測値。paper assets / paper pnl 由来）")
    return "\n".join(lines)


def build_facts(arms: Sequence[tuple[str, Sequence[dict]]], since: datetime | None, now: datetime, tz_name: str) -> str:
    """一言を書かせるための材料。ハエ版の腕の、前回の振り返りから今回までの様子。

    `personality.md` の「ハエの語彙」にある事実だけを渡す（約定や総資産はハエの知らないことなので渡さない）。
    数字はここにだけ置く。書き手には数字を書かせない（`narrate.is_acceptable` が弾く）。
    """
    start = since if since is not None else now - timedelta(hours=12)
    lines = ["前回の振り返りから今回までの、ハエ版の腕（同じ脳の模型を、入力の条件だけ変えて並べたもの）ごとの様子:"]
    for name, records in arms:
        counts = {APPROACH: 0, AVOID: 0, NONE: 0}
        unseen = walls = rewards = aversives = 0
        for r in records:
            at = _at(r, tz_name)
            if at is None or not start < at <= now:
                continue
            direction = r.get("direction")
            value = direction.get("value") if isinstance(direction, dict) else None
            if value in counts:
                counts[value] += 1
            else:
                unseen += 1
            if r.get("cage") in ("wall", "cooldown"):
                walls += 1
            reinforcement = r.get("reinforcement")
            stimulus = reinforcement.get("stimulus") if isinstance(reinforcement, dict) else None
            rewards += stimulus == "reward"
            aversives += stimulus == "aversive"
        lines.append(
            f"- {name}: 近づいた {counts[APPROACH]} 回 / 逃げた {counts[AVOID]} 回 / じっとしてた {counts[NONE]} 回 / "
            f"見えなかった {unseen} 回 / 壁 {walls} 回 / ドパきた {rewards} 回 / イヤなのきた {aversives} 回"
        )
    return "\n".join(lines)


# --- 送る ---


def _post(url: str, content: str, timeout: int) -> None:
    payload = json.dumps({"content": content}).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json", "User-Agent": USER_AGENT}
    )
    urllib.request.urlopen(request, timeout=timeout).close()  # noqa: S310 - 宛先は運用者が環境変数で決める


def webhook_url(settings: Settings, environ: Mapping[str, str] | None = None) -> str | None:
    """環境変数から URL を読む。無ければ通知しない。"""
    value = (environ if environ is not None else os.environ).get(settings.webhook_env, "").strip()
    return value or None


def send(
    settings: Settings,
    messages: Sequence[str],
    poster: Poster | None = None,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """まとめて 1 通にして送る。失敗したら理由を返す（例外は投げない。URL は出さない）。"""
    if not settings.enabled or not messages:
        return None
    url = webhook_url(settings, environ)
    if url is None:
        return f"{settings.webhook_env} が設定されていないため通知しません"
    body = "\n".join(messages)[:MAX_CHARS]
    try:
        (poster or _post)(url, body, settings.timeout_sec)
    except urllib.error.HTTPError as exc:
        # 状態コードは秘密ではなく、原因の切り分けに要る。
        return f"通知を送れませんでした: HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - 通知の失敗で定期実行を止めない（http.client の例外も含む）
        return f"通知を送れませんでした: {type(exc).__name__}"
    return None


def state_file(settings: Settings, root: Path | None = None) -> Path:
    base = root if root is not None else REPO_ROOT
    return base / settings.state_path


def _quote(remark: str) -> str:
    return "\n".join(f"> {line}" for line in remark.splitlines())


def after_round(
    arms: Sequence[tuple[str, Path]],
    rounds: Sequence[ArmRound],
    *,
    config: Any = None,
    poster: Poster | None = None,
    writer: narrate_module.Writer | None = None,
    environ: Mapping[str, str] | None = None,
    now: datetime | None = None,
    root: Path | None = None,
) -> str | None:
    """全部の腕を回し終えたあとに 1 回だけ呼ぶ。送れなかった理由などを返す。**例外は投げない。**

    `arms` は (腕の名前, 腕の agent.yaml) の並び。設定（`notify` と `narrate`）は
    リポジトリ直下の `agent.yaml`（`config`）から読み、腕の設定からは読まない。
    """
    warnings: list[str] = []
    try:
        cfg = config if config is not None else config_module.load()
        settings = load_settings(cfg.raw)
    except ConfigError as exc:
        return f"通知の設定を読めません: {exc}"
    if not settings.enabled:
        return None
    tz = cfg.timezone
    now = now if now is not None else timeutil.now(tz)
    path = state_file(settings, root)
    previous = read_previous(path, tz)
    failures = next_failures(previous.failures, rounds)
    messages = build_messages(settings, rounds, failures)
    reported_at = previous.reported_at

    if crossed_report_times(settings, previous.at, now):
        reported_at = now
        summaries: list[ArmSummary] = []
        fly_arms: list[tuple[str, Sequence[dict]]] = []
        for name, config_path in arms:
            try:
                arm_cfg = config_module.load(config_path)
                records = performance_module.all_records(arm_cfg, root)
            except (ConfigError, OSError) as exc:
                warnings.append(f"{name} の判断ログを読めません: {type(exc).__name__}")
                summaries.append(ArmSummary(name, None, None, None, None, None, None, None))
                continue
            summaries.append(summarize_arm(name, records, arm_cfg.initial_jpy, now, tz))
            if arm_cfg.direction_source == "fly":
                fly_arms.append((name, records))
        report = build_report(now, summaries)
        if fly_arms and cfg.narrate_enabled and cfg.narrate_targets.get("report"):
            try:
                # 一言を書く claude には Webhook を見せない（腕のプロセスと同じ扱い）。
                remark = narrate_module.comment_report(
                    cfg,
                    writer or narrate_module.writer_for(cfg, hidden_env=(settings.webhook_env,)),
                    build_facts(fly_arms, previous.reported_at, now, tz),
                    root,
                )
                if remark:
                    report = f"{report}\n{_quote(remark)}"
            except narrate_module.NarrateError as exc:
                warnings.append(f"一言を書けませんでした: {exc}")
                if narrate_module.needs_login(str(exc)):
                    report = f"{report}\nclaude のログインが切れていて、一言を書けませんでした。{narrate_module.LOGIN_HINT}"
            except Exception as exc:  # noqa: BLE001 - 一言が書けなくても振り返りは送る
                warnings.append(f"一言を書けませんでした: {type(exc).__name__}")
        messages.append(report)

    failure = send(settings, messages, poster=poster, environ=environ)
    if failure:
        warnings.append(failure)
    try:
        write_previous(path, now, failures, reported_at)
    except OSError as exc:
        warnings.append(f"通知の前回の様子を書けません: {type(exc).__name__}")
    return " / ".join(warnings) or None
