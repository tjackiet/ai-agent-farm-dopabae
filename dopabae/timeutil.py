"""時刻の扱い。判断に使う時刻はすべて tz-aware で持つ。"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def tz(name: str) -> ZoneInfo:
    return ZoneInfo(name)


def now(tz_name: str) -> datetime:
    return datetime.now(tz(tz_name))


def from_iso(value: str, tz_name: str) -> datetime:
    """CLI が返す ISO8601（UTC の Z 表記）を tz-aware に読む。"""
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"タイムゾーンのない時刻は受け付けません: {value}")
    return parsed.astimezone(tz(tz_name))


def from_epoch_ms(value: float, tz_name: str) -> datetime:
    return datetime.fromtimestamp(value / 1000, timezone.utc).astimezone(tz(tz_name))


def to_iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def date_key(value: datetime) -> str:
    """memory/decisions/{date}.jsonl などに使う YYYY-MM-DD。"""
    return value.strftime("%Y-%m-%d")
