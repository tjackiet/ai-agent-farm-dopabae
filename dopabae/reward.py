"""報酬・嫌悪を決める（Phase 5）。決定的なコードで、ハエは関わらない。

- **決済が確定した回だけ**出す（CLAUDE.md）。含み損益では出さない
- 前の観測のあとに閉じた建玉のラウンド（買い始めてから売り切るまで。`state.rounds`）の確定損益を足し、
  `threshold_jpy` 以上の利益なら報酬、`threshold_jpy` 以上の損なら嫌悪、間なら何も出さない
- どの決済まで使ったかを台帳（`fly.reward.ledger_path`）に残し、同じ決済で二度出さない
- 台帳が無いとき（学習を入れた最初の回）は、それより前の決済を使わない。台帳を作るだけで何も出さない
- 約定履歴が読めない回は何も出さず、損益は null のまま残す

ハエ（網）に渡すのは「報酬 / 嫌悪 / なし」だけで、金額は渡さない。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from . import state as state_module

NONE = "none"
REWARD = "reward"
AVERSIVE = "aversive"


class LedgerError(Exception):
    """台帳が読めない。報酬・嫌悪を出さずに HOLD にする（二重に出すよりよい）。"""


@dataclass(frozen=True)
class Reinforcement:
    """その回に与える刺激と、その根拠。観測した値だけを持つ。"""

    stimulus: str  # none / reward / aversive
    realized_pnl_jpy: float | None  # 根拠にした確定損益の合計。使った決済が無ければ null
    closed_rounds: int  # 今回はじめて使う、閉じたラウンドの数
    last_closed_at: str | None  # 使った決済のうち最後のもの（ISO 8601）。台帳に書く
    threshold_jpy: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "stimulus": self.stimulus,
            "realized_pnl_jpy": self.realized_pnl_jpy,
            "closed_rounds": self.closed_rounds,
            "last_closed_at": self.last_closed_at,
            "threshold_jpy": self.threshold_jpy,
            "reason": self.reason,
        }


def load_ledger(path: Path) -> str | None | bool:
    """台帳を読む。無ければ False、まだ決済を使っていなければ None、あれば最後に使った決済の時刻。"""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LedgerError(f"報酬の台帳が読めません: {path}: {exc}") from exc
    if not isinstance(data, dict) or "last_closed_at" not in data:
        raise LedgerError(f"報酬の台帳の形が違います: {path}")
    value = data["last_closed_at"]
    if value is not None and not isinstance(value, str):
        raise LedgerError(f"報酬の台帳の last_closed_at が文字列ではありません: {path}")
    return value


def save_ledger(path: Path, last_closed_at: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps({"last_closed_at": last_closed_at}, ensure_ascii=False) + "\n", encoding="utf-8")
    part.replace(path)


def judge(
    trades: Sequence[state_module.Trade],
    ledger: str | None | bool,
    threshold_jpy: float,
    unit: Decimal | None = None,
) -> Reinforcement:
    """約定履歴と台帳から、その回の刺激を決める。"""
    closed = [r for r in state_module.rounds(trades, unit) if r.is_closed]
    latest = max((r.closed_at for r in closed), default=None)
    latest_iso = latest.isoformat() if latest is not None else None
    if ledger is False:
        return Reinforcement(NONE, None, 0, latest_iso, threshold_jpy, "学習を入れた最初の回。これより前の決済は使わない")
    since = datetime.fromisoformat(ledger) if ledger else None
    new = [r for r in closed if since is None or r.closed_at > since]
    if not new:
        return Reinforcement(NONE, None, 0, ledger or None, threshold_jpy, "前の観測のあとに確定した決済は無い")
    pnl = float(sum((r.realized_pnl_jpy for r in new), Decimal(0)))
    last = max(r.closed_at for r in new).isoformat()
    if pnl >= threshold_jpy:
        stimulus, rule = REWARD, f"確定損益 {pnl:+,.0f} 円 ≥ +{threshold_jpy:,.0f} 円"
    elif pnl <= -threshold_jpy:
        stimulus, rule = AVERSIVE, f"確定損益 {pnl:+,.0f} 円 ≤ −{threshold_jpy:,.0f} 円"
    else:
        stimulus, rule = NONE, f"確定損益 {pnl:+,.0f} 円は ±{threshold_jpy:,.0f} 円の内側"
    return Reinforcement(stimulus, pnl, len(new), last, threshold_jpy, f"決済 {len(new)} 回: {rule}")
