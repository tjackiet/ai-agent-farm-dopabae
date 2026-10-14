"""報酬・嫌悪を決める（Phase 5）。決済が確定した回だけ、二重に出さず、閾値の内側では出さない。"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from dopabae import reward
from dopabae.state import Trade

JST = timezone(timedelta(hours=9))
T0 = datetime(2026, 10, 13, 12, 0, tzinfo=JST)


def trade(minutes: int, side: str, amount: str, price: int) -> Trade:
    return Trade(
        id=f"t{minutes}{side}",
        side=side,
        order_type="limit",
        amount=Decimal(amount),
        fill_price=Decimal(price),
        fee_quote=Decimal(0),
        filled_at=T0 + timedelta(minutes=minutes),
    )


# 1 回目: 0.0225 を買って +664 円で売り切る。2 回目: 0.0150 を買って −1,000 円で売り切る。3 回目: 買ったまま
TRADES = [
    trade(0, "buy", "0.0075", 13223276),
    trade(15, "buy", "0.0075", 13245164),
    trade(30, "buy", "0.0075", 13245687),
    trade(45, "sell", "0.0225", 13267534),
    trade(60, "buy", "0.0150", 13300000),
    trade(75, "sell", "0.0150", 13233334),
    trade(90, "buy", "0.0075", 13292494),
]


class JudgeTest(unittest.TestCase):
    def test_first_time_uses_no_past_settlement(self):
        """台帳が無い回（学習を入れた最初の回）は、それより前の決済を使わない。"""
        out = reward.judge(TRADES, False, 500)
        self.assertEqual(out.stimulus, reward.NONE)
        self.assertIsNone(out.realized_pnl_jpy)
        self.assertEqual(out.last_closed_at, (T0 + timedelta(minutes=75)).isoformat())

    def test_reward_and_aversion_by_threshold(self):
        first = reward.judge(TRADES[:4], None, 500)
        self.assertEqual(first.stimulus, reward.REWARD)
        self.assertAlmostEqual(first.realized_pnl_jpy, 664, delta=1)
        self.assertEqual(first.closed_rounds, 1)
        second = reward.judge(TRADES[:6], first.last_closed_at, 500)
        self.assertEqual(second.stimulus, reward.AVERSIVE)
        self.assertAlmostEqual(second.realized_pnl_jpy, -1000, delta=1)
        inside = reward.judge(TRADES[:4], None, 1000)
        self.assertEqual(inside.stimulus, reward.NONE)
        self.assertAlmostEqual(inside.realized_pnl_jpy, 664, delta=1)

    def test_same_settlement_is_not_used_twice(self):
        first = reward.judge(TRADES, None, 500)
        again = reward.judge(TRADES, first.last_closed_at, 500)
        self.assertEqual(again.stimulus, reward.NONE)
        self.assertEqual(again.closed_rounds, 0)
        self.assertEqual(again.last_closed_at, first.last_closed_at)

    def test_open_position_is_not_rewarded(self):
        """含み損益では出さない。買ったままのラウンドは使わない。"""
        out = reward.judge(TRADES[:3], None, 1)
        self.assertEqual(out.stimulus, reward.NONE)
        self.assertEqual(out.closed_rounds, 0)

    def test_several_settlements_are_summed(self):
        out = reward.judge(TRADES, None, 100)
        self.assertEqual(out.closed_rounds, 2)
        self.assertAlmostEqual(out.realized_pnl_jpy, -336, delta=1)
        self.assertEqual(out.stimulus, reward.AVERSIVE)


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.dir = TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "fly" / "reward-ledger.json"

    def test_round_trip(self):
        self.assertIs(reward.load_ledger(self.path), False)
        reward.save_ledger(self.path, None)
        self.assertIsNone(reward.load_ledger(self.path))
        reward.save_ledger(self.path, T0.isoformat())
        self.assertEqual(reward.load_ledger(self.path), T0.isoformat())

    def test_broken_ledger_is_an_error(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{", encoding="utf-8")
        with self.assertRaises(reward.LedgerError):
            reward.load_ledger(self.path)
        self.path.write_text('{"other": 1}', encoding="utf-8")
        with self.assertRaises(reward.LedgerError):
            reward.load_ledger(self.path)


if __name__ == "__main__":
    unittest.main()
