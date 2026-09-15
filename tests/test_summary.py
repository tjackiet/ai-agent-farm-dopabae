"""日次サマリ。ハエの言葉と観測を混ぜないこと、既にある日誌を壊さないこと。"""

from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from dopabae import journal, summary
from tests import helpers

NOW = helpers.at(helpers.NOW_ISO)  # 2026-09-15T09:00


def record(minute: int, value: str | None = "APPROACH", cage: str = "bought",
           equity: float | None = 1_000_000.0, executed: bool = False, vision: bool = True) -> dict:
    at = NOW - timedelta(days=1) + timedelta(minutes=minute)
    rec: dict = {
        "run_id": at.isoformat(timespec="seconds"),
        "config": {"version": "0.1", "fingerprint": "aaaa1111"},
        "price": 12_000_000.0,
        "position": {"amount": 0.0, "avg_cost": None},
        "account": None if equity is None else {"cash_jpy": equity, "equity_jpy": equity, "drawdown_pct": 0.0},
        "vision": {"sha256": "x" * 64} if vision else None,
        "direction": None if value is None else {"source": "fly", "value": value, "error": None},
        "cage": cage,
        "orders": [{"op": "place", "side": "buy", "executed": executed}] if executed else [],
    }
    return rec


class BuildDailyTest(unittest.TestCase):
    def setUp(self):
        self.cfg = helpers.load_config()

    def build(self, records):
        return summary.build_daily(self.cfg, "2026-09-14", records)

    def test_body_is_left_blank_for_narrate(self):
        text = self.build([record(0)])
        self.assertIn(summary.UNWRITTEN, text)
        self.assertIn("## おれの一日", text)

    def test_numbers_live_outside_the_flys_words(self):
        """数字は観測の節にだけ置く（personality.md「数字の扱い」）。"""
        text = self.build([record(0)])
        words, observed = text.split("## 観測")
        self.assertNotIn("1,000,000", words)
        self.assertIn("1,000,000", observed)

    def test_counts_directions(self):
        text = self.build([record(0, "APPROACH"), record(15, "AVOID"), record(30, "NONE"), record(45, "APPROACH")])
        self.assertIn("| 近づいた（APPROACH） | 2 |", text)
        self.assertIn("| 逃げた（AVOID） | 1 |", text)
        self.assertIn("| じっとしてた（NONE） | 1 |", text)

    def test_counts_unavailable_directions(self):
        failed = record(0, None)
        failed["direction"] = {"source": "fly", "value": None, "error": "見えなかった"}
        text = self.build([failed])
        self.assertIn("| 方向を得られなかった | 1 |", text)

    def test_counts_runs_that_never_consulted(self):
        text = self.build([record(0, None)])
        self.assertIn("方向を諮らなかった", text)

    def test_cage_breakdown_uses_japanese_labels(self):
        text = self.build([record(0, cage="wall"), record(15, cage="wall"), record(30, cage="bought")])
        self.assertIn("| 壁で止まった（wall） | 2 |", text)
        self.assertIn("| 買い指値を置いた（bought） | 1 |", text)

    def test_only_executed_orders_are_counted(self):
        """dry_run の回は注文を出していない。数えたら嘘になる。"""
        text = self.build([record(0, executed=False), record(15, executed=True)])
        self.assertIn("| 出した買い注文 | 1 |", text)

    def test_missing_equity_is_not_invented(self):
        text = self.build([record(0, equity=None)])
        self.assertIn("| 総資産 | 観測できなかった |", text)

    def test_equity_change_is_computed_from_observations(self):
        text = self.build([record(0, equity=1_000_000.0), record(15, equity=1_010_000.0)])
        self.assertIn("| 増減 | +1.00% |", text)

    def test_source_is_cited(self):
        text = self.build([record(0), record(15)])
        self.assertIn("var/memory/decisions/2026-09-14.jsonl", text)
        self.assertIn("2 件", text)

    def test_no_lessons_section(self):
        """ドパバエは反省も計画もしない。学習はシナプス重みだけ（CLAUDE.md）。"""
        text = self.build([record(0)])
        self.assertNotIn("学び", text)
        self.assertNotIn("次は", text)


class EnsureTest(unittest.TestCase):
    def setUp(self):
        self.cfg = helpers.load_config()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_journal(self, day, records):
        for rec in records:
            journal.append(self.cfg, day, rec, root=self.root)

    def test_writes_a_finished_day(self):
        self.write_journal(NOW - timedelta(days=1), [record(0)])
        written = summary.ensure(self.cfg, NOW, self.root)
        self.assertEqual([p.name for p in written], ["2026-09-14.md"])

    def test_does_not_write_today_before_the_write_time(self):
        """途中で書くと、半端な一日をハエに語らせることになる。"""
        self.write_journal(NOW, [record(0)])
        self.assertEqual(summary.ensure(self.cfg, NOW, self.root), [])

    def test_writes_today_after_the_write_time(self):
        self.write_journal(NOW, [record(0)])
        late = helpers.at("2026-09-15T23:55:00+09:00")
        self.assertEqual([p.name for p in summary.ensure(self.cfg, late, self.root)], ["2026-09-15.md"])

    def test_never_overwrites_an_existing_diary(self):
        """埋めた本文を消さない。"""
        self.write_journal(NOW - timedelta(days=1), [record(0)])
        path = summary.ensure(self.cfg, NOW, self.root)[0]
        path.write_text("手で直した本文", encoding="utf-8")
        self.assertEqual(summary.ensure(self.cfg, NOW, self.root), [])
        self.assertEqual(path.read_text(encoding="utf-8"), "手で直した本文")
