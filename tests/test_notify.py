"""通知。判断に影響させないこと、URL を出さないこと、うるさくしないこと。外には出ない。"""

from __future__ import annotations

import copy
import dataclasses
import http.client
import json
import tempfile
import unittest
import urllib.error
from datetime import timedelta
from pathlib import Path
from unittest import mock

import yaml

from dopabae import narrate, notify
from dopabae.config import ConfigError
from dopabae.direction import APPROACH, AVOID, NONE
from tests import helpers

TZ = helpers.TZ
NOW = helpers.at("2026-09-15T09:05:00+09:00")
HOOK = "https://example.invalid/hook"  # .invalid は名前解決されない予約ドメイン
ENV = {"DOPABAE_DISCORD_WEBHOOK_URL": HOOK}


def settings(**overrides) -> notify.Settings:
    base = notify.load_settings(helpers.load_config().raw)
    return dataclasses.replace(base, **overrides)


def fill(side="buy", price=14699000, amount=0.0068) -> dict:
    return {"id": "f1", "side": side, "fillPrice": price, "amount": amount}


def record(at: str, equity=None, direction=NONE, fills=None, amount=0.0, cage="still", stimulus=None) -> dict:
    row = {
        "run_id": at,
        "account": None if equity is None else {"cash_jpy": equity, "equity_jpy": equity, "drawdown_pct": 0.0},
        "position": {"amount": amount, "avg_cost": None},
        "direction": None if direction is None else {"source": "fly", "value": direction},
        "cage": cage,
        "reinforcement": None if stimulus is None else {"stimulus": stimulus},
    }
    if fills is not None:
        row["fills"] = fills
    return row


class SettingsTest(unittest.TestCase):
    def test_repo_config_has_every_key(self):
        s = notify.load_settings(helpers.load_config().raw)
        self.assertEqual(s.webhook_env, "DOPABAE_DISCORD_WEBHOOK_URL")
        self.assertTrue(s.fill and s.error)
        self.assertEqual(s.report_at, ("09:00", "21:00"))

    def test_missing_or_broken_values_stop_loading(self):
        raw = helpers.load_config().raw
        for path, value in (
            (("notify", "timeout_sec"), None),
            (("notify", "report_at"), ["9:00"]),
            (("notify", "events", "fill"), "yes"),
            (("notify", "error_streak"), 0),
        ):
            broken = copy.deepcopy(raw)
            target = broken
            for key in path[:-1]:
                target = target[key]
            if value is None:
                del target[path[-1]]
            else:
                target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ConfigError):
                notify.load_settings(broken)
        with self.assertRaises(ConfigError):
            notify.load_settings({k: v for k, v in raw.items() if k != "notify"})

    def test_notify_does_not_change_the_fingerprint(self):
        """腕の設定の指紋は strategy / risk / direction / fly だけ。通知の値を変えても腕を作り直さなくてよい。"""
        from dopabae import config as config_module

        raw = helpers.load_config().raw
        changed = copy.deepcopy(raw)
        changed["notify"]["report_at"] = ["08:00"]
        self.assertEqual(config_module.fingerprint(raw), config_module.fingerprint(changed))


class StreakTest(unittest.TestCase):
    def test_threshold_then_every_n(self):
        due = [n for n in range(1, 60) if notify.streak_due(n, 3, 24)]
        self.assertEqual(due, [3, 27, 51])

    def test_zero_repeat_means_once(self):
        self.assertEqual([n for n in range(1, 10) if notify.streak_due(n, 3, 0)], [3])

    def test_failures_count_up_reset_and_hold_on_skip(self):
        rounds = [
            notify.ArmRound("a", "failed", error="終了コード 1"),
            notify.ArmRound("b", "ok"),
            notify.ArmRound("c", "skipped"),
            notify.ArmRound("d", "ok", error="観測に失敗した"),
        ]
        counts = notify.next_failures({"a": 2, "b": 5, "c": 4, "gone": 9}, rounds)
        self.assertEqual(counts, {"a": 3, "b": 0, "c": 4, "d": 1})


class MessageTest(unittest.TestCase):
    def test_fill_lines_carry_the_arm_name(self):
        lines = notify.fill_lines("fly-hs1", [fill(), fill(side="sell", price=None)])
        self.assertEqual(lines[0], "[fly-hs1] 約定 買い 14,699,000 × 0.0068（99,953 JPY）")
        self.assertEqual(lines[1], "[fly-hs1] 約定 売り — × 0.0068（— JPY）")

    def test_error_is_sent_only_when_due(self):
        s = settings()
        rounds = [notify.ArmRound("fly", "failed", error="終了コード 1")]
        self.assertEqual(notify.build_messages(s, rounds, {"fly": 2}), [])
        self.assertEqual(notify.build_messages(s, rounds, {"fly": 3}), ["[fly] 3回続けて失敗しています: 終了コード 1"])

    def test_events_can_be_turned_off(self):
        s = settings(fill=False, error=False)
        rounds = [notify.ArmRound("fly", "failed", fills=(fill(),), error="x")]
        self.assertEqual(notify.build_messages(s, rounds, {"fly": 3}), [])

    def test_orders_are_never_sent(self):
        """檻は方向が続くあいだ毎回指値を置き直す。発注・取消を送ると 15 分ごとにうるさくなる。"""
        rounds = [notify.ArmRound("fly", "ok")]
        self.assertEqual(notify.build_messages(settings(), rounds, {"fly": 0}), [])


class ReportTimeTest(unittest.TestCase):
    def test_crossing_a_slot(self):
        prev = helpers.at("2026-09-15T08:50:00+09:00")
        self.assertEqual(notify.crossed_report_times(settings(), prev, NOW), ["2026-09-15 09:00"])
        self.assertEqual(notify.crossed_report_times(settings(), NOW - timedelta(minutes=1), NOW), [])

    def test_first_round_sends_nothing(self):
        self.assertEqual(notify.crossed_report_times(settings(), None, NOW), [])

    def test_a_stop_over_midnight_is_still_noticed(self):
        prev = helpers.at("2026-09-14T20:00:00+09:00")
        self.assertEqual(
            notify.crossed_report_times(settings(), prev, NOW), ["2026-09-14 21:00", "2026-09-15 09:00"]
        )

    def test_long_stop_is_capped(self):
        prev = helpers.at("2026-09-01T00:00:00+09:00")
        self.assertEqual(len(notify.crossed_report_times(settings(), prev, NOW)), 5)


class SummaryTest(unittest.TestCase):
    def test_values_come_from_observed_equity(self):
        records = [
            record("2026-09-14T08:00:00+09:00", equity=1_010_000),  # 24 時間より前の最高値
            record("2026-09-14T10:00:00+09:00", equity=1_000_000, fills=[]),
            record("2026-09-15T09:00:00+09:00", equity=990_000, direction=APPROACH, fills=[fill()], amount=0.0068),
        ]
        s = notify.summarize_arm("fly", records, 1_000_000, NOW, TZ)
        self.assertEqual(s.equity_jpy, 990_000)
        self.assertAlmostEqual(s.total_pct, -1.0)
        self.assertAlmostEqual(s.change_24h_pct, -1.0)
        self.assertAlmostEqual(s.drawdown_pct, (1_010_000 - 990_000) / 1_010_000 * 100)
        self.assertEqual(s.fills_24h, 1)
        self.assertEqual((s.position_btc, s.direction), (0.0068, APPROACH))
        line = notify.report_line(s)
        self.assertIn("[fly] 総資産 990,000 JPY (-1.00%)", line)
        self.assertIn("24時間の約定 1件 / 建玉 0.0068 BTC / 方向 APPROACH", line)

    def test_fills_before_recording_started_are_unknown(self):
        """約定を記録し始める前の回が 24 時間に入っていたら、数は「—」にする（観測していない値を作らない）。"""
        records = [
            record("2026-09-15T08:00:00+09:00", equity=1_000_000),  # fills のキーが無い
            record("2026-09-15T09:00:00+09:00", equity=1_000_000, fills=[fill()]),
        ]
        s = notify.summarize_arm("fly", records, 1_000_000, NOW, TZ)
        self.assertIsNone(s.fills_24h)
        self.assertIn("24時間の約定 —件 / 建玉なし", notify.report_line(s))

    def test_no_observation_is_dashes(self):
        s = notify.summarize_arm("fly", [record("2026-09-15T09:00:00+09:00", equity=None, direction=None)], 1_000_000, NOW, TZ)
        self.assertEqual(notify.report_line(s), "[fly] 総資産 — JPY (—) / 24時間 — / 最高からの下落 — / 24時間の約定 —件 / 建玉なし / 方向 —")

    def test_report_has_a_header_and_the_source(self):
        s = notify.summarize_arm("fly", [], 1_000_000, NOW, TZ)
        report = notify.build_report(NOW, [s])
        self.assertTrue(report.startswith("**2026-09-15T09:05:00+09:00 の振り返り**"))
        self.assertIn("paper assets", report)

    def test_facts_use_only_the_flys_vocabulary(self):
        records = [
            record("2026-09-15T06:00:00+09:00", direction=APPROACH, cage="wall"),
            record("2026-09-15T07:00:00+09:00", direction=AVOID, stimulus="reward"),
            record("2026-09-15T08:00:00+09:00", direction=None, cage="guard", fills=[fill()]),
            record("2026-09-14T01:00:00+09:00", direction=APPROACH),  # 窓の外
        ]
        facts = notify.build_facts([("fly", records)], helpers.at("2026-09-15T00:00:00+09:00"), NOW, TZ)
        self.assertIn("- fly: 近づいた 1 回 / 逃げた 1 回 / じっとしてた 0 回 / 見えなかった 1 回 / 壁 1 回 / ドパきた 1 回", facts)
        for word in ("約定", "総資産", "買", "売"):
            self.assertNotIn(word, facts)


class SendTest(unittest.TestCase):
    def test_nothing_to_send_or_disabled(self):
        self.assertIsNone(notify.send(settings(), [], environ=ENV))
        self.assertIsNone(notify.send(settings(enabled=False), ["x"], environ=ENV))

    def test_missing_variable_names_the_variable_only(self):
        reason = notify.send(settings(), ["x"], environ={})
        self.assertIn("DOPABAE_DISCORD_WEBHOOK_URL", reason)

    def test_one_post_capped_at_1900_chars(self):
        sent = []
        self.assertIsNone(notify.send(settings(), ["a" * 1500, "b" * 1500], poster=lambda *a: sent.append(a), environ=ENV))
        self.assertEqual(len(sent), 1)
        url, body, timeout = sent[0]
        self.assertEqual((url, len(body), timeout), (HOOK, notify.MAX_CHARS, 10))

    def test_failures_never_carry_the_url(self):
        def http_error(*_):
            raise urllib.error.HTTPError(HOOK, 429, "Too Many Requests", {}, None)

        def broken_response(*_):
            raise http.client.IncompleteRead(b"")  # urllib の例外ではないものも捕まえる

        for poster, expected in ((http_error, "HTTP 429"), (broken_response, "IncompleteRead")):
            reason = notify.send(settings(), ["x"], poster=poster, environ=ENV)
            self.assertIn(expected, reason)
            self.assertNotIn("example.invalid", reason)

    def test_post_names_itself_and_sends_json(self):
        seen = {}

        class Response:
            def close(self):
                pass

        def fake_urlopen(request, timeout):
            seen["request"], seen["timeout"] = request, timeout
            return Response()

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            notify._post(HOOK, "本文", 7)
        request = seen["request"]
        self.assertEqual(request.get_header("User-agent"), notify.USER_AGENT)
        self.assertEqual(json.loads(request.data.decode("utf-8")), {"content": "本文"})
        self.assertEqual(seen["timeout"], 7)


class PreviousTest(unittest.TestCase):
    def test_roundtrip_and_broken_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "var" / "notify_last.yaml"
            self.assertEqual(notify.read_previous(path, TZ), notify.Previous(at=None))
            notify.write_previous(path, NOW, {"fly": 2})
            self.assertEqual(notify.read_previous(path, TZ), notify.Previous(at=NOW, failures={"fly": 2}))
            reported = NOW - timedelta(hours=1)
            notify.write_previous(path, NOW, {}, reported)
            self.assertEqual(notify.read_previous(path, TZ).reported_at, reported)
            path.write_text(": : :", encoding="utf-8")
            self.assertIsNone(notify.read_previous(path, TZ).at)


class AfterRoundTest(unittest.TestCase):
    """run_arms から呼ぶ入口。設定はリポジトリ直下の agent.yaml、記録は腕ごとの判断ログ。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = helpers.load_config()
        self.sent: list[str] = []
        self.arms = [("fly", self.arm("fly", "fly")), ("cage-only", self.arm("cage-only", "always_approach"))]

    def tearDown(self):
        self.tmp.cleanup()

    def arm(self, name: str, source: str) -> Path:
        raw = copy.deepcopy(self.cfg.raw)
        raw["direction"]["source"] = source
        raw["memory"]["decisions"]["path"] = f"var/arms/{name}/memory/decisions/{{date}}.jsonl"
        path = self.root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        log = self.root / "var" / "arms" / name / "memory" / "decisions" / "2026-09-15.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            record("2026-09-15T08:45:00+09:00", equity=1_000_000, direction=APPROACH, fills=[]),
            record("2026-09-15T09:00:00+09:00", equity=995_000, direction=APPROACH, fills=[fill()], amount=0.0068),
        ]
        log.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        return path

    def call(self, rounds, now, writer=None, config=None):
        return notify.after_round(
            self.arms, rounds, config=config or self.cfg, poster=lambda url, body, timeout: self.sent.append(body),
            writer=writer or (lambda system, user: "近づいた。ブン。ドパまだ。"), environ=ENV, now=now, root=self.root,
        )

    def test_first_round_sends_events_and_remembers_the_time(self):
        rounds = [notify.ArmRound("fly", "ok", fills=(fill(),)), notify.ArmRound("cage-only", "ok")]
        self.assertIsNone(self.call(rounds, helpers.at("2026-09-15T08:55:00+09:00")))
        self.assertEqual(self.sent, ["[fly] 約定 買い 14,699,000 × 0.0068（99,953 JPY）"])
        state = yaml.safe_load((self.root / "var" / "notify_last.yaml").read_text(encoding="utf-8"))
        self.assertEqual(state["at"], "2026-09-15T08:55:00+09:00")

    def test_report_after_crossing_the_slot_with_one_remark(self):
        rounds = [notify.ArmRound("fly", "ok"), notify.ArmRound("cage-only", "ok")]
        self.call(rounds, helpers.at("2026-09-15T08:55:00+09:00"))
        seen_facts = []

        def writer(system, user):
            seen_facts.append(user)
            return "近づいた。ブン。ドパまだ。"

        self.assertIsNone(self.call(rounds, NOW, writer=writer))
        report = self.sent[-1]
        self.assertIn("の振り返り**", report)
        self.assertIn("[fly] 総資産 995,000 JPY (-0.50%)", report)
        self.assertIn("[cage-only] 総資産 995,000 JPY", report)
        self.assertTrue(report.endswith("> 近づいた。ブン。ドパまだ。"))
        # 一言の材料はハエ版の腕だけ
        self.assertIn("- fly:", seen_facts[0])
        self.assertNotIn("cage-only", seen_facts[0])

    def test_remark_covers_the_time_since_the_last_report(self):
        """一言の材料は「前回の振り返りから今回まで」。前回の回（15 分前）からではない。"""
        rounds = [notify.ArmRound("fly", "ok"), notify.ArmRound("cage-only", "ok")]
        seen_facts = []

        def writer(system, user):
            seen_facts.append(user)
            return "近づいた。"

        self.call(rounds, helpers.at("2026-09-15T08:55:00+09:00"), writer=writer)
        self.call(rounds, NOW, writer=writer)  # 振り返り 1 回目。前回の振り返りは無いので直近 12 時間
        self.assertIn("近づいた 2 回", seen_facts[-1])
        self.call(rounds, helpers.at("2026-09-15T21:05:00+09:00"), writer=writer)  # 2 回目は 09:05 から
        self.assertIn("近づいた 0 回", seen_facts[-1])
        state = yaml.safe_load((self.root / "var" / "notify_last.yaml").read_text(encoding="utf-8"))
        self.assertEqual(state["reported_at"], "2026-09-15T21:05:00+09:00")

    def test_unacceptable_remark_is_dropped(self):
        rounds = [notify.ArmRound("fly", "ok"), notify.ArmRound("cage-only", "ok")]
        self.call(rounds, helpers.at("2026-09-15T08:55:00+09:00"))
        self.call(rounds, NOW, writer=lambda s, u: "3回買った。")
        self.assertNotIn(">", self.sent[-1])

    def test_login_expiry_is_told_in_the_report(self):
        rounds = [notify.ArmRound("fly", "ok"), notify.ArmRound("cage-only", "ok")]
        self.call(rounds, helpers.at("2026-09-15T08:55:00+09:00"))

        def expired(system, user):
            raise narrate.NarrateError("claude が異常終了しました: Failed to authenticate: OAuth session expired")

        warning = self.call(rounds, NOW, writer=expired)
        self.assertIn("一言を書けませんでした", warning)
        self.assertIn("/login", self.sent[-1])

    def test_disabled_or_missing_settings_send_nothing(self):
        rounds = [notify.ArmRound("fly", "ok", fills=(fill(),))]
        raw = copy.deepcopy(self.cfg.raw)
        raw["notify"]["enabled"] = False
        self.assertIsNone(self.call(rounds, NOW, config=dataclasses.replace(self.cfg, raw=raw)))
        del raw["notify"]
        self.assertIn("通知の設定を読めません", self.call(rounds, NOW, config=dataclasses.replace(self.cfg, raw=raw)))
        self.assertEqual(self.sent, [])
        self.assertFalse((self.root / "var" / "notify_last.yaml").exists())

    def test_error_streak_survives_between_rounds(self):
        failed = [notify.ArmRound("fly", "failed", error="終了コード 1"), notify.ArmRound("cage-only", "ok")]
        for minute in (0, 15, 30):
            self.call(failed, helpers.at(f"2026-09-15T08:{minute:02d}:00+09:00"))
        self.assertEqual(self.sent, ["[fly] 3回続けて失敗しています: 終了コード 1"])


if __name__ == "__main__":
    unittest.main()
