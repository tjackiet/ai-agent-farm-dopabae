"""記録の言語化。ハエが知らないことを言わせないこと、失敗しても運用を止めないこと。"""

from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

from dopabae import config as config_module, narrate, summary
from tests import helpers

SKELETON = f"# 2026-09-14 — ドパバエ\n\n## おれの一日\n\n{summary.UNWRITTEN}\n\n## 観測\n\n| 回した回数 | 11 |\n"


class PromptTest(unittest.TestCase):
    def test_carries_the_vocabulary_table_and_the_bans(self):
        prompt = narrate.build_prompt()
        self.assertIn("ハエの語彙", prompt)  # personality.md の対応表
        self.assertIn("数字を書かない", prompt)
        self.assertIn("反省しない", prompt)
        self.assertIn("おれ", prompt)

    def test_missing_personality_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("ドパバエ", narrate.build_prompt(Path(tmp)))


class AcceptanceTest(unittest.TestCase):
    """空欄のまま残るほうが、ハエが知らないはずのことを言った記録より良い。"""

    def test_accepts_the_flys_words(self):
        self.assertTrue(narrate.is_acceptable("近づいた。もっかい近づいた。\nドパまだ。"))
        self.assertTrue(narrate.is_acceptable("壁。なんで壁。ブン。"))

    def test_rejects_digits(self):
        self.assertFalse(narrate.is_acceptable("3回近づいた。"))
        self.assertFalse(narrate.is_acceptable("８回近づいた。"))

    def test_rejects_words_the_fly_does_not_have(self):
        for text in ("買った。", "売った。", "損した。", "相場が動いた。", "価格が上がった。"):
            self.assertFalse(narrate.is_acceptable(text), text)

    def test_rejects_reflection_and_plans(self):
        for text in ("次はもっと近づく。", "今度は逃げる。", "反省した。", "もっと待つべき。"):
            self.assertFalse(narrate.is_acceptable(text), text)

    def test_rejects_empty(self):
        self.assertFalse(narrate.is_acceptable(""))


class CleanTest(unittest.TestCase):
    def test_strips_bullets_and_quotes(self):
        self.assertEqual(narrate._clean("- 近づいた。\n> ドパまだ。"), "近づいた。\nドパまだ。")

    def test_limits_the_number_of_lines(self):
        many = "\n".join(f"行{'あ' * i}" for i in range(10))
        self.assertLessEqual(len(narrate._clean(many).splitlines()), narrate.MAX_LINES)

    def test_drops_blank_lines(self):
        self.assertEqual(narrate._clean("近づいた。\n\n\nドパまだ。"), "近づいた。\nドパまだ。")


class FillTest(unittest.TestCase):
    def test_fills_the_blank(self):
        text, changed = narrate.fill(SKELETON, lambda s, u: "近づいた。ドパまだ。", "p")
        self.assertTrue(changed)
        self.assertNotIn(summary.UNWRITTEN, text)
        self.assertIn("近づいた。ドパまだ。", text)

    def test_leaves_the_blank_when_the_answer_is_rejected(self):
        text, changed = narrate.fill(SKELETON, lambda s, u: "3回買った。", "p")
        self.assertFalse(changed)
        self.assertIn(summary.UNWRITTEN, text)

    def test_a_nested_blank_is_refused(self):
        """返答が空欄を含むと、次の実行が入れ子に壊す。"""
        text, changed = narrate.fill(SKELETON, lambda s, u: summary.UNWRITTEN, "p")
        self.assertFalse(changed)
        self.assertEqual(text.count(summary.UNWRITTEN), 1)

    def test_nothing_to_fill_is_not_a_change(self):
        _, changed = narrate.fill("空欄のない本文", lambda s, u: "x", "p")
        self.assertFalse(changed)


class FillUnwrittenTest(unittest.TestCase):
    def setUp(self):
        self.cfg = helpers.load_config()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dir = self.root / "var" / "memory" / "daily"
        self.dir.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name: str, text: str) -> Path:
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_fills_every_diary_with_a_blank(self):
        a, b = self.write("2026-09-13.md", SKELETON), self.write("2026-09-14.md", SKELETON)
        filled = narrate.fill_unwritten(self.cfg, lambda s, u: "近づいた。", self.root)
        self.assertEqual(sorted(p.name for p in filled), [a.name, b.name])

    def test_skips_diaries_already_written(self):
        self.write("2026-09-13.md", "もう書いてある")
        self.assertEqual(narrate.fill_unwritten(self.cfg, lambda s, u: "近づいた。", self.root), [])

    def test_disabled_writes_nothing(self):
        self.write("2026-09-13.md", SKELETON)
        cfg = dataclasses.replace(self.cfg, narrate_enabled=False)
        self.assertEqual(narrate.fill_unwritten(cfg, lambda s, u: "近づいた。", self.root), [])

    def test_target_off_writes_nothing(self):
        self.write("2026-09-13.md", SKELETON)
        cfg = dataclasses.replace(self.cfg, narrate_targets={"daily": False})
        self.assertEqual(narrate.fill_unwritten(cfg, lambda s, u: "近づいた。", self.root), [])


class ModelGuardTest(unittest.TestCase):
    """料金が上位帯のモデルは、設定に書かれていても受け付けない。"""

    def test_rejects_forbidden_models(self):
        import yaml

        raw = config_module.load().raw
        for name in ("claude-fable-5-1", "claude-mythos-5"):
            bad = dict(raw)
            bad["narrate"] = dict(bad["narrate"])
            bad["narrate"]["model"] = name
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
                yaml.safe_dump(bad, tmp, allow_unicode=True)
            with self.assertRaises(config_module.ConfigError):
                config_module.load(tmp.name)
