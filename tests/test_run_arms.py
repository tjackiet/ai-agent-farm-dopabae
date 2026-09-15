"""腕をまとめて回すスケジューラ。腕が落ちても定期実行を止めないこと。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_arms  # noqa: E402

sys.path.pop(0)


class DiscoverTest(unittest.TestCase):
    def test_explicit_configs_win(self):
        self.assertEqual(run_arms.discover(["a.yaml", "b.yaml"]), [Path("a.yaml"), Path("b.yaml")])

    def test_arms_directory_is_used_when_nothing_is_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            arms = Path(tmp)
            (arms / "fly.yaml").write_text("x", encoding="utf-8")
            (arms / "random.yaml").write_text("x", encoding="utf-8")
            (arms / "notes.txt").write_text("x", encoding="utf-8")
            found = run_arms.discover(None, arms_dir=arms)
        self.assertEqual([p.name for p in found], ["fly.yaml", "random.yaml"])

    def test_falls_back_to_the_repo_config_when_there_are_no_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            found = run_arms.discover(None, arms_dir=Path(tmp))
        self.assertEqual(found, [REPO_ROOT / "agent.yaml"])

    def test_missing_arms_directory_is_not_an_error(self):
        found = run_arms.discover(None, arms_dir=REPO_ROOT / "does-not-exist")
        self.assertEqual(found, [REPO_ROOT / "agent.yaml"])


class SummarizeTest(unittest.TestCase):
    def test_reads_action_and_cage(self):
        status, action, detail = run_arms.summarize(json.dumps({"action": "BUY", "cage": "bought"}))
        self.assertEqual((status, action, detail), ("ok", "BUY", "bought"))

    def test_error_wins_over_cage(self):
        _, _, detail = run_arms.summarize(json.dumps({"action": "HOLD", "cage": "guard", "error": "boom"}))
        self.assertEqual(detail, "boom")

    def test_skipped_is_not_a_failure(self):
        status, _, detail = run_arms.summarize(json.dumps({"skipped": True, "reason": "まだ走っている"}))
        self.assertEqual(status, "skipped")
        self.assertIn("まだ走っている", detail)

    def test_unreadable_output_is_a_failure(self):
        self.assertEqual(run_arms.summarize("not json")[0], "failed")
        self.assertEqual(run_arms.summarize(json.dumps([1, 2]))[0], "failed")


class LogPathTest(unittest.TestCase):
    def test_arm_logs_are_separated_per_arm(self):
        out, err = run_arms.log_paths(REPO_ROOT / "arms" / "fly.yaml")
        self.assertTrue(str(out).endswith("var/arms/fly/run.log"))
        self.assertTrue(str(err).endswith("var/arms/fly/run.err.log"))

    def test_repo_config_logs_to_var(self):
        out, _ = run_arms.log_paths(REPO_ROOT / "agent.yaml")
        self.assertTrue(str(out).endswith("var/run.log"))


class CliTest(unittest.TestCase):
    def run_script(self, *args):
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_arms.py"), *args],
            capture_output=True, text=True, cwd=REPO_ROOT, timeout=180,
        )

    def test_missing_config_is_reported_and_fails(self):
        proc = self.run_script("--config", "arms/does-not-exist.yaml")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("設定が見つかりません", json.loads(proc.stdout)["error"])

    def test_runs_the_repo_config_and_always_exits_zero(self):
        """bitbank が無い環境でも、腕は HOLD を返して定期実行は成功する。"""
        proc = self.run_script("--config", "agent.yaml", "--timeout-sec", "60")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(len(report["arms"]), 1)
        arm = report["arms"][0]
        self.assertEqual(arm["name"], "agent")
        self.assertIn(arm["status"], ("ok", "skipped"))

    def test_a_held_lock_shows_up_as_skipped(self):
        from dopabae import config as config_module, lock

        cfg = config_module.load()
        with lock.hold(REPO_ROOT / cfg.lock_path):
            proc = self.run_script("--config", "agent.yaml", "--timeout-sec", "60")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["arms"][0]["status"], "skipped")
