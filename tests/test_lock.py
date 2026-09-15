"""二重起動を防ぐロック。重なった回は何もせずに終わること。"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dopabae import lock


class LockTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "nested" / "run.lock"

    def tearDown(self):
        self.tmp.cleanup()

    def test_creates_the_parent_directory(self):
        with lock.hold(self.path):
            self.assertTrue(self.path.parent.is_dir())

    def test_records_the_holding_pid(self):
        import os

        with lock.hold(self.path):
            self.assertEqual(self.path.read_text(encoding="utf-8").strip(), str(os.getpid()))

    def test_second_acquire_is_refused_while_held(self):
        with lock.hold(self.path):
            with self.assertRaises(lock.AlreadyRunning):
                with lock.hold(self.path):
                    pass

    def test_released_on_exit(self):
        with lock.hold(self.path):
            pass
        with lock.hold(self.path) as enforced:
            self.assertTrue(enforced)

    def test_released_even_when_the_body_raises(self):
        with self.assertRaises(RuntimeError):
            with lock.hold(self.path):
                raise RuntimeError("boom")
        with lock.hold(self.path):
            pass  # 取れれば解放されている

    def test_the_error_names_the_holder(self):
        import os

        with lock.hold(self.path):
            try:
                with lock.hold(self.path):
                    self.fail("取れてはいけない")
            except lock.AlreadyRunning as exc:
                self.assertIn(str(os.getpid()), str(exc))


class RunSkipTest(unittest.TestCase):
    """ロックが握られている間の `dopabae.run` は、何もせず終了コード 0 で終わる。"""

    def test_run_skips_and_exits_zero(self):
        from dopabae import config as config_module

        cfg = config_module.load()
        root = Path(__file__).resolve().parent.parent
        with lock.hold(root / cfg.lock_path):
            proc = subprocess.run(
                [sys.executable, "-m", "dopabae.run", "--dry-run"],
                capture_output=True, text=True, cwd=root, timeout=120,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        import json

        payload = json.loads(proc.stdout)
        self.assertTrue(payload["skipped"])
        self.assertIn("まだ走っている", payload["reason"])
