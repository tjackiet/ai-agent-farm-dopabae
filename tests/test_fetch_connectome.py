"""配線図データの取得。指紋の合わないものを置かず、既にあるものを上書きしないこと。

外には出ない。取得は差し替えて、ネットワークに触れたら失敗させる。
"""

from __future__ import annotations

import hashlib
import io
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from dopabae.config import ConnectomeFile

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import fetch_connectome as fetch  # noqa: E402

sys.path.pop(0)

BODY = b"connectome bytes"
EXPECTED = ConnectomeFile(path="d/f.feather", sha256=hashlib.sha256(BODY).hexdigest(), size_bytes=len(BODY))


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class UrlTest(unittest.TestCase):
    def test_gs_becomes_public_https(self):
        self.assertEqual(
            fetch.https_url("gs://flyem-male-cns/v1.0/", "connectome-data/a.feather"),
            "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/a.feather",
        )

    def test_trailing_slash_is_added(self):
        self.assertEqual(fetch.https_url("https://example.org/v1", "a"), "https://example.org/v1/a")

    def test_other_schemes_are_refused(self):
        with self.assertRaises(fetch.FetchError):
            fetch.https_url("ftp://example.org/", "a")


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self.dir = TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.dest = Path(self.dir.name) / "d" / "f.feather"

    def test_matching_download_is_kept(self):
        with mock.patch.object(fetch.urllib.request, "urlopen", return_value=Response(BODY)):
            fetch.download("https://example.org/f", self.dest, EXPECTED)
        self.assertEqual(self.dest.read_bytes(), BODY)
        self.assertFalse(self.dest.with_name("f.feather.part").exists())

    def test_mismatched_download_is_discarded(self):
        """指紋が合わなければ、所定の名前にも一時ファイルにも残さない。"""
        with mock.patch.object(fetch.urllib.request, "urlopen", return_value=Response(b"tampered bytes!!")):
            with self.assertRaises(fetch.FetchError):
                fetch.download("https://example.org/f", self.dest, EXPECTED)
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.dest.with_name("f.feather.part").exists())

    def test_network_error_leaves_nothing(self):
        with mock.patch.object(fetch.urllib.request, "urlopen", side_effect=OSError("offline")):
            with self.assertRaises(fetch.FetchError):
                fetch.download("https://example.org/f", self.dest, EXPECTED)
        self.assertFalse(self.dest.exists())

    def test_matches_checks_size_and_hash(self):
        self.dest.parent.mkdir(parents=True)
        self.dest.write_bytes(BODY)
        self.assertTrue(fetch.matches(self.dest, EXPECTED))
        self.dest.write_bytes(BODY + b"x")
        self.assertFalse(fetch.matches(self.dest, EXPECTED))


class FetchTest(unittest.TestCase):
    """既にあるファイルは、指紋が合えば取らず、合わなければ上書きせずに止まる。"""

    def setUp(self):
        self.dir = TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        from dopabae.config import load

        import dataclasses

        base = load()
        self.config = dataclasses.replace(
            base, connectome_local_dir="var/connectome", connectome_files={"annotations": EXPECTED}
        )
        self.dest = self.root / "var" / "connectome" / EXPECTED.path
        self.dest.parent.mkdir(parents=True)

    def run_fetch(self):
        with mock.patch.object(fetch, "load", return_value=self.config), mock.patch.object(
            fetch, "REPO_ROOT", self.root
        ), mock.patch.object(
            fetch.urllib.request, "urlopen", side_effect=AssertionError("ネットワークに触れた")
        ), mock.patch("sys.stdout", new_callable=io.StringIO), mock.patch(
            "sys.stderr", new_callable=io.StringIO
        ):
            return fetch.fetch(None)

    def test_existing_match_is_not_fetched_again(self):
        self.dest.write_bytes(BODY)
        self.assertEqual(self.run_fetch(), 0)

    def test_existing_mismatch_is_not_overwritten(self):
        self.dest.write_bytes(b"someone else's file")
        self.assertEqual(self.run_fetch(), 1)
        self.assertEqual(self.dest.read_bytes(), b"someone else's file")

    def test_unknown_name_is_refused(self):
        with mock.patch.object(fetch, "load", return_value=self.config), mock.patch(
            "sys.stderr", new_callable=io.StringIO
        ):
            self.assertEqual(fetch.fetch(["synapses"]), 1)


if __name__ == "__main__":
    unittest.main()
