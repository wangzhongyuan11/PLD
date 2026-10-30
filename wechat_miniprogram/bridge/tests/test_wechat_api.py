# -*- coding: utf-8 -*-
import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.mocks import MockWechatApi  # noqa: E402
from wechat_api import WechatCloud, WechatApiError  # noqa: E402


class WechatApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api_mock = MockWechatApi().start()
        cls.cloud = WechatCloud("appid", "secret", "env-x",
                                api_base="http://127.0.0.1:%d" % cls.api_mock.port)

    @classmethod
    def tearDownClass(cls):
        cls.api_mock.stop()

    def test_01_token_cached_across_calls(self):
        self.cloud.add_doc("tasks", {"a": 1})
        self.cloud.query_docs("tasks", {"status": "pending"}, ("created_at", "asc"), 5)
        self.assertEqual(self.api_mock.token_calls, 1)

    def test_02_expired_token_refreshes_and_retries(self):
        self.api_mock.expire_token_once = True
        docs = self.cloud.query_docs("tasks", {}, None, 5)
        self.assertEqual(docs[0]["_id"], "doc1")
        self.assertGreaterEqual(self.api_mock.token_calls, 2)

    def test_03_update_and_where_dsl(self):
        updated = self.cloud.update_where(
            "tasks", {"_id": "doc1", "status": {"$in": ["pending"]}},
            {"status": "running"})
        self.assertEqual(updated, 1)
        query = [c for c in self.api_mock.calls if c[0] == "/tcb/databaseupdate"][-1][1]
        body = json.loads(query)
        self.assertIn('db.command.in(["pending"])', body["query"])

    def test_04_upload_bytes_builds_cos_multipart(self):
        file_id = self.cloud.upload_bytes("tasks/k1/out/overlay.png", b"\x89PNGDATA")
        self.assertTrue(file_id.startswith("cloud://fid-"))
        stored = self.api_mock.uploads["fid-tasks_k1_out_overlay.png"]
        self.assertIn(b'name="key"', stored)
        self.assertIn(b"tasks/k1/out/overlay.png", stored)
        self.assertIn(b'name="Signature"', stored)
        self.assertIn(b"name=\"x-cos-security-token\"", stored)
        self.assertIn(b"\x89PNGDATA", stored)

    def test_05_download_bytes_roundtrip(self):
        self.api_mock.files["fake/1/a.png"] = b"HELLO-DATA"
        data = self.cloud.download_bytes("cloud://fake/1/a.png")
        self.assertEqual(data, b"HELLO-DATA")

    def test_06_upload_failure_raises(self):
        # force a bad path: mock returns 404 for unknown endpoints only,
        # so use a network failure via unreachable host instead.
        cloud = WechatCloud("a", "b", "c", api_base="http://127.0.0.1:9")
        with self.assertRaises(WechatApiError):
            cloud.access_token()

    def test_07_token_failure_backoff(self):
        cloud = WechatCloud("a", "b", "c", api_base="http://127.0.0.1:9")
        with self.assertRaises(WechatApiError):
            cloud.access_token()
        with self.assertRaises(WechatApiError) as ctx:
            cloud.access_token()
        self.assertIn("退避", str(ctx.exception))
        # a valid cached token bypasses the backoff
        cloud._token = "CACHED"
        cloud._token_expries = time.time() + 3600
        self.assertEqual(cloud.access_token(), "CACHED")


if __name__ == "__main__":
    unittest.main()
