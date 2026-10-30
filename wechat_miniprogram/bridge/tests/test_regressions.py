"""Regression cases from the real three-model service contracts."""
import copy
import io
import json
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge import Bridge, TaskFailure, WaitingBoard
from board_client import BoardError, _classify
from preprocess import prepare_skin_image
from wechat_api import WechatCloud, WechatApiError
from tests.mocks import FakeCloud


class RecordedCloud(FakeCloud):
    def __init__(self):
        super().__init__()
        self.writes = []

    def update_doc(self, collection, doc_id, fields):
        self.writes.append((collection, doc_id, copy.deepcopy(fields)))
        return super().update_doc(collection, doc_id, copy.deepcopy(fields))


class BridgeRegressionTests(unittest.TestCase):
    def setUp(self):
        self.cloud = RecordedCloud()
        self.bridge = Bridge(self.cloud, None, {"async_results": False})
        self.bridge.board_state = {"online": True, "model": "skin", "phase": "ready", "source": "entry"}

    def tearDown(self):
        self.bridge.close()

    def test_current_model_is_not_starved_by_older_other_model_tasks(self):
        for i in range(7):
            self.cloud.add_doc("tasks", dict(type="acdc", status="pending", created_at=i))
        task_id = self.cloud.add_doc("tasks", dict(type="skin", status="pending", created_at=8))
        self.assertEqual(self.bridge.pick_task()["_id"], task_id)
        self.bridge.board_state["phase"] = "switching"
        self.assertIsNone(self.bridge.pick_task())

    def test_legacy_null_result_is_rejected_before_board_inference(self):
        task_id = self.cloud.add_doc("tasks", dict(type="skin", status="pending", created_at=1, result=None))
        self.bridge.board = Mock()
        self.assertEqual(self.bridge.process_task(), "error")
        self.assertEqual(self.cloud.collections["tasks"][task_id]["status"], "error")
        self.bridge.board.skin_submit.assert_not_called()

    def test_unchanged_heartbeat_writes_only_after_force_interval(self):
        self.bridge.board = SimpleNamespace(entry_status=lambda: {"model": "skin", "phase": "ready"})
        for now in (1000, 1005, 1021):
            with patch("bridge.time.time", return_value=now):
                self.bridge.heartbeat_once()
        self.assertEqual(len(self.cloud.writes), 2)
        self.assertEqual(self.cloud.collections["board"]["board_current"]["updated_at"], 1021000)

    def test_crashed_switch_command_reaches_terminal_error(self):
        cmd = self.cloud.add_doc("commands", dict(status="running", started_at=1))
        self.bridge.recover_stale_tasks(force=True)
        self.assertEqual(self.cloud.collections["commands"][cmd]["status"], "error")

    def test_busy_response_and_invalid_upload_are_distinct(self):
        for text in ("ContractError: another task is active", "ContractError: PL engine is not ready",
                     "ContractError: HDMI transition is not ready"):
            self.assertEqual(_classify(400, text).kind, "busy")
        self.assertEqual(_classify(400, "invalid ED/ES upload length").kind, "rejected")

    def test_busy_timeout_stays_queued_and_uncertain_submit_is_not_replayed(self):
        self.bridge.clock = Mock(side_effect=[0, 61, 61])
        with self.assertRaises(WaitingBoard):
            self.bridge._submit_with_retry("x", Mock(side_effect=BoardError("busy", "busy")))
        self.bridge.clock = lambda: 0
        submit = Mock(side_effect=BoardError("response lost", "uncertain"))
        with self.assertRaises(TaskFailure):
            self.bridge._submit_with_retry("x", submit)
        self.assertEqual(submit.call_count, 1)

    def test_acdc_csv_uses_headers_and_retains_invalid_reason(self):
        func = (b"structure,EDV_mL,ESV_mL,SV_mL,EF_percent,valid,reason\n"
                b"LV,120,50,70,58.333,True,\nRV,0,0,0,0,False,\"empty, no EF\"\n")
        stats = b"phase,structure,voxels,volume_mL\nED,LV,1000,120\n"
        values = {m["label"]: m["value"] for m in Bridge._acdc_metrics(func, stats)}
        self.assertEqual(values["LV EDV"], "120 mL")
        self.assertEqual(values["LV ESV"], "50 mL")
        self.assertEqual(values["LV SV"], "70 mL")
        self.assertEqual(values["LV EF"], "58.333 %")
        self.assertIn("empty, no EF", values["RV 心功能"])
        self.assertNotIn("RV EF", values)

    def test_acdc_unready_health_keeps_task_queued_without_upload(self):
        self.bridge.board_state["model"] = "acdc"
        tick = iter(range(100, 2000, 100))
        self.bridge.clock = lambda: next(tick)
        self.bridge.board = SimpleNamespace(acdc_health=lambda: {"ready": False, "status": "initializing"},
                                            acdc_submit=Mock())
        ed = self.cloud.upload_bytes("ed", b"ED"); es = self.cloud.upload_bytes("es", b"ES")
        task_id = self.cloud.add_doc("tasks", dict(type="acdc", status="pending", created_at=1,
            inputs=[{"fileID": ed, "name": "ed.nii"}, {"fileID": es, "name": "es.nii"}]))
        self.assertEqual(self.bridge.process_task(), "waiting")
        self.assertEqual(self.cloud.collections["tasks"][task_id]["status"], "pending")
        self.bridge.board.acdc_submit.assert_not_called()

    def test_valid_jpeg_preserves_original_bytes_and_exif(self):
        from PIL import Image
        image = Image.new("RGB", (48, 32), (55, 100, 205))
        exif = Image.Exif(); exif[274] = 6
        buf = io.BytesIO(); image.save(buf, "JPEG", quality=95, exif=exif)
        raw = buf.getvalue()
        self.assertEqual(prepare_skin_image(raw), raw)
        png = io.BytesIO(); image.save(png, "PNG")
        self.assertTrue(prepare_skin_image(png.getvalue()).startswith(b"\xff\xd8"))

    def test_previews_publish_while_zip_uploads_and_next_task_can_be_selected(self):
        gate = threading.Event(); zip_started = threading.Event()
        upload = self.cloud.upload_bytes
        def slow_zip(path, data):
            if path.endswith(".zip"):
                zip_started.set()
                if not gate.wait(3):
                    raise AssertionError("test did not release ZIP")
            return upload(path, data)
        self.cloud.upload_bytes = slow_zip
        self.bridge.async_results = True
        self.cloud.add_doc("tasks", dict(_id="first", type="skin", status="running"))
        self.cloud.add_doc("tasks", dict(_id="next", type="skin", status="pending", created_at=1))
        task = {"_id": "first", "key": "first"}
        items = [("mask.png", "mask", b"PNG"), ("result.zip", "ZIP", b"ZIP")]
        try:
            self.bridge._finish_task(task, items, [])
            self.assertTrue(zip_started.wait(2))
            # ZIP can start before the completion iterator has written the PNG.
            until = time.monotonic() + 2
            while not self.cloud.collections["tasks"]["first"].get("result") and time.monotonic() < until:
                time.sleep(.005)
            first = self.cloud.collections["tasks"]["first"]
            self.assertEqual(first["status"], "running")
            self.assertEqual(first["result"]["views"][0]["name"], "mask.png")
            self.assertIsNone(first["result"]["zip"])
            self.assertEqual(self.bridge.pick_task()["_id"], "next")
            self.bridge._finish_task({"_id": "second"}, items, [])
            self.assertIsNone(self.bridge.pick_task())  # two buffered sets is the limit
        finally:
            gate.set()
            self.bridge.close()
        result = self.cloud.collections["tasks"]["first"]["result"]
        self.assertEqual(self.cloud.storage[result["views"][0]["fileID"]], b"PNG")
        self.assertEqual(self.cloud.storage[result["zip"]], b"ZIP")

    def test_partial_upload_resumes_without_reuploading_successful_images(self):
        upload = self.cloud.upload_bytes; failed = False; paths = []
        def fail_zip_once(path, data):
            nonlocal failed
            paths.append(path)
            if path.endswith(".zip") and not failed:
                failed = True
                raise WechatApiError("temporary upload failure")
            return upload(path, data)
        self.cloud.upload_bytes = fail_zip_once
        task = {"_id": "resume", "key": "resume"}
        self.cloud.add_doc("tasks", task)
        items = [("mask.png", "mask", b"PNG"), ("result.zip", "ZIP", b"ZIP")]
        with self.assertRaises(WechatApiError):
            self.bridge._finish_task(task, items, [])
        task["result"] = self.cloud.collections["tasks"]["resume"]["result"]
        self.bridge._finish_task(task, items, [])
        self.assertEqual(sum(p.endswith("mask.png") for p in paths), 1)
        self.assertEqual(self.cloud.collections["tasks"]["resume"]["status"], "done")

    def test_result_retry_does_not_submit_another_inference(self):
        self.bridge.board = SimpleNamespace(
            skin_submit=Mock(side_effect=AssertionError("must not infer again")),
            skin_job=lambda _: {"state": "done", "views": ["mask"]},
            skin_file=lambda _, name: name.encode())
        task = {"_id": "retry", "type": "skin", "resume_job": {"model": "skin", "id": "existing"}}
        self.cloud.add_doc("tasks", task)
        self.bridge._run_skin(task)
        self.assertEqual(self.cloud.collections["tasks"]["retry"]["status"], "done")
        self.bridge.board.skin_submit.assert_not_called()


class CloudConcurrencyTests(unittest.TestCase):
    def test_api_reuses_keepalive_connection_and_concurrent_calls_are_exclusive(self):
        connections = []
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def setup(self):
                super().setup(); connections.append(self.connection)
            def log_message(self, *_): pass
            def do_GET(self):
                self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers()
                self.wfile.write(b"OK")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
        thread.start()
        cloud = WechatCloud("a", "b", "c", api_base="http://127.0.0.1:%d" % server.server_port)
        def get():
            return cloud._request("http", "127.0.0.1", server.server_port, "GET", "/")[2]
        try:
            self.assertEqual([get(), get(), get()], [b"OK"] * 3)
            self.assertEqual(len(connections), 1)
            with ThreadPoolExecutor(max_workers=4) as pool:
                self.assertEqual(list(pool.map(lambda _: get(), range(12))), [b"OK"] * 12)
            self.assertLessEqual(len(connections), 4)
        finally:
            cloud.close(); server.shutdown(); server.server_close(); thread.join(2)

    def test_concurrent_token_requests_fetch_only_once(self):
        cloud = WechatCloud("a", "b", "c")
        fetched = []
        def fetch(now):
            fetched.append(1); time.sleep(.02)
            cloud._token = "one-token"; cloud._token_expries = now + 7200
            return cloud._token
        cloud._fetch_token = fetch
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(list(pool.map(lambda _: cloud.access_token(), range(8))), ["one-token"] * 8)
        self.assertEqual(len(fetched), 1)


if __name__ == "__main__":
    unittest.main()
