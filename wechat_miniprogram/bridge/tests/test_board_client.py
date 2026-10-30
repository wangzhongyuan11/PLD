# -*- coding: utf-8 -*-
import http.client
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.mocks import MockBoard  # noqa: E402
from board_client import BoardClient, BoardError  # noqa: E402


def make_client(mock, hub_token=None):
    return BoardClient("127.0.0.1", mock.entry_port, mock.hub_port,
                       mock.acdc_port,
                       MockBoard.HUB_TOKEN if hub_token is None else hub_token)


class BoardClientTests(unittest.TestCase):
    def setUp(self):
        self.mock = MockBoard().start()
        self.client = make_client(self.mock)

    def tearDown(self):
        self.mock.stop()

    def test_entry_status_and_absence(self):
        snap = self.client.entry_status()
        self.assertEqual(snap["model"], "skin")
        self.mock.entry_enabled = False
        self.assertIsNone(self.client.entry_status())

    def test_entry_switch_requires_origin(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.mock.entry_port, timeout=5)
        conn.request("POST", "/entry/switch", json.dumps({"model": "camus"}).encode(),
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        self.assertEqual(resp.status, 403)
        conn.close()

    def test_entry_switch_to_ready(self):
        self.client.entry_switch("camus")
        snap = self.client.wait_entry_ready("camus", deadline=1e18,
                                            interval=0, sleep=lambda s: None)
        self.assertEqual((snap["model"], snap["phase"]), ("camus", "ready"))

    def test_hub_login_wrong_token(self):
        client = make_client(self.mock, hub_token="wrong-token")
        with self.assertRaises(BoardError) as ctx:
            client.hub_login()
        self.assertEqual(ctx.exception.kind, "auth")

    def test_hub_generation_and_skin_flow(self):
        model, gen = self.client.hub_generation()
        self.assertEqual((model, gen), ("skin", 3))
        job = self.client.skin_submit(b"\xff\xd8FAKEJPG", thickness=3,
                                      name="t.jpg")
        self.assertEqual(job["state"], "queued")
        job_id = job["id"]
        first = self.client.skin_job(job_id)
        done = self.client.skin_job(job_id)
        self.assertEqual(done["state"], "done")
        self.assertIn("outlined", done["views"])
        data = self.client.skin_file(job_id, "outlined.png")
        self.assertTrue(data.startswith(b"\x89PNG-fake-"))
        self.mock.hub_generation += 1  # simulate a model switch
        refreshed = self.client.skin_job(job_id)  # must self-heal via 409
        self.assertEqual(refreshed["state"], "done")

    def test_skin_busy_and_paused_classification(self):
        self.mock.hub_busy_polls = 1
        with self.assertRaises(BoardError) as ctx:
            self.client.skin_submit(b"\xff\xd8FAKEJPG")
        self.assertEqual(ctx.exception.kind, "busy")
        self.mock.hub_suspended_model = "skin"
        with self.assertRaises(BoardError) as ctx:
            self.client.skin_submit(b"\xff\xd8FAKEJPG")
        self.assertEqual(ctx.exception.kind, "paused")

    def test_camus_flow(self):
        self.mock.hub_model = "camus"
        job = self.client.camus_submit([("case.npz", b"NPZDATA")], "case.npz")
        job_id = job["id"]
        self.client.camus_job(job_id)
        done = self.client.camus_job(job_id)
        self.assertEqual(done["state"], "done")
        report = self.client.camus_file(job_id, "report.json")
        names = json.loads(report)["output"]["class_names"]
        self.assertEqual(names[1], "lv_cavity")

    def test_acdc_raw_upload_and_assets(self):
        job = self.client.acdc_submit(b"ED-BYTES"*10, b"ES-BYTES"*5,
                                      "ED.nii.gz", "ES.nii.gz")
        self.assertEqual(job["status"], "queued")
        upload = self.mock.acdc_uploads[0]
        self.assertEqual(upload["ed"], b"ED-BYTES"*10)
        self.assertEqual(upload["es"], b"ES-BYTES"*5)
        self.assertEqual(upload["ed_name"], "ED.nii.gz")
        self.assertEqual(upload["auto_display"], "false")
        job_id = job["run_id"]
        for _ in range(3):
            status = self.client.acdc_job(job_id)
        self.assertEqual(status["status"], "assets_ready")
        data = self.client.acdc_file(job_id, "mpr/ED/axial.png")
        self.assertTrue(data.startswith(b"ACDC-asset:mpr/ED/axial.png"))
        self.client.acdc_cancel(job_id)
        self.assertEqual(self.mock.acdc_cancelled, [job_id])

    def test_acdc_rejects_foreign_origin_and_bad_suffix(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.mock.acdc_port, timeout=5)
        conn.request("POST", "/api/v2/jobs?ed=ED.nii.gz&es=ES.nii.gz", b"x"*10,
                     {"Content-Type": "application/octet-stream",
                      "X-ED-Bytes": "5", "Origin": "http://evil.example"})
        self.assertEqual(conn.getresponse().status, 401)
        conn.close()
        with self.assertRaises(BoardError) as ctx:
            self.client.acdc_submit(b"a", b"b", "ED.dcm", "ES.dcm")
        self.assertEqual(ctx.exception.kind, "rejected")

    def test_hub_loopback_falls_back_to_entry_proxy(self):
        self.mock.close_hub()
        self.mock.hub_generation = 4
        self.mock.entry_generation = 4
        self.mock.entry_state = {"model": "skin", "phase": "ready",
                                 "url": "/skin/g/4/"}
        model, gen = self.client.hub_generation()
        self.assertEqual((model, gen), ("skin", 4))
        self.assertEqual(self.client._hub_mode, "proxy")
        job = self.client.skin_submit(b"\xff\xd8FAKE", thickness=2,
                                      name="t.jpg")
        job_id = job["id"]
        self.client.skin_job(job_id)
        done = self.client.skin_job(job_id)
        self.assertEqual(done["state"], "done")
        data = self.client.skin_file(job_id, "outlined.png")
        self.assertTrue(data.startswith(b"\x89PNG-fake-"))

    def test_offline_classification(self):
        client = BoardClient("127.0.0.1", 9, 9, 9)
        with self.assertRaises(BoardError) as ctx:
            client.entry_status()
        self.assertEqual(ctx.exception.kind, "offline")

    def test_lost_submit_response_is_not_replayed_via_entry_proxy(self):
        self.client._generation = 3
        self.client._request = Mock(side_effect=http.client.RemoteDisconnected("response lost"))
        with self.assertRaises(BoardError) as ctx:
            self.client.skin_submit(b"\xff\xd8FAKE")
        self.assertEqual(ctx.exception.kind, "uncertain")
        self.assertEqual(self.client._request.call_count, 1)
        self.assertEqual(self.client._hub_mode, "direct")


if __name__ == "__main__":
    unittest.main()
