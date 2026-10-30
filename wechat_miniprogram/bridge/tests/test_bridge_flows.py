# -*- coding: utf-8 -*-
import io
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.mocks import MockBoard, FakeCloud  # noqa: E402
from board_client import BoardClient  # noqa: E402
import bridge as bridge_mod  # noqa: E402
from bridge import Bridge  # noqa: E402

try:
    from PIL import Image
    PILLOW = True
except ImportError:
    PILLOW = False


def make_bridge(mock, cloud=None, **overrides):
    cfg = {"poll_interval_ms": 10, "heartbeat_interval_ms": 1000,
           "async_results": False,
           "switch_timeout_s": 5,
           "task_timeouts": {"skin": 60, "camus": 60, "acdc": 60}}
    cfg.update(overrides)
    board = BoardClient("127.0.0.1", mock.entry_port, mock.hub_port,
                        mock.acdc_port, MockBoard.HUB_TOKEN)
    br = Bridge(cloud or FakeCloud(), board, cfg,
                sleep=lambda s: None, clock=time.monotonic)
    return br


def seed_task(cloud, kind, inputs, params=None, status="pending"):
    return cloud.add_doc("tasks", {
        "type": kind, "status": status, "inputs": inputs,
        "params": params or {}, "created_at": int(time.time() * 1000),
        "key": "k-%s" % kind})


def fake_png():
    im = Image.new("RGB", (6, 6), (200, 60, 60))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


class BridgeFlowTests(unittest.TestCase):
    def setUp(self):
        self.mock = MockBoard().start()
        self.cloud = FakeCloud()
        self.bridge = make_bridge(self.mock, self.cloud)

    def tearDown(self):
        self.bridge.close()
        self.mock.stop()

    def heartbeat(self):
        return self.bridge.heartbeat_once(force=True)

    def test_01_camus_task_end_to_end(self):
        self.mock.entry_state = {"model": "camus", "phase": "ready"}
        self.mock.hub_model = "camus"
        fid = self.cloud.upload_bytes("demos/camus/case.npz", b"NPZ-INPUT")
        task_id = seed_task(self.cloud, "camus",
                            [{"fileID": fid, "name": "case.npz"}],
                            {"input_name": "case.npz"})
        state = self.heartbeat()
        self.assertEqual((state["online"], state["model"]), (True, "camus"))
        self.bridge.process_task()
        task = self.cloud.collections["tasks"][task_id]
        self.assertEqual(task["status"], "done")
        labels = [v["label"] for v in task["result"]["views"]]
        self.assertIn("叠加图", labels)
        self.assertIn("分割图", labels)
        metric_labels = [m["label"] for m in task["result"]["metrics"]]
        self.assertIn("面积 lv_cavity", metric_labels)
        values = {m["label"]: m["value"] for m in task["result"]["metrics"]}
        self.assertEqual(values["面积 lv_cavity"], "12.50 mm²")
        self.assertEqual(values["面积 myocardium"], "33 像素")
        self.assertTrue(task["result"]["zip"])
        # input must round-trip through the mock camus backend
        job_ids = list(self.mock.jobs.keys())
        self.assertTrue(job_ids)

    def test_02_acdc_task_end_to_end(self):
        ed = self.cloud.upload_bytes("demos/acdc/ED.nii.gz", b"ED-BLOB")
        es = self.cloud.upload_bytes("demos/acdc/ES.nii.gz", b"ES-BLOB")
        task_id = seed_task(self.cloud, "acdc",
                            [{"fileID": ed, "name": "ED.nii.gz"},
                             {"fileID": es, "name": "ES.nii.gz"}],
                            {"ed_name": "ED.nii.gz", "es_name": "ES.nii.gz"})
        self.mock.entry_state = {"model": "acdc", "phase": "ready"}
        self.heartbeat()
        self.assertEqual(self.bridge.process_task(), "done")
        task = self.cloud.collections["tasks"][task_id]
        self.assertEqual(task["status"], "done")
        self.assertEqual(len(task["result"]["views"]), 6)  # six MPR views
        file_labels = [f["label"] for f in task["result"]["files"]]
        self.assertIn("统计表", file_labels)
        self.assertIn("心功能表", file_labels)
        upload = self.mock.acdc_uploads[0]
        self.assertEqual(upload["ed"], b"ED-BLOB")
        values = {m["label"]: m["value"] for m in task["result"]["metrics"]}
        self.assertEqual(values.get("LV EDV"), "120.5 mL")

    def test_03_task_waits_until_model_switched(self):
        self.mock.entry_state = {"model": "acdc", "phase": "ready"}
        self.heartbeat()
        # newest entry snapshots may carry a switching stage; bridge passes it through
        self.mock.entry_state = {"model": "acdc", "phase": "switching",
                                 "stage": "release_previous_model"}
        self.heartbeat()
        board_doc = self.cloud.collections["board"]["board_current"]
        self.assertEqual(board_doc["stage"], "release_previous_model")
        self.mock.entry_state = {"model": "acdc", "phase": "ready"}
        fid = self.cloud.upload_bytes("demos/camus/case.npz", b"NPZ")
        task_id = seed_task(self.cloud, "camus",
                            [{"fileID": fid, "name": "case.npz"}])
        self.assertIsNone(self.bridge.pick_task())  # acdc owns the FPGA
        self.assertIsNone(self.bridge.process_task())
        # user issues a switch command from the mini program
        cmd_id = self.cloud.add_doc("commands", {
            "type": "switch", "model": "camus", "status": "pending",
            "created_at": int(time.time() * 1000)})
        self.assertEqual(self.bridge.process_commands(), "done")
        self.assertEqual(self.cloud.collections["commands"][cmd_id]["status"],
                         "done")
        self.assertEqual(self.mock.entry_switch_calls, ["camus"])
        self.heartbeat()
        self.assertEqual(self.bridge.process_task(), "done")
        self.assertEqual(self.cloud.collections["tasks"][task_id]["status"],
                         "done")

    def test_04_acdc_rejected_without_entry(self):
        self.mock.entry_enabled = False  # dual-model deployment fallback
        self.heartbeat()
        state = self.bridge.current_model()
        self.assertEqual(state["source"], "hub")
        fid = self.cloud.upload_bytes("demos/acdc/ED.nii.gz", b"ED")
        task_id = seed_task(self.cloud, "acdc", [{"fileID": fid,
                                                  "name": "ED.nii.gz"}])
        self.bridge.process_task()
        task = self.cloud.collections["tasks"][task_id]
        self.assertEqual(task["status"], "error")
        self.assertIn("三模型入口", task["error"])

    def test_05_busy_board_retries_then_finishes(self):
        self.mock.entry_state = {"model": "camus", "phase": "ready"}
        self.mock.hub_model = "camus"
        self.mock.hub_busy_polls = 2
        fid = self.cloud.upload_bytes("demos/camus/case.npz", b"NPZ")
        task_id = seed_task(self.cloud, "camus",
                            [{"fileID": fid, "name": "case.npz"}])
        self.heartbeat()
        self.assertEqual(self.bridge.process_task(), "done")
        self.assertEqual(self.cloud.collections["tasks"][task_id]["status"],
                         "done")

    def test_06_task_timeout_marks_error(self):
        self.mock.entry_state = {"model": "camus", "phase": "ready"}
        self.mock.hub_model = "camus"
        self.mock.polls_to_done = 10 ** 9
        fid = self.cloud.upload_bytes("demos/camus/case.npz", b"NPZ")
        task_id = seed_task(self.cloud, "camus",
                            [{"fileID": fid, "name": "case.npz"}])
        self.bridge = make_bridge(self.mock, self.cloud,
                                  task_timeouts={"camus": 0.2})
        self.heartbeat()
        self.assertEqual(self.bridge.process_task(), "error")
        self.assertIn("超时", self.cloud.collections["tasks"][task_id]["error"])

    def test_07_offline_board_blocks_tasks(self):
        dead = MockBoard().start()
        port = dead.entry_port
        dead.stop()
        self.bridge = make_bridge(self.mock, self.cloud)
        self.bridge.board = BoardClient("127.0.0.1", port, port, port,
                                        MockBoard.HUB_TOKEN)
        state = self.heartbeat()
        self.assertFalse(state["online"])
        board_doc = self.cloud.collections["board"]["board_current"]
        self.assertFalse(board_doc["online"])
        self.assertIsNone(self.bridge.process_task())

    def test_08_stale_running_task_recovered(self):
        task_id = self.cloud.add_doc("tasks", {
            "type": "camus", "status": "running", "inputs": [],
            "created_at": int(time.time() * 1000) - 10 ** 7,
            "started_at": int(time.time() * 1000) - 10 ** 7})
        self.bridge.recover_stale_tasks()
        self.assertEqual(self.cloud.collections["tasks"][task_id]["status"],
                         "error")

    @unittest.skipUnless(PILLOW, "Pillow 未安装时跳过皮肤镜全流程")
    def test_09_skin_task_end_to_end(self):
        png = fake_png()
        fid = self.cloud.upload_bytes("tasks/k-skin/in/a.png", png)
        task_id = seed_task(self.cloud, "skin",
                            [{"fileID": fid, "name": "a.png"}],
                            {"thickness": 3})
        self.heartbeat()
        self.assertEqual(self.bridge.process_task(), "done")
        task = self.cloud.collections["tasks"][task_id]
        self.assertEqual(task["status"], "done")
        labels = [v["label"] for v in task["result"]["views"]]
        self.assertEqual(len(labels), 6)
        values = {m["label"]: m["value"] for m in task["result"]["metrics"]}
        self.assertEqual(values.get("病变面积占比"), "23.45%")

    @unittest.skipUnless(PILLOW, "Pillow 未安装时跳过")
    def test_10_skin_paused_becomes_waiting_model(self):
        self.mock.hub_suspended_model = "skin"
        png = fake_png()
        fid = self.cloud.upload_bytes("tasks/k2/in/a.png", png)
        task_id = seed_task(self.cloud, "skin",
                            [{"fileID": fid, "name": "a.png"}])
        self.heartbeat()
        self.assertEqual(self.bridge.process_task(), "waiting")
        task = self.cloud.collections["tasks"][task_id]
        self.assertEqual(task["status"], "waiting_model")
        self.assertIn("切换", task["note"])

    def test_11_dual_model_switch_uses_authenticated_hub(self):
        self.mock.entry_enabled = False
        self.heartbeat()
        cmd_id = self.cloud.add_doc("commands", dict(type="switch", model="camus", status="pending", created_at=1))
        self.assertEqual(self.bridge.process_commands(), "done")
        self.assertEqual(self.cloud.collections["commands"][cmd_id]["status"], "done")
        self.assertEqual(self.bridge.current_model()["model"], "camus")
        self.assertEqual(self.mock.entry_switch_calls, [])

    def test_12_real_acdc_busy_response_retries(self):
        self.mock.entry_state = {"model": "acdc", "phase": "ready"}
        self.mock.acdc_busy_polls = 1
        ed = self.cloud.upload_bytes("ED.nii", b"ED")
        es = self.cloud.upload_bytes("ES.nii", b"ES")
        task_id = seed_task(self.cloud, "acdc", [{"name": "ED.nii", "fileID": ed}, {"name": "ES.nii", "fileID": es}])
        self.heartbeat()
        self.assertEqual(self.bridge.process_task(), "done")
        self.assertEqual(self.cloud.collections["tasks"][task_id]["status"], "done")
        self.assertEqual(len(self.mock.acdc_uploads), 1)


if __name__ == "__main__":
    unittest.main()
