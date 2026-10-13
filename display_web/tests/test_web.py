import base64
import io
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from unittest.mock import patch, Mock
import numpy as np
from PIL import Image
from contour import extract_contour
from pld_hdmi import fit_rgb
from server import (Console, make_server, decode_upload, persistent_token,
                    PNG_COMPRESS_LEVEL, PNG_SAVE_WORKERS, VIEWS)


def encoded(a):
    b = io.BytesIO(); Image.fromarray(a).save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


class WebTests(unittest.TestCase):
    def test_history_survives_service_restart_but_not_reboot(self):
        self.app.boot_id = 'boot-a'
        j = self.wait(self.app.submit({'mode':'demo'})['id'])
        self.assertEqual(j['state'],'done')
        code, raw = self.request('/api/history')
        self.assertEqual(code,200)
        self.assertEqual(json.loads(raw)['items'][0]['id'],j['id'])
        same = Console(self.temp.name, token='test', boot_id='boot-a')
        self.assertEqual(same.history()['items'][0]['id'],j['id'])
        self.assertTrue((Path(self.temp.name)/j['id']/'results.zip').is_file())
        other = Console(self.temp.name, token='test', boot_id='boot-b')
        self.assertEqual(other.history()['items'],[])
        with self.assertRaises(KeyError): other.job(j['id'])

    def test_history_excludes_failed_incomplete_and_broken_records(self):
        self.app.boot_id = 'boot-a'
        j = self.wait(self.app.submit({'mode':'demo'})['id'])
        folder=Path(self.temp.name)/j['id']
        (folder/'results.zip').unlink()
        again=Console(self.temp.name, token='test', boot_id='boot-a')
        self.assertEqual(again.history()['items'],[])
        (folder/'job.json').write_text('{broken')
        self.assertEqual(Console(self.temp.name,token='test',boot_id='boot-a').history()['items'],[])
        self.app.jobs['failed']={'state':'error'}
        self.app.jobs['pending']={'state':'running'}
        self.assertEqual(len(self.app.history()['items']),1)
        self.assertEqual(self.request('/api/history',token=False)[0],401)

    def test_history_order_and_uploaded_cutout(self):
        first=self.wait(self.app.submit({'mode':'demo'})['id'])
        rgb=np.full((12,12,3),100,np.uint8)
        mask=np.zeros((12,12),np.uint8);mask[3:9,3:9]=255
        last=self.wait(self.app.submit({'mode':'mask','image':encoded(rgb),'mask':encoded(mask),'name':'图像.png'})['id'])
        entries=self.app.history()['items']
        self.assertEqual([j['id'] for j in entries],[last['id'],first['id']])
        code,content=self.request('/files/'+last['id']+'/cutout.png')
        self.assertEqual(code,200)
        a=np.asarray(Image.open(io.BytesIO(content)))
        self.assertTrue((a[mask==0]==0).all())
        self.assertTrue((a[mask!=0]==100).all())

    def test_access_survives_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            a = Console(folder)
            b = Console(folder)
            self.assertEqual(a.token, b.token)
            self.assertEqual(len(a.token),32)
            Path(folder,".access_token").write_text("corrupted")
            with self.assertRaises(ValueError): Console(folder)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = Console(self.temp.name, token="test-only-token")
        self.app.board = False; self.app.runtime_ok = False
        self.server = make_server(self.app, port=0)
        self.url = "http://127.0.0.1:"+str(self.server.server_port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(); self.temp.cleanup()

    def request(self, path, body=None, token=True, origin=None):
        headers = {"Authorization":"Bearer test-only-token"} if token else {}
        if origin: headers["Origin"] = origin
        req = urllib.request.Request(self.url+path, data=json.dumps(body).encode() if body is not None else None, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as r: return r.status, r.read()
        except urllib.error.HTTPError as e: return e.code, e.read()

    def wait(self, ident):
        deadline = time.monotonic()+8
        while time.monotonic()<deadline:
            j = self.app.job(ident)
            if j["state"]!="running": return j
            time.sleep(.01)
        self.fail("job timeout")

    def test_page_and_auth(self):
        code, html = self.request("/", token=False)
        self.assertEqual(code,200); self.assertIn("图像分割工作台".encode(),html)
        self.assertEqual(self.request("/api/status",token=False)[0],401)
        self.assertEqual(self.request("/api/status")[0],200)

    def test_cross_origin_rejected(self):
        self.assertEqual(self.request("/api/jobs",{"mode":"demo"},origin="http://foreign.invalid")[0],403)

    def test_real_demo_and_download(self):
        code, raw = self.request("/api/jobs",{"mode":"demo"})
        self.assertEqual(code,202)
        j = self.wait(json.loads(raw)["id"])
        self.assertEqual(j["state"],"done"); self.assertNotIn("original",j["views"])
        self.assertAlmostEqual(j["statistics"]["area_percent"],35.467529296875)
        code, content = self.request("/files/"+j["id"]+"/results.zip")
        self.assertEqual(code,200)
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            self.assertIn("contour.png",z.namelist()); self.assertIn("report.json",z.namelist())

    def test_upload_postprocess(self):
        rgb = np.full((40,60,3),120,np.uint8)
        mask = np.zeros((40,60),np.uint8); mask[10:30,20:40]=255
        code, raw = self.request("/api/jobs",{"mode":"mask","image":encoded(rgb),"mask":encoded(mask),"name":"test.png"})
        self.assertEqual(code,202)
        j=self.wait(json.loads(raw)["id"])
        self.assertEqual(j["state"],"done"); self.assertIn("original",j["views"])
        self.assertEqual(j["statistics"]["foreground_pixels"],400)
        self.assertFalse(j["statistics"].get("b_seconds"))

    def test_output_pixels_zip_members_and_stage_timings(self):
        rgb = (np.arange(40 * 60 * 3, dtype=np.uint32).reshape((40, 60, 3)) % 251).astype(np.uint8)
        mask = np.zeros((40, 60), np.uint8); mask[9:31, 17:43] = 255
        expected_contour = extract_contour(mask != 0, thickness=2)
        j = self.wait(self.app.submit({"mode":"mask","image":encoded(rgb),
                                        "mask":encoded(mask),"thickness":2})["id"])
        self.assertEqual(j["state"],"done")
        self.assertEqual(j["views"],["original","mask","contour","outlined","cutout","comparison"])
        folder = Path(self.temp.name)/j["id"]
        decoded = {name: np.asarray(Image.open(folder/(name+".png")))
                   for name in j["views"]}
        np.testing.assert_array_equal(decoded["original"], rgb)
        np.testing.assert_array_equal(decoded["mask"], mask)
        np.testing.assert_array_equal(decoded["contour"], expected_contour.astype(np.uint8)*255)
        expected_outlined = rgb.copy(); expected_outlined[expected_contour] = (255, 220, 0)
        np.testing.assert_array_equal(decoded["outlined"], expected_outlined)
        expected_cutout = rgb.copy(); expected_cutout[mask == 0] = 0
        np.testing.assert_array_equal(decoded["cutout"], expected_cutout)
        expected_comparison = np.concatenate((fit_rgb(rgb,640,480), fit_rgb(expected_outlined,640,480)),axis=1)
        np.testing.assert_array_equal(decoded["comparison"], expected_comparison)

        expected_members = [name+".png" for name in j["views"]] + ["report.json"]
        with zipfile.ZipFile(folder/"results.zip") as z:
            self.assertEqual(z.namelist(), expected_members)
            self.assertTrue(all(info.compress_type == zipfile.ZIP_STORED for info in z.infolist()))
            self.assertEqual(z.read("report.json"), (folder/"report.json").read_bytes())
        report = json.loads((folder/"report.json").read_text(encoding="utf-8"))
        self.assertIn("png_save", report["stage_seconds"])
        self.assertIn("zip_payload", report["stage_seconds"])
        self.assertNotIn("report_write", report["stage_seconds"])
        self.assertNotIn("zip_package", report["stage_seconds"])
        self.assertNotIn("job_total", report["stage_seconds"])
        self.assertIn("stage_timing_scope", report)
        self.assertNotIn("final_stage_seconds", report)
        stages = j["statistics"]["final_stage_seconds"]
        self.assertEqual(set(stages["png_save"]), set(j["views"]))
        self.assertGreater(j["statistics"]["processing_seconds"],0)
        for name in ("report_write","zip_report_append","zip_package","job_total","request_to_job_complete"):
            self.assertIn(name,stages)
        self.assertGreaterEqual(stages["job_total"],j["statistics"]["processing_seconds"])
        self.assertGreaterEqual(stages["request_to_job_complete"],stages["job_total"])
        self.assertEqual(j["statistics"]["job_total_seconds"],stages["job_total"])
        self.assertEqual(j["statistics"]["request_to_job_complete_seconds"],stages["request_to_job_complete"])
        for value in stages["png_save"].values():
            self.assertGreaterEqual(value,0)
        self.assertEqual(j["statistics"]["output_png_compress_level"],PNG_COMPRESS_LEVEL)
        self.assertEqual(j["statistics"]["output_png_workers"],PNG_SAVE_WORKERS)
        self.assertEqual(j["statistics"]["output_zip_compression"],"stored")

    def test_png_saves_overlap_on_two_bounded_worker_threads(self):
        rgb = (np.arange(48 * 64 * 3, dtype=np.uint32).reshape((48, 64, 3)) % 199).astype(np.uint8)
        mask = np.zeros((48, 64), np.uint8); mask[10:40, 12:52] = 255
        image_payload = encoded(rgb); mask_payload = encoded(mask)
        saves = []
        # Every real save must rendezvous with a second saver. A regression to
        # serial saving breaks the barrier after the bounded timeout, failing
        # the job quickly instead of hanging the suite.
        barrier = threading.Barrier(PNG_SAVE_WORKERS, timeout=3)
        original_save = Image.Image.save

        def coordinated_save(self, fp, *args, **kwargs):
            saves.append((threading.get_ident(), Path(fp).name))
            barrier.wait()
            return original_save(self, fp, *args, **kwargs)

        with patch.object(Image.Image, "save", coordinated_save):
            j = self.wait(self.app.submit({"mode":"mask","image":image_payload,
                                           "mask":mask_payload,"thickness":2})["id"])
        self.assertEqual(j["state"],"done")
        self.assertEqual(j["views"],list(VIEWS))
        names = [name for _, name in saves]
        self.assertEqual(sorted(names),sorted(name+".png" for name in VIEWS))
        self.assertEqual(len(names),len(set(names)))
        thread_ids = {thread_id for thread_id, _ in saves}
        self.assertEqual(len(thread_ids),PNG_SAVE_WORKERS)
        self.assertNotIn(threading.get_ident(),thread_ids)
        folder = Path(self.temp.name)/j["id"]
        np.testing.assert_array_equal(np.asarray(Image.open(folder/"original.png")),rgb)
        np.testing.assert_array_equal(np.asarray(Image.open(folder/"mask.png")),mask)
        with zipfile.ZipFile(folder/"results.zip") as z:
            self.assertEqual(z.namelist(),[name+".png" for name in VIEWS]+["report.json"])

    def test_png_save_failure_fails_job_without_package_or_history(self):
        rgb = np.full((32,40,3),90,np.uint8)
        mask = np.zeros((32,40),np.uint8); mask[8:24,10:30] = 255
        image_payload = encoded(rgb); mask_payload = encoded(mask)
        workers = []
        original_save = Image.Image.save

        def failing_save(self, fp, *args, **kwargs):
            workers.append(threading.current_thread())
            if Path(fp).name == "outlined.png":
                raise OSError("simulated PNG encoder failure")
            return original_save(self, fp, *args, **kwargs)

        with patch.object(Image.Image, "save", failing_save):
            j = self.wait(self.app.submit({"mode":"mask","image":image_payload,
                                           "mask":mask_payload})["id"])
        self.assertEqual(j["state"],"error")
        self.assertIn("simulated PNG encoder failure",j["error"])
        folder = Path(self.temp.name)/j["id"]
        self.assertFalse((folder/"results.zip").exists())
        self.assertFalse((folder/"report.json").exists())
        self.assertFalse((folder/"job.json").exists())
        self.assertEqual([item["id"] for item in self.app.history()["items"]],[])
        # The bounded savers must all have exited before the job reports error.
        for worker in set(workers):
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())

    def test_mismatch_and_invalid_mask(self):
        rgb=np.zeros((3,3,3),np.uint8)
        for mask in (np.zeros((2,2),np.uint8),np.full((3,3),2,np.uint8)):
            self.assertEqual(self.request("/api/jobs",{"mode":"mask","image":encoded(rgb),"mask":encoded(mask)})[0],400)

    def test_corrupt_file_and_options(self):
        self.assertEqual(self.request("/api/jobs",{"mode":"mask","image":"%%%"})[0],400)
        self.assertEqual(self.request("/api/jobs",{"mode":"demo","thickness":100})[0],400)
        self.assertEqual(self.request("/api/jobs",{"mode":"unknown"})[0],400)

    def test_pc_never_runs_fpga_or_hdmi(self):
        with patch("server.subprocess.Popen") as child:
            self.assertEqual(self.request("/api/jobs",{"mode":"infer"})[0],400)
            self.assertEqual(self.request("/api/hdmi",{"action":"show","id":"fake","view":"mask"})[0],400)
            child.assert_not_called()

    def test_busy_and_fault_block_new_jobs(self):
        self.app.active="busy"
        self.assertEqual(self.request("/api/jobs",{"mode":"demo"})[0],409)
        self.app.active=None; self.app.fault="hardware fault"
        self.assertEqual(self.request("/api/jobs",{"mode":"demo"})[0],409)

    def test_no_arbitrary_file_read(self):
        self.assertEqual(self.request("/files/../server.py")[0],404)
        j=self.wait(self.app.submit({"mode":"demo"})["id"])
        self.assertEqual(self.request("/files/"+j["id"]+"/hdmi.log")[0],404)

    def test_hdmi_exclusion_and_order(self):
        self.app.board=True
        j=self.wait(self.app.submit({"mode":"demo"})["id"])
        old=Mock(); old.poll.return_value=None; old.wait.return_value=0; old.returncode=0
        self.app.hdmi=old; self.app.hdmi_job=j["id"]
        new=Mock(); new.poll.return_value=None
        def spawn(*args,**kwargs):
            old.terminate.assert_called_once(); old.wait.assert_called_once()
            return new
        with patch("server.subprocess.Popen",side_effect=spawn):
            self.app.start_hdmi(j["id"],"contour")
        self.assertEqual(self.app.status()["hdmi"],"starting")
        self.app.active="processing"
        with self.assertRaises(RuntimeError): self.app.start_hdmi(j["id"],"mask")
        self.app.hdmi=None

    def test_inference_waits_for_hdmi_release(self):
        self.app.runtime_ok=True; self.app.runtime=Path(self.temp.name)
        arr=np.zeros((8,8,3),np.uint8); raw=io.BytesIO();Image.fromarray(arr).save(raw,format="JPEG")
        events=[]
        fake=Mock();fake.wait.return_value=1
        def spawn(*args,**kwargs): events.append("inference");return fake
        with patch.object(self.app,"stop_hdmi",side_effect=lambda:events.append("stop_hdmi")), patch("server.verify_runtime"), patch("server.inference_command",return_value=["fake"]), patch("server.subprocess.Popen",side_effect=spawn):
            j=self.wait(self.app.submit({"mode":"infer","image":base64.b64encode(raw.getvalue()).decode()})["id"])
        self.assertEqual(events,["stop_hdmi","inference"])
        self.assertEqual(j["state"],"error");self.assertIsNotNone(self.app.fault)
        stages=j["stage_seconds"]
        for name in ("stop_hdmi","input_write","runtime_verify","inference_launch",
                     "inference_wait","inference_subprocess"):
            self.assertIn(name,stages)
        folder=Path(self.temp.name)/j["id"]
        self.assertTrue((folder/"input.jpg").is_file())
        self.assertFalse(list(folder.glob("*.png")))
        self.assertFalse((folder/"report.json").exists())
        self.assertFalse((folder/"results.zip").exists())

    def test_unhealthy_hdmi_prevents_restart(self):
        p=Mock();p.poll.return_value=1;p.returncode=1
        self.app.hdmi=p
        with self.assertRaises(RuntimeError): self.app.stop_hdmi()
        self.assertIsNotNone(self.app.fault)


if __name__=="__main__":unittest.main()
