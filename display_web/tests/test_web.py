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
from server import Console, make_server, decode_upload, persistent_token


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

    def test_unhealthy_hdmi_prevents_restart(self):
        p=Mock();p.poll.return_value=1;p.returncode=1
        self.app.hdmi=p
        with self.assertRaises(RuntimeError): self.app.stop_hdmi()
        self.assertIsNotNone(self.app.fault)


if __name__=="__main__":unittest.main()
