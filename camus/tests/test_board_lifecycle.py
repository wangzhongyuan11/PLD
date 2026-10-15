import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from camus.board_service import unit_text
from camus.package import verify_runtime
from camus.web import CamusServer


class BoardLifecycleTests(unittest.TestCase):
    def test_xsa_sibling_is_rejected_before_pynq_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'overlay').mkdir();(root/'overlay/dma_top.xsa').write_bytes(b'xsa')
            with self.assertRaisesRegex(ValueError,'shadows'):verify_runtime(root,root/'model/int8_export')

    def test_service_conflicts_and_login_initialization(self):
        text=unit_text('/home/xilinx/camus','/home/xilinx/deps','/home/xilinx/data',8090)
        self.assertIn('Conflicts=pld-web.service',text)
        self.assertIn('After=pld-web.service',text)
        self.assertIn("ExecStart=/bin/bash -lc 'exec ",text)
        self.assertIn('--backend pl',text)
        with self.assertRaises(ValueError):unit_text("/home/xilinx/a'b",'/tmp/deps','/tmp/data',8090)

    def test_hdmi_release_then_next_job_reacquires_pl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);events=[]
            original=Mock();original.backend='pl';original.model.root=root/'model/int8_export'
            original.close.side_effect=lambda:events.append('release')
            process=Mock();process.poll.return_value=None;process.returncode=0
            process.wait.side_effect=lambda **_:events.append('hdmi_stopped')
            def launch(*a,**kw):
                self.assertEqual(events,['release']);events.append('hdmi_started');return process
            server=CamusServer(('127.0.0.1',0),original,root/'jobs',root)
            identifier='a'*32;folder=root/'jobs'/identifier/'output';folder.mkdir(parents=True)
            server.jobs[identifier]=dict(id=identifier,state='done')
            try:
                with patch('camus.web.subprocess.Popen',side_effect=launch):server.start_hdmi(identifier,'segmentation')
                self.assertTrue(server.released)
                replacement=Mock();replacement.run.return_value=dict(output_dir='x',report={},backend='pl')
                def acquire(*a,**kw):
                    self.assertIn('hdmi_stopped',events);events.append('acquire');return replacement
                job=dict(id='b'*32,state='queued',_started=0)
                server.busy.acquire()
                with patch('camus.web.CamusPipeline',side_effect=acquire):server.execute(job,'frame.nii.gz',None)
                self.assertEqual(job['state'],'done');self.assertFalse(server.released)
                self.assertEqual(events,['release','hdmi_started','hdmi_stopped','acquire'])
            finally:server.server_close()

    def test_failed_hdmi_release_blocks_new_hardware(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            pipeline=Mock();pipeline.model.root=Path(tmp)/'model/int8_export'
            server=CamusServer(('127.0.0.1',0),pipeline,Path(tmp)/'jobs')
            process=Mock();process.poll.return_value=None;process.wait.side_effect=subprocess.TimeoutExpired('hdmi',45)
            server.hdmi=process
            with self.assertRaisesRegex(RuntimeError,'did not confirm'):server.stop_hdmi()
            self.assertTrue(server.fault)
            server.hdmi=None;server.server_close()

    def test_http_rejects_cross_origin_hardware_control(self):
        import threading
        from urllib.request import Request,urlopen
        from urllib.error import HTTPError
        with tempfile.TemporaryDirectory() as tmp:
            pipeline=Mock();pipeline.backend='reference';pipeline.model.root=Path(tmp)/'model/int8_export'
            server=CamusServer(('127.0.0.1',0),pipeline,Path(tmp)/'jobs')
            thread=threading.Thread(target=server.serve_forever);thread.start()
            try:
                url='http://127.0.0.1:'+str(server.server_port)
                status=json.loads(urlopen(url+'/api/status',timeout=5).read())
                self.assertTrue(status['ready']);self.assertEqual(status['hdmi'],'off')
                req=Request(url+'/api/hdmi',data=b'{"action":"stop"}',headers={'Origin':'http://unrelated.invalid'})
                with self.assertRaises(HTTPError) as exc:urlopen(req,timeout=5)
                self.assertEqual(exc.exception.code,403)
            finally:server.shutdown();thread.join();server.server_close()


if __name__=='__main__':unittest.main()
