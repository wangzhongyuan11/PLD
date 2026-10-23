import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from http.server import ThreadingHTTPServer

import nibabel as nib
import numpy as np

from acdc.contract import atomic_json,file_sha
from acdc.web import JobManager,handler


class PresentationWebTest(unittest.TestCase):
    def test_saved_results_can_generate_views_without_backend_or_uploads(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);job='a'*32;result=root/job;(result/'labels').mkdir(parents=True)
            labels=np.zeros((6,6,4),np.uint8);labels[1:5,1:5,:]=2
            source=result/'labels/ED.nii.gz';nib.save(nib.Nifti1Image(labels,np.diag([2.,2.,8.,1.])),source)
            atomic_json(result/'status.json',dict(status='completed'))
            atomic_json(result/'manifest.json',dict(files={'labels/ED.nii.gz':file_sha(source)}))
            manager=JobManager(root,None);server=ThreadingHTTPServer(('127.0.0.1',0),handler(manager))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(base+'/api/capabilities') as response:cap=json.load(response)
                self.assertTrue(cap['read_only']);self.assertEqual(cap['backend'],'viewer')
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(base+'/api/jobs',data=b'not an upload',method='POST'))
                self.assertEqual(error.exception.code,400)
                self.assertFalse((root/'inputs').exists())
                with urlopen(Request(base+f'/api/jobs/{job}/presentation/ED',data=b'',method='POST')) as response:
                    report=json.load(response)
                self.assertEqual(report['settings']['generator'],'local_pc')
                self.assertTrue(report['meshes'][1]['smoothed'])
                with urlopen(base+report['base_url']+'MYO.vertices.f32') as response:
                    self.assertEqual(len(response.read()),report['meshes'][1]['vertices']*12)
                with urlopen(base+'/surface_view.js') as response:self.assertIn(b'CardiacSurfaceView',response.read())
                self.assertIsNone(manager.active)
            finally:server.shutdown();server.server_close();thread.join(2)
