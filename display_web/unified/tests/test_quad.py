import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HAS_NUMPY = importlib.util.find_spec('numpy') is not None
if HAS_NUMPY:
    import numpy as np
    from PIL import Image
    import quad_display as quad
    import camus_quad


@unittest.skipUnless(HAS_NUMPY, 'NumPy/Pillow required')
class QuadTests(unittest.TestCase):
    def test_panels_are_distinct_and_inputs_unchanged(self):
        colors = [(240,20,20),(20,240,20),(20,20,240),(230,230,20)]
        images = [np.full((40,60,3),c,np.uint8) for c in colors]
        before = [im.copy() for im in images]
        frame = quad.compose(images)
        self.assertEqual(frame.shape,(480,640,3))
        for point,color in zip([(130,160),(130,480),(345,160),(345,480)],colors):
            np.testing.assert_array_equal(frame[point],color)
        for a,b in zip(images,before): np.testing.assert_array_equal(a,b)
    def test_enhancement_is_bounded_and_handles_constant(self):
        image = np.full((1200,1600,3),90,np.uint8)
        result = quad.enhance(image)
        self.assertEqual(result.size,(640,480))
        self.assertEqual(np.asarray(result).min(),90)
        self.assertEqual(image.min(),90)
    def test_skin_missing_original_is_marked_as_mask_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for name in ['mask','outlined']:
                Image.fromarray(np.zeros((32,32),np.uint8)).save(folder/(name+'.png'))
            self.assertEqual(quad.skin_frame(folder).shape,(480,640,3))
    def fake_web(self):
        class Pipeline:
            def run(self,*a,**k):return {'stages_seconds':{},'end_to_end_seconds':1.}
        class Server:
            def start_hdmi(self,*a):return 'legacy'
        class Handler:
            def do_GET(self):return 'legacy'
        return SimpleNamespace(CamusPipeline=Pipeline,CamusServer=Server,Handler=Handler,
                               ARTIFACTS=set(),urlsplit=lambda p:SimpleNamespace(path=p),
                               __file__='/runtime/camus/web.py')
    def test_hdmi_releases_before_launch_and_rejects_incomplete(self):
        web=self.fake_web();camus_quad.install(web)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'j/output';folder.mkdir(parents=True)
            (folder/'quad.png').write_bytes(b'fixture')
            events=[]
            server=web.CamusServer();server.pipeline=SimpleNamespace(backend='pl',close=lambda:events.append('close'))
            server.data=Path(temp);server.guard=threading.Lock();server.jobs={'j':{'state':'running'}}
            server.stop_hdmi=lambda:events.append('stop')
            with patch.object(camus_quad.subprocess,'Popen',side_effect=lambda *a,**k:events.append('launch')):
                with self.assertRaises(ValueError):server.start_hdmi('j','quad')
                self.assertEqual(events,[])
                server.jobs['j']['state']='done';server.start_hdmi('j','quad')
            self.assertEqual(events,['stop','close','launch'])
            self.assertTrue(server.released)
    def test_web_pipeline_does_not_wait_for_hdmi_quad(self):
        web=self.fake_web();camus_quad.install(web)
        with patch.object(camus_quad,'build_views',side_effect=AssertionError('quad must be deferred')):
            self.assertEqual(web.CamusPipeline().run('input','output')['end_to_end_seconds'],1.)
    def test_camus_preview_keeps_label_files_unchanged(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            mask=np.zeros((40,60,3),np.uint8);mask[5:20,10:30]=[255,20,20]
            for name in ['segmentation.png','overlay.png']:
                Image.fromarray(mask).save(folder/name)
            before={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['segmentation.png','overlay.png']}
            raw=np.arange(2400,dtype=np.float32).reshape(40,60)%256
            fake=SimpleNamespace(load_gray_frame=lambda _:SimpleNamespace(data=raw))
            with patch.dict(sys.modules,{'camus.preprocess':fake}):
                camus_quad.build_views('fixture',folder)
            with Image.open(folder/'quad.png') as im:
                self.assertEqual(im.size,(640,480))
            for name,sha in before.items():self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),sha)
            self.assertTrue((folder/'quad_original.png').is_file())
            self.assertTrue((folder/'quad_enhanced.png').is_file())

    def test_shared_renderers_match(self):
        root=Path(__file__).resolve().parents[2]
        self.assertEqual((root/'quad_display.py').read_bytes(),(root/'unified/quad_display.py').read_bytes())


if __name__ == '__main__':unittest.main()
