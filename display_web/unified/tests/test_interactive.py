import base64
import io
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'display_web/unified'))
from prompt_seg.contracts import Request,InputError
from prompt_seg.refinement import seeded_cut
from prompt_seg.service import SegmentationService
from interactive_bridge import decode_image,Predictor

class FakePredictor:
    def predict(self,image,model,*args):
        self.shape=image.shape
        p=np.zeros((2 if model=='skin' else 4,*image.shape[:2]),np.float32)
        p[0]=.2;p[1]=.8
        return p,{'prediction_cache_hit':False}

def labels(result):
    return np.array(Image.open(io.BytesIO(base64.b64decode(result['outputs']['labels_png'].split(',')[1]))))

class PromptTests(unittest.TestCase):
    def test_seeds_and_box(self):
        rgb=np.zeros((32,48,3),np.uint8);prior=np.full((32,48),.8)
        mask,info=seeded_cut(rgb,prior,[(15,15,True),(20,20,False)],(8,8,40,28))
        self.assertTrue(mask[15,15]);self.assertFalse(mask[20,20]);self.assertFalse(mask[:8].any());self.assertFalse(mask[:,40:].any())
        self.assertLess(mask.sum(),32*48)
    def test_roi_precedes_inference_and_remaps(self):
        pred=FakePredictor();svc=SegmentationService(pred)
        row=svc.process('skin',np.zeros((32,40,3),np.uint8),{'roi':dict(x0=10,y0=8,x1=30,y1=24)})
        self.assertEqual(pred.shape,(16,20,3));a=labels(row)
        self.assertEqual(int(a.sum()),320);self.assertFalse(a[:8].any());self.assertEqual(row['statistics']['processed_pixels'],320)
    def test_reject_invalid_prompts(self):
        for params in [{'points':[dict(x=-1,y=2,label='foreground')]},{'points':[dict(x=2,y=2,label='foreground'),dict(x=2,y=2,label='background')]},{'threshold':float('nan')},{'roi':dict(x0=2,y0=2,x1=3,y1=3)},{'target_class':3}]:
            with self.assertRaises(InputError):Request.parse('skin',params,32,32)
    def test_camus_selected_structure(self):
        svc=SegmentationService(FakePredictor())
        row=svc.process('camus',np.zeros((32,32,3),np.uint8),{'target_class':2,'points':[dict(x=16,y=16,label='foreground')],'box':dict(x0=8,y0=8,x1=24,y1=24)})
        a=labels(row);self.assertEqual(a[16,16],2);self.assertEqual(a[0,0],1)
    def test_decode_limits(self):
        b=io.BytesIO();Image.new('RGB',(1025,1024)).save(b,format='PNG')
        with self.assertRaises(ValueError):decode_image(base64.b64encode(b.getvalue()).decode())
    def test_cache_scoped_by_generation(self):
        class C:generation=1
        class M:controller=C()
        pred=Predictor(M());image=np.zeros((8,8,3),np.uint8)
        import hashlib
        pred.cache_key=(1,'skin',image.shape,hashlib.sha256(image.tobytes()).hexdigest());pred.cached=np.zeros((2,8,8));pred.identity={}
        p,identity=pred.predict(image,'skin','delivery','int8');self.assertTrue(identity['prediction_cache_hit'])
        p[0]=5;self.assertEqual(pred.cached.max(),0)
        pred.clear();self.assertIsNone(pred.cached)

if __name__=='__main__':unittest.main()

class BridgeTests(unittest.TestCase):
    def test_camus_reads_real_head_and_caches_without_second_job(self):
        import tempfile,json,threading
        from interactive_bridge import SPECS
        with tempfile.TemporaryDirectory() as temp:
            ident='a'*32;out=Path(temp)/ident/'output';out.mkdir(parents=True)
            head=np.zeros((4,256,256),np.int8);head[2]=20
            (out/'head.bin').write_bytes(head.tobytes());(out/'head.json').write_text(json.dumps(dict(spec_sha256=SPECS['camus'],shape=[1,4,256,256],dtype='int8',scale=.1)))
            class C:generation=2
            class M:
                controller=C();backends={'camus':{'data':temp}};calls=0
                def request(self,*args):
                    # Exercise the real frozen reader, rather than bypassing
                    # the upload contract with a fabricated successful job.
                    from camus.preprocess import load_gray_frame
                    payload=args[3]['files'][0]
                    path=Path(temp)/payload['name']
                    path.write_bytes(base64.b64decode(payload['data_base64']))
                    gray=load_gray_frame(path).data
                    np.testing.assert_array_equal(gray,np.asarray(Image.fromarray(rgb).convert('L')))
                    self.calls+=1
                    return dict(id=ident,state='done')
            manager=M();pred=Predictor(manager);rgb=np.zeros((48,64,3),np.uint8)
            rgb[:,:,0]=130;rgb[:,:,1]=85
            probs,identity=pred.predict(rgb,'camus','candidate','int8')
            self.assertEqual(probs.shape,(4,48,64));self.assertTrue((probs.argmax(0)==2).all())
            np.testing.assert_allclose(probs.sum(0),1,rtol=1e-6)
            pred.predict(rgb,'camus','candidate','int8');self.assertEqual(manager.calls,1)
            manager.controller.generation+=1;pred.predict(rgb,'camus','candidate','int8');self.assertEqual(manager.calls,2)
    def test_png_optimization_preserves_pixels_and_timing_covers_encoding(self):
        from prompt_seg.service import png_data
        image=np.random.RandomState(42).randint(0,256,(37,59,3),dtype=np.uint8)
        decoded=np.asarray(Image.open(io.BytesIO(base64.b64decode(png_data(image).split(',')[1]))))
        np.testing.assert_array_equal(decoded,image)
        row=SegmentationService(FakePredictor()).process('skin',image,{})
        self.assertTrue(all(v>=0 for v in row['timings_ms'].values()))
        self.assertLessEqual(sum(row['timings_ms'].values()),row['elapsed_ms']+.02)
        for array in (image[:,:,0],image[:,:1],image[:1,:,0]):
            decoded=np.array(Image.open(io.BytesIO(base64.b64decode(png_data(array).split(',')[1]))))
            np.testing.assert_array_equal(decoded,array)
    def test_manager_caches_display_images_and_counts_all_outputs(self):
        import threading
        from types import SimpleNamespace
        from interactive_bridge import Manager
        manager=Manager.__new__(Manager)
        manager.controller=SimpleNamespace(generation=1,lock=threading.RLock(),interactive_busy=True)
        manager.guard=threading.Lock();manager.job=None
        manager.display_cache_key=None;manager.display_cache=None
        manager.service=SegmentationService(FakePredictor());manager.predictor=Predictor(manager)
        image=np.zeros((24,32,3),np.uint8)
        manager.run('a'*32,'skin',image,{})
        first=manager.snapshot('a'*32)['result']
        manager.run('b'*32,'skin',image,{'roi':dict(x0=2,y0=2,x1=30,y1=22)})
        second=manager.snapshot('b'*32)['result']
        self.assertFalse(first['identity']['display_cache_hit'])
        self.assertTrue(second['identity']['display_cache_hit'])
        self.assertEqual(first['outputs']['original_png'],second['outputs']['original_png'])
        self.assertEqual(len(second['outputs']),5)
        self.assertLessEqual(sum(second['timings_ms'].values()),second['elapsed_ms']+.02)
        manager.clear();self.assertIsNone(manager.display_cache)
    def test_switch_and_other_posts_blocked_during_interactive(self):
        import tempfile
        import hub
        with tempfile.TemporaryDirectory() as temp:
            c=hub.Controller(None,temp);c.phase='ready';c.model='skin';c.interactive_busy=True
            with self.assertRaises(RuntimeError):c.switch('camus')
            with self.assertRaises(RuntimeError):c.proxy('skin',0,'POST','/api/jobs',b'{}','application/json')

class LegacyDependencyTests(unittest.TestCase):
    def test_legacy_pillow_and_scipy_run_real_cut(self):
        import types
        import prompt_seg.refinement as refinement
        real_flow=refinement.maximum_flow
        def legacy_flow(graph,source,sink):
            result=real_flow(graph,source,sink,method='edmonds_karp')
            return types.SimpleNamespace(flow_value=result.flow_value,residual=result.flow)
        legacy_image=types.SimpleNamespace(fromarray=Image.fromarray,BILINEAR=2,NEAREST=0)
        with patch.object(refinement,'Image',legacy_image),patch.object(refinement,'maximum_flow',legacy_flow):
            mask,_=refinement.seeded_cut(np.zeros((16,16,3),np.uint8),np.full((16,16),.5),[(3,3,True),(12,12,False)],(2,2,14,14))
        self.assertTrue(mask[3,3]);self.assertFalse(mask[12,12]);self.assertFalse(mask[:2].any())
    def test_unrelated_type_error_not_suppressed(self):
        import prompt_seg.refinement as refinement
        with patch.object(refinement,'maximum_flow',side_effect=TypeError('invalid graph')):
            with self.assertRaisesRegex(TypeError,'invalid graph'):
                refinement.seeded_cut(np.zeros((8,8,3),np.uint8),np.full((8,8),.5),[(2,2,True)])
