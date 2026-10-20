"""End-to-end cache, replay, and serialization checks using synthetic heads."""
import base64
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'display_web/unified'))
import hub
import interactive_bridge as bridge
from prompt_seg.service import SegmentationService


class Board:
    def __init__(self,data):
        self.data=data
        self.jobs=0
        self.states={'skin':'inactive','camus':'active'}

    def props(self,model):return {'ActiveState':self.states[model]}
    def status(self,model):return dict(active=False,fault=False,warmup='ready',ready=True,backend='pl',mode='board',inference_available=True,hdmi='off')
    def stop(self,model):self.states[model]='inactive'
    def start(self,model):self.states[model]='active'
    def request(self,model,method,path,body=None):
        assert model=='camus' and method=='POST' and path=='/api/jobs'
        self.jobs+=1
        ident=f'{self.jobs:032x}'
        out=self.data/ident/'output'
        out.mkdir(parents=True)
        head=np.full((4,256,256),-24,np.int8)
        head[0]=8
        head[1,60:190,65:195]=32
        (out/'head.bin').write_bytes(head.tobytes())
        (out/'head.json').write_text(json.dumps(dict(spec_sha256=bridge.SPECS['camus'],shape=[1,4,256,256],dtype='int8',scale=.1)))
        return 200,'application/json',json.dumps(dict(id=ident,state='done')).encode()


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)
        self.board=Board(self.path)
        c=hub.Controller(self.board,self.path)
        c.model='camus';c.phase='ready';c.generation=7
        m=bridge.Manager.__new__(bridge.Manager)
        m.controller=c;m.guard=threading.Lock();m.job=m.thread=None
        m.image_slot=None;m.display_cache_key=m.display_cache=None
        m.backends={'camus':{'data':str(self.path)}}
        m.predictor=bridge.Predictor(m);m.service=SegmentationService(m.predictor)
        c.interactive=m
        self.c=c;self.m=m
        self.image=np.full((40,48,3),100,np.uint8)
        stream=io.BytesIO();Image.fromarray(self.image).save(stream,format='PNG')
        self.encoded='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()

    def tearDown(self):
        if self.m.thread:self.m.thread.join(5)
        if self.c.thread:self.c.thread.join(5)
        self.temp.cleanup()

    def upload(self,**extra):
        return self.m.upload(dict(model='camus',generation=7,image=self.encoded,**extra))

    def run_job(self,token,params):
        job=self.m.submit(dict(model='camus',generation=7,image_token=token,parameters=dict(variant='candidate',backend='int8',**params)))
        self.m.thread.join(5)
        self.assertFalse(self.m.thread.is_alive())
        row=self.m.snapshot(job['id'])
        self.assertEqual(row['state'],'done',row)
        return row['result']

    def record(self):
        uploaded=self.upload()
        result=self.run_job(uploaded['image_token'],{})
        record=copy.deepcopy(result['replay'])
        record.update(image_png=self.encoded,original={'width':96,'height':80,'transform':{'mode':'fit_long_side_1024','processed_width':48,'processed_height':40}},outputs=result['outputs'],baseline_outputs=result['baseline']['outputs'])
        return record

    def test_one_upload_one_prediction_with_real_head_restore(self):
        with patch.object(bridge,'decode_image',wraps=bridge.decode_image) as decoded:
            uploaded=self.upload()
            first=self.run_job(uploaded['image_token'],{})
            second=self.run_job(uploaded['image_token'],{'points':[dict(x=12,y=10,label='background')]})
            third=self.run_job(uploaded['image_token'],{'points':[dict(x=25,y=20,label='foreground')]})
            self.assertEqual(decoded.call_count,1)
            self.assertEqual(self.board.jobs,1)
            self.assertTrue(first['execution']['pl_executed_this_request'])
            self.assertFalse(second['execution']['pl_executed_this_request'])
            self.assertFalse(third['execution']['pl_executed_this_request'])
            self.assertTrue(second['baseline']['cache_hit'])
            self.assertTrue(third['baseline']['cache_hit'])
            self.assertEqual(first['label_hashes']['baseline'],second['label_hashes']['baseline'])
            self.assertTrue(third['refinement']['context_cache_hit'])
            self.assertNotIn('backend_job_wait',third['identity']['prediction_timings_ms'])
            roi=self.run_job(uploaded['image_token'],{'roi':dict(x0=4,y0=4,x1=44,y1=36)})
            self.assertEqual(self.board.jobs,2)
            self.assertTrue(roi['execution']['pl_executed_this_request'])
            self.assertFalse(roi['baseline']['cache_hit'])
            self.assertEqual(decoded.call_count,1)

    def test_replay_canonical_backend_and_live_model_variant(self):
        record=self.record()
        check=self.m.replay_check({'replay':record})
        self.assertEqual(check['parameters']['backend'],'int8')
        self.assertTrue(check['replay_ready'])
        for values in [dict(backend='fp32'),dict(variant='delivery')]:
            with self.subTest(values=values):
                bad=copy.deepcopy(record);bad['parameters'].update(values)
                with self.assertRaises(ValueError):self.m.replay_check({'replay':bad})
        bad=copy.deepcopy(record);bad['results']['final_labels_sha256']='broken'
        with self.assertRaises(ValueError):self.m.replay_check({'replay':bad})
        for invalid_width in (True, 48.5, '48'):
            bad=copy.deepcopy(record);bad['image']['width']=invalid_width
            with self.assertRaises(ValueError):self.m.replay_check({'replay':bad})

    def test_frozen_v4_manifest_is_accepted_as_upgrade_source(self):
        from install_interactive import HUB
        frozen=json.loads((ROOT/'delivery/pld_dual_20261017/MANIFEST.json').read_text())['files']
        accepted=json.loads((ROOT/'display_web/unified/interactive_accepted_versions.json').read_text())
        for name in HUB:
            self.assertIn(frozen['hub/'+name],accepted['hub/'+name],name)

    def test_bad_expected_hash_preserves_original_token(self):
        first=self.upload()
        with self.assertRaises(ValueError):self.upload(expected_sha256='0'*64)
        _,token=self.m._image_for({'image_token':first['image_token']})
        self.assertEqual(token,first['image_token'])

    def test_switch_waits_for_atomic_upload_and_clears_old_slot(self):
        entered=threading.Event();release=threading.Event();switched=threading.Event()
        real=bridge.decode_image
        def blocked(value):
            entered.set()
            if not release.wait(3):raise RuntimeError('test release timed out')
            return real(value)
        uploaded=[];errors=[]
        def upload():
            try:uploaded.append(self.upload())
            except Exception as exc:errors.append(exc)
        def switch():
            try:self.c.switch('skin');switched.set()
            except Exception as exc:errors.append(exc)
        with patch.object(bridge,'decode_image',side_effect=blocked):
            a=threading.Thread(target=upload);a.start()
            try:
                self.assertTrue(entered.wait(2))
                b=threading.Thread(target=switch);b.start()
                self.assertFalse(switched.wait(.05),'switch overtook image validation')
            finally:
                release.set();a.join(3)
                if 'b' in locals():b.join(3)
        self.assertEqual(errors,[])
        self.assertEqual(len(uploaded),1)
        self.assertTrue(switched.is_set())
        self.assertIsNone(self.m.image_slot)
        with self.assertRaises(ValueError):self.m._image_for({'image_token':uploaded[0]['image_token']})


if __name__=='__main__':unittest.main(verbosity=2)
