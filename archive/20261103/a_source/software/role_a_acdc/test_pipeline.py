import unittest,tempfile,json
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import nibabel as nib
from .common import save_json,sha256
from .contract import write_board_request,read_board_response,HEAD_SHAPE
from .pipeline import segmentation_metrics,export_surface,cardiac_function

class PipelineTests(unittest.TestCase):
    def test_model_switch_clears_caches_and_serializes(self):
        from . import board_adapter as a
        class Device:pass
        class Graph:
            def __init__(self,model,device):self.device=device
            def forward(self,q):return {'head':np.zeros((1,4,256,256),np.int8)}
        device=Device();device._conv_plans={'old':1};device._lut_cache={'old':1};device._parameter_blobs={'old':1}
        one=SimpleNamespace(model_sha256='a'*64,spec={'profile_id':'acdc_axial_v1'})
        two=SimpleNamespace(model_sha256='b'*64,spec={'profile_id':'acdc_coronal_v1'})
        q=np.zeros((1,3,256,256),np.int8)
        with patch.object(a,'_cam2_imports',return_value=(Graph,Device,None,None,None)):
            a.forward(one,q,device);self.assertFalse(device._parameter_blobs)
            device._parameter_blobs['warm']=2;a.forward(one,q,device);self.assertIn('warm',device._parameter_blobs)
            a.forward(two,q,device);self.assertFalse(device._parameter_blobs);self.assertIs(device._acdc_active_model,two)
            device._acdc_serial_lock.acquire()
            try:
                with self.assertRaises(RuntimeError):a.forward(two,q,device)
            finally:device._acdc_serial_lock.release()
    def test_ef_and_invalid_phase_order(self):
        ed={'RV_cavity':150,'LV_cavity':100};es={'RV_cavity':60,'LV_cavity':40}
        r=cardiac_function(ed,es);self.assertEqual(r['LV_cavity']['ef_percent'],60)
        r=cardiac_function(es,ed);self.assertIsNone(r['LV_cavity']['ef_percent']);self.assertFalse(r['LV_cavity']['ef_valid_range'])
        r=cardiac_function({'RV_cavity':0,'LV_cavity':0},es);self.assertIsNone(r['RV_cavity']['raw_ef_percent'])
    def test_physical_distance_and_volume(self):
        a=np.zeros((9,9,9),np.uint8);b=a.copy();a[3,3,3]=1;b[4,3,3]=1
        affine=np.diag([2.5,3.,5.,1.]);r=segmentation_metrics(a,b,affine)
        self.assertEqual(r['1']['dice'],0)
        self.assertAlmostEqual(r['1']['hd95_mm'],2.5);self.assertAlmostEqual(r['1']['pred_volume_ml'],.0375)
        self.assertEqual(r['2']['surface_status'],'both_empty')
        a[1,1,1]=3;r=segmentation_metrics(a,b,affine)
        self.assertIsNone(r['3']['hd95_mm']);self.assertEqual(r['3']['surface_status'],'one_empty_undefined_distance')

    def test_mesh_uses_affine_not_display_pixels(self):
        a=np.zeros((8,8,8),np.uint8);a[2:5,2:5,2:5]=1
        affine=np.diag([2,3,5,1]);affine[:3,3]=[20,30,40]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'rv.ply';r=export_surface(a,affine,path,1)
            self.assertEqual(r['coordinate_system'],'RAS mm');self.assertGreater(r['vertices'],0)
            lines=path.read_text().split('end_header\n')[1].splitlines();points=np.array([[float(v) for v in line.split()[:3]] for line in lines[:r['vertices']]])
            np.testing.assert_allclose(points.min(0),[23,34.5,47.5]);np.testing.assert_allclose(points.max(0),[29,43.5,62.5])

    def test_board_result_bound_to_request_and_profile(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);req=write_board_request(d,np.zeros((1,3,256,256)),'axial','a'*64,
                    context={'run_id':'unit','volume_id':'synthetic','phase':'ED','center_index':0,'slice_count':5,'geometry_sha256':'c'*64},head_scale=.1)
            self.assertEqual(req['neighbor_indices'],[0,0,1])
            q=np.zeros(HEAD_SHAPE,np.int8);q[:,0]=-1;f=d/'head.bin';f.write_bytes(q.tobytes())
            response={k:req[k] for k in ('architecture_id','model_id','model_spec_sha256','request_id')}
            response.update(input_sha256=req['input']['sha256'],backend='pynq_pl',synthetic=False,head_scale=.1,
                            output={'path':f.name,'sha256':sha256(f),'shape':list(HEAD_SHAPE),'bytes':262144,'dtype':'int8','layout':'NCHW'})
            save_json(d/'response.json',response);out=read_board_response(d,req,.1);self.assertLess(out[0,0,0,0],0)
            for key,value in [('model_id','acdc_coronal_v1'),('input_sha256','b'*64),('request_id','old-job'),('synthetic',True),('head_scale',.2)]:
                bad={**response,key:value};save_json(d/'response.json',bad)
                with self.assertRaises(ValueError):read_board_response(d,req,.1)
            response['output']['path']='../head.bin';save_json(d/'response.json',response)
            with self.assertRaises(ValueError):read_board_response(d,req,.1)

if __name__=='__main__':unittest.main()
