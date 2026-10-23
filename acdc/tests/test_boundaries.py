import json
import tempfile
import unittest
from pathlib import Path
import nibabel as nib
import numpy as np

from acdc.contract import ContractError,atomic_json,file_sha
from acdc.hardware import validate_overlay,VLNV
from acdc.reconstruct import Collector
from acdc.volume import load_volume,AxialInputs
from acdc.backend import DemoBackend
from acdc.tests.test_integration import request
from acdc.evaluate import evaluate


class BoundariesTest(unittest.TestCase):
    def test_old_or_wrong_hardware_metadata_is_rejected(self):
        info=dict(type=VLNV,phys_addr=0x43c00000,addr_range=4096)
        dma=dict(phys_addr=0x40400000,parameters={k:'64' for k in (
            'c_m_axi_mm2s_data_width','c_m_axi_s2mm_data_width','c_m_axis_mm2s_tdata_width')})
        validate_overlay(info,dma)
        info['type']='user:pld:axis_tensor_accel_camus:1.0'
        with self.assertRaises(ValueError):validate_overlay(info,dma)
        info['type']=VLNV;dma['parameters']['c_include_sg']='1'
        with self.assertRaises(ValueError):validate_overlay(info,dma)

    def test_collector_rejects_duplicate_missing_and_wrong_phase(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'v.nii.gz';a=np.diag([2.,2.,2.,1.])
            image=nib.Nifti1Image(np.arange(27,dtype=np.float32).reshape(3,3,3),a);image.header.set_xyzt_units('mm');nib.save(image,p)
            v=load_volume(p);b=DemoBackend();inputs=AxialInputs(v);collector=Collector(v,b.model_sha,b.head_scale,'ED')
            x,ids=inputs.get(0);r=request(b);r.update(geometry_sha256=v.geometry_sha,total_slices=3)
            head,meta=b.infer(x,r);collector.add(head,meta,inputs.spec)
            with self.assertRaises(ContractError):collector.add(head,meta,inputs.spec)
            with self.assertRaises(ContractError):collector.finish()
            with self.assertRaises(ContractError):collector.add(head,dict(meta,phase='ES',center_index=1),inputs.spec)

    def test_patient_metrics_preserve_pair_and_hash_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);result=root/'result';(result/'labels').mkdir(parents=True);(result/'geometry').mkdir()
            rows=[];files={}
            for phase in ('ED','ES'):
                label=np.zeros((4,4,4),np.uint8);label[:2,:2,:2]=3
                image=nib.Nifti1Image(label,np.diag([2.,2.,2.,1.]));image.header.set_xyzt_units('mm')
                gt=root/(phase+'_gt.nii.gz');pred=result/'labels'/(phase+'.nii.gz');nib.save(image,gt);nib.save(image,pred)
                files[f'labels/{phase}.nii.gz']=file_sha(pred)
                atomic_json(result/'geometry'/(phase+'.json'),dict(geometry_policy='strict'))
                files[f'geometry/{phase}.json']=file_sha(result/'geometry'/(phase+'.json'))
                rows.append(dict(patient='synthetic001',phase=phase,result_dir='result',gt=gt.name,gt_sha256=file_sha(gt)))
            atomic_json(result/'manifest.json',dict(model_sha256='a'*64,execution_profile='axial_only_v1',files=files,synthetic=True))
            cases=root/'cases.json';atomic_json(cases,dict(schema='pld-acdc-evaluation-cases-v1',cases=rows,cohort='synthetic_test'))
            report=evaluate(cases,'a'*64)
            self.assertEqual(report['macro_dice'],1);self.assertEqual(report['patient_count'],1)
            geom=result/'geometry/ED.json';original=geom.read_bytes();geom.write_bytes(original+b' ')
            with self.assertRaises(ContractError):evaluate(cases,'a'*64)
            geom.write_bytes(original)
            rows.pop();atomic_json(cases,dict(schema='pld-acdc-evaluation-cases-v1',cases=rows))
            with self.assertRaises(ContractError):evaluate(cases,'a'*64)
