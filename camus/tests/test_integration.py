import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import zipfile
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image

from camus.accelerator import CamusMMIO, validate_overlay_metadata
from camus.contract import shape
from camus.integer import IntegerOperatorError, requantize_int8, quantize_letterbox_input
from camus.model import DEFAULT_MODEL_ROOT, ModelError, load_model, load_golden
from camus.pipeline import CamusPipeline, protect_output
from camus.web import decode_upload
from camus.preprocess import load_gray_frame, PreprocessError


class ContractRegressionTests(unittest.TestCase):
    def test_weight_bank_guard_and_preflight_before_mmio(self):
        io = CamusMMIO.__new__(CamusMMIO); io.core = Mock(); io.closed = False; io.frame = 0
        with self.assertRaises(ValueError): io.configure(0,10,10,191,33,3)
        io.core.read.assert_not_called(); io.core.write.assert_not_called()
        self.assertEqual(shape(0,10,10,128,56,3)['parameter_bytes'], 56*128*9+56*8)
        io.core.read.return_value = 0
        io.configure(0,10,10,192,32,3,relu=True)
        self.assertIn(unittest.mock.call(0x5c,1),io.core.write.call_args_list)
        io.core.reset_mock(); io.core.read.return_value = 0
        io.configure(0,8,8,16,4,1,relu=False)
        self.assertIn(unittest.mock.call(0x5c,0),io.core.write.call_args_list)

    def test_old_overlay_and_modes_rejected(self):
        with self.assertRaises(ValueError): validate_overlay_metadata({'type':'user:pld:axis_tensor_accel:1.0'}, {})
        for mode in (5,6):
            with self.assertRaises(ValueError): shape(mode,8,8,16,8,3)
        with self.assertRaises(ValueError): shape(1,8,8,192,relu=True)

    def test_quantization_rejects_invalid_values(self):
        for array in (np.zeros((1,1,1,1),np.float32),np.full((1,1,1,1),1<<40,np.int64),np.full((1,1,1,1),-(1<<63),np.int64)):
            with self.assertRaises(IntegerOperatorError): requantize_int8(array,np.ones(1,np.int32))
        for value in (float('nan'),float('inf'),2.,-.1):
            with self.assertRaises(IntegerOperatorError): quantize_letterbox_input(np.array([[value]],np.float32))

    def test_output_cannot_modify_model_delivery(self):
        with self.assertRaises(ValueError): protect_output(DEFAULT_MODEL_ROOT/'new_output', DEFAULT_MODEL_ROOT)

    def test_upload_paths_and_bad_payload_rejected(self):
        for payload in (None,{}, {'files':[None]}, {'files':[{'name':'../x','data_base64':'AA=='}],'input_name':'../x'}):
            with self.assertRaises(ValueError): decode_upload(payload)
        value=base64.b64encode(b'raw').decode()
        files,selected=decode_upload(dict(files=[dict(name='frame.mhd',data_base64=value),dict(name='frame.raw',data_base64=value)],input_name='frame.mhd'))
        self.assertEqual(selected,'frame.mhd'); self.assertEqual(len(files),2)

    def test_repeated_model_load_rechecks_tamper(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'int8_export';shutil.copytree(DEFAULT_MODEL_ROOT,root)
            model=load_model(root)
            self.assertFalse(model.layers[0].weight.flags.writeable)
            weight=root/model.layers[0].directory/'weight.bin';data=bytearray(weight.read_bytes());data[0]^=1;weight.write_bytes(data)
            with self.assertRaises(ModelError): load_model(root)

    def test_nifti_gzip_xy_geometry_without_runtime_dependency(self):
        from unittest.mock import patch
        data=np.arange(6,dtype=np.uint8).reshape(3,2,1)
        header=SimpleNamespace(get_data_shape=lambda:data.shape,get_zooms=lambda:(.3,.2,1.))
        nib=SimpleNamespace(load=lambda path:SimpleNamespace(header=header,dataobj=data))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,{'nibabel':nib}):
            p=Path(tmp)/'frame.nii.gz';p.write_bytes(b'fake loader fixture')
            frame=load_gray_frame(p)
            self.assertEqual(frame.data.tolist(),[[0,2,4],[1,3,5]])
            self.assertEqual(frame.spacing_yx_mm,(.2,.3))
            header.get_data_shape=lambda:(10000,10000,1)
            with self.assertRaises(PreprocessError):load_gray_frame(p)


class PipelineIntegrationTests(unittest.TestCase):
    def test_new_gray_input_to_golden_head_and_complete_zip(self):
        model=load_model();expected=load_golden(model,model.layers[0],'input_full')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); image=root/'frame.png'
            gray=np.rint(expected[0,0].astype(np.float64)*255/127).astype(np.uint8)
            Image.fromarray(gray).save(image)
            with CamusPipeline(model.root,'emulator') as pipeline:
                first=pipeline.run(image,root/'result',spacing=(.2,.3))
                events=first['report']['command_events']
                self.assertEqual(len(events),605)
                self.assertEqual(sum(e['bytes'] for e in events),33273648)
                self.assertEqual((root/'result/head.bin').read_bytes(),load_golden(model,model.layers[-1],'output_full_golden').tobytes())
                with zipfile.ZipFile(root/'result/result.zip') as archive:
                    self.assertEqual(len(archive.namelist()),len(set(archive.namelist())))
                    self.assertEqual(archive.read('report.json'),(root/'result/report.json').read_bytes())
                    self.assertIn('labels_native.npz',archive.namelist())
                self.assertNotIn('labels',first['report']['output']['postprocess'])
                self.assertEqual(sum(v['pixels'] for v in first['report']['output']['areas'].values()),256*256)
                self.assertFalse(first['report']['session_reused'])
                self.assertAlmostEqual(sum(v['area_mm2'] for v in first['report']['output']['areas'].values()),256*256*.2*.3)
            self.assertEqual(json.loads((root/'result/head.json').read_text())['shape'],[1,4,256,256])


if __name__=='__main__': unittest.main()
