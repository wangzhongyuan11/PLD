"""Display metadata must leave frozen inference and exported artifacts intact."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from display_web.camus_display_identity import read_runtime_identity,write_runtime_identity
from display_web.telemetry_hud import load_telemetry


class CamusDisplayIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.runtime=self.root/'runtime';self.folder=self.root/'result'
        (self.runtime/'overlay').mkdir(parents=True);self.folder.mkdir()
        self.bit=self.runtime/'overlay/dma_top.bit';self.bit.write_bytes(b'validated runtime fixture')
        self.model='a'*64;self.bit_sha=hashlib.sha256(self.bit.read_bytes()).hexdigest()
        self.manifest=dict(schema='camus-runtime-v1',model_spec_sha256=self.model,
                           files={'overlay/dma_top.bit':self.bit_sha})
        self.write_manifest()
        self.report=dict(schema='pld-camus-result-v1',backend='pl',model=dict(spec_sha256=self.model,head=dict(dtype='int8')),
                         timings_seconds=dict(network=.2),seconds_to_metadata_finalize=1)
        (self.folder/'report.json').write_text(json.dumps(self.report),encoding='utf-8')
        (self.folder/'result.zip').write_bytes(b'frozen result ZIP must not be opened or amended')
        self.pipeline=SimpleNamespace(backend='pl',device=SimpleNamespace(io=SimpleNamespace(
            overlay=SimpleNamespace(bitfile_name=str(self.bit)))))

    def write_manifest(self):
        (self.runtime/'runtime_manifest.json').write_text(json.dumps(self.manifest),encoding='utf-8')

    def test_sidecar_preserves_report_zip_and_never_reopens_bit(self):
        before={p.name:p.read_bytes() for p in self.folder.iterdir()}
        original_open=Path.open
        def guarded(path,*args,**kwargs):
            if path==self.bit:raise AssertionError('display adapter reread the bitstream')
            return original_open(path,*args,**kwargs)
        with patch.object(Path,'open',guarded):
            self.assertEqual(write_runtime_identity(self.pipeline,self.folder),self.bit_sha)
        for name,data in before.items():self.assertEqual((self.folder/name).read_bytes(),data)
        self.assertEqual(read_runtime_identity(self.folder,self.model),self.bit_sha)
        self.assertEqual(load_telemetry(self.folder,'camus')['runtime_sha256'],self.bit_sha)

    def test_edited_report_or_other_model_invalidates_sidecar(self):
        write_runtime_identity(self.pipeline,self.folder)
        self.assertIsNone(read_runtime_identity(self.folder,'b'*64))
        with (self.folder/'report.json').open('a',encoding='utf-8') as stream:stream.write(' ')
        self.assertIsNone(read_runtime_identity(self.folder,self.model))

    def test_software_unknown_driver_and_mismatched_manifest_get_no_identity(self):
        self.pipeline.backend='reference'
        self.assertIsNone(write_runtime_identity(self.pipeline,self.folder))
        self.pipeline.backend='pl';self.pipeline.device=object()
        self.assertIsNone(write_runtime_identity(self.pipeline,self.folder))
        self.pipeline.device=SimpleNamespace(io=SimpleNamespace(overlay=SimpleNamespace(bitfile_name=str(self.bit))))
        self.manifest['model_spec_sha256']='b'*64;self.write_manifest()
        self.assertIsNone(write_runtime_identity(self.pipeline,self.folder))
        self.assertFalse((self.folder/'hud_runtime.json').exists())

    def test_optional_identity_io_failure_cannot_fail_completed_inference(self):
        with patch.object(Path,'replace',side_effect=OSError('fixture full disk')):
            self.assertIsNone(write_runtime_identity(self.pipeline,self.folder))
        self.assertFalse((self.folder/'hud_runtime.json').exists())
        self.assertFalse((self.folder/'hud_runtime.json.tmp').exists())


if __name__=='__main__':unittest.main()
