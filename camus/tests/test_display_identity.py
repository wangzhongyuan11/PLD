"""Result annotation identity must not bypass the existing PL validation."""
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from camus.pipeline import CamusPipeline


class DisplayIdentityTests(unittest.TestCase):
    def test_bit_identity_is_recorded_with_original_hardware_constructor(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'overlay').mkdir();(root/'overlay/dma_top.bit').write_bytes(b'fixture bit')
            device=SimpleNamespace(io=SimpleNamespace(native_control=None))
            with patch('camus.pipeline.load_model',return_value=SimpleNamespace(root=root)),patch('camus.pipeline.CamusGraph'),patch('camus.accelerator.CamusTileAccelerator',return_value=device) as driver:
                pipeline=CamusPipeline(root,backend='pl',runtime=root)
            self.assertEqual(pipeline.runtime_sha256,hashlib.sha256(b'fixture bit').hexdigest())
            driver.assert_called_once_with(root,root,3.)

    def test_missing_identity_input_does_not_allocate_hardware(self):
        with tempfile.TemporaryDirectory() as directory,patch('camus.pipeline.load_model',return_value=SimpleNamespace(root=directory)),patch('camus.accelerator.CamusTileAccelerator') as driver:
            with self.assertRaises(OSError):CamusPipeline(directory,backend='pl',runtime=directory)
            driver.assert_not_called()


if __name__=='__main__':unittest.main()
