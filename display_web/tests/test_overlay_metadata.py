"""Metadata caching must not reuse a different bit/HWH or stale download flag."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace, ModuleType
import tempfile
import unittest
from unittest.mock import patch, Mock

spec = importlib.util.spec_from_file_location('overlay_metadata', Path(__file__).parents[1]/'overlay_metadata.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class MetadataCacheTests(unittest.TestCase):
    def test_metadata_load_restores_gc_policy_even_when_pickle_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'snapshot';path.write_bytes(b'invalid pickle')
            initially=module.gc.isenabled()
            try:
                for enabled in (True,False):
                    module.gc.enable() if enabled else module.gc.disable()
                    with self.assertRaises(Exception):module._load_snapshot(path)
                    self.assertEqual(module.gc.isenabled(),enabled)
            finally:
                module.gc.enable() if initially else module.gc.disable()

    def test_download_publishes_new_state_without_serializing_graph_again(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);bit=folder/'base.bit';bit.write_bytes(b'bit');bit.with_suffix('.hwh').write_bytes(b'hwh')
            sdk=folder/'sdk.py';sdk.write_bytes(b'installed SDK')
            parser=SimpleNamespace(ip_dict={'shutdown':dict(type='xilinx.com:ip:dfx_axi_shutdown_manager:1.0',phys_addr=123)},
                mem_dict={'PSDDR':{'size':512}},systemgraph={'pure_metadata':True},tag='matched graph')
            fallback=Mock();saved=[]
            class State:
                def __init__(self,**kw):self.values=kw;self.shutdown=[]
                def add(self,**kw):self.shutdown.append(kw)
            vendor=SimpleNamespace(__file__=str(sdk),GlobalState=State,
                bitstream_hash=lambda _: 'actual-bit-hash',save_global_state=saved.append)
            device=SimpleNamespace(name='embedded_xrt0',get_bitfile_metadata=lambda *a,**kw:parser,gen_cache=fallback)
            modules={'pynq':SimpleNamespace(__version__='3.1.1'),
                'pynq.pl_server':SimpleNamespace(embedded_device=vendor),
                'pynq.pl_server.device':SimpleNamespace(Device=SimpleNamespace(active_device=device))}
            with patch.dict('sys.modules',modules):
                module.enable_metadata_cache(folder/'cache')
                metadata=device.get_bitfile_metadata(str(bit))
                with patch.object(module.pickle,'dump',side_effect=AssertionError('repeated serialization')):
                    device.gen_cache(SimpleNamespace(bitfile_name=str(bit)),metadata)
            fallback.assert_not_called()
            self.assertEqual(saved[0].values['bitfile_hash'],'actual-bit-hash')
            self.assertEqual(saved[0].shutdown,[dict(name='shutdown',addr=123)])
            with (folder/'_current_metadata.pkl').open('rb') as stream:stored=module.pickle.load(stream)
            self.assertEqual(stored.tag,metadata.tag)

    def test_cache_hit_changed_hwh_and_partial_passthrough(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder); bit = folder/'base.bit'; bit.write_bytes(b'bit')
            hwh = bit.with_suffix('.hwh'); hwh.write_bytes(b'hwh')
            sdk = folder/'sdk.py'; sdk.write_bytes(b'installed SDK')
            calls = []
            def original(path, partial=False):
                calls.append((path, partial));return SimpleNamespace(_from_cache=True, tag=hwh.read_text())
            device = SimpleNamespace(get_bitfile_metadata=original)
            modules = {
                'pynq': SimpleNamespace(__version__='3.1.1'),
                'pynq.pl_server': SimpleNamespace(embedded_device=SimpleNamespace(__file__=str(sdk))),
                'pynq.pl_server.device': SimpleNamespace(Device=SimpleNamespace(active_device=device)),
            }
            with patch.dict('sys.modules', modules):
                module.enable_metadata_cache(folder/'cache')
                device.get_bitfile_metadata(str(bit))
                cached = device.get_bitfile_metadata(str(bit))
                self.assertEqual(len(calls), 1)
                self.assertFalse(hasattr(cached, '_from_cache'))
                hwh.write_bytes(b'changed')
                self.assertEqual(device.get_bitfile_metadata(str(bit)).tag, 'changed')
                device.get_bitfile_metadata(str(bit), partial=True)
                self.assertEqual(len(calls), 3)


if __name__ == '__main__': unittest.main()
