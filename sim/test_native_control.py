"""Ownership and cache ordering at the optional native control boundary."""
import ctypes as ct
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'board'))
from native_control import NativeControl
from simple_dma import SimpleDMA


class Tests(unittest.TestCase):
    def setup_backend(self,code=0,receive=False):
        events=[]
        parameters={'c_include_sg':0,'c_micro_dma':0,'c_sg_length_width':26,'c_addr_width':32,
                    'c_m_axi_mm2s_data_width':64,'c_m_axi_s2mm_data_width':64}
        dma=SimpleDMA(SimpleNamespace(array=np.zeros(4096//4,dtype=np.uint32)),parameters)
        backend=object.__new__(NativeControl); backend.result=(ct.c_uint32*8)()
        core=SimpleNamespace(array=np.zeros(4096//4,dtype=np.uint32))
        backend.bind(core,dma)
        buffer=SimpleNamespace(physical_address=0x10000000,nbytes=128,
                               flush=lambda:events.append('flush'),invalidate=lambda:events.append('invalidate'))
        device=SimpleNamespace(config={},timeout=.05,frame=23,job=0xffffffff)
        def packet(*args):
            events.append('packet')
            self.assertTrue(dma.active)
            self.assertIs(dma._pending[1 if receive else 0],buffer)
            self.assertEqual(args[5:9],(23,1,0x10000000,64))
            backend.result[0]=456; backend.result[1]=123; backend.result[4]=100000
            return code
        backend.packet_fn=packet
        return backend,dma,buffer,device,events

    def test_success_cache_order_and_release(self):
        for receive in (False,True):
            backend,dma,buffer,device,events=self.setup_backend(receive=receive)
            counters,elapsed=backend.execute(device,4 if receive else 2,buffer,64)
            self.assertEqual(counters,{'core_cycles':456,'compute_cycles':123})
            self.assertGreaterEqual(elapsed,.0001)
            self.assertEqual(events,['flush','packet']+(['invalidate'] if receive else []))
            self.assertFalse(dma.active); self.assertIsNone(dma._pending)

    def test_error_retains_owner_and_skips_invalidation(self):
        for code in range(1,12):
            backend,dma,buffer,device,events=self.setup_backend(code,receive=True)
            with self.assertRaisesRegex(RuntimeError,f'code={code} '):
                backend.execute(device,4,buffer,64)
            self.assertTrue(dma.active); self.assertIs(dma._pending[1],buffer)
            self.assertEqual(events,['flush','packet'])
            with self.assertRaisesRegex(RuntimeError,'still active'):
                backend.execute(device,4,buffer,64)

    def test_preflight_never_starts_or_flushes(self):
        for mode in ('length','address','frame','timeout','configuration'):
            backend,dma,buffer,device,events=self.setup_backend()
            if mode=='address': buffer.physical_address+=1
            if mode=='frame': device.frame=-1
            if mode=='timeout': device.timeout=float('nan')
            if mode=='configuration': device.config=None
            with self.assertRaises(ValueError): backend.execute(device,2,buffer,129 if mode=='length' else 64)
            self.assertFalse(dma.active); self.assertEqual(events,[])

    def test_mapping_contract(self):
        backend,dma,_,_,_=self.setup_backend()
        for array in (np.zeros(128,dtype=np.uint8),np.zeros(2,dtype=np.uint32),
                      np.zeros(256,dtype=np.uint32)[::2]):
            with self.assertRaises(ValueError): backend.bind(SimpleNamespace(array=array),dma)


if __name__=='__main__': unittest.main()
