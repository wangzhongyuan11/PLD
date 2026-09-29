"""Exercise the DMA copy boundary with non-contiguous signed branch tensors."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'board'))
from tensor_accel import TileAccelerator


class Tests(unittest.TestCase):
    def test_rejected_native_second_branch_never_touches_hardware(self):
        device=object.__new__(TileAccelerator)
        device.tx=np.full(64,0xa5,dtype=np.uint8); device._tx_view=device.tx
        device.config={'mode':4,'split':1,'ci':2,'h':2,'w':2,'input_bytes':5}
        calls=[]
        def copy(target,source):
            calls.append(source.copy())
            np.copyto(target,source)
            if np.any(source==-128): raise ValueError('input contains -128')
        device.native_copy=SimpleNamespace(copy=copy)
        device._guard=lambda operation:self.fail('preflight entered hardware guard')
        with self.assertRaisesRegex(ValueError,'-128'):
            device.load_input(np.ones((1,1,2,2),dtype=np.int8),np.full((1,1,1,1),-128,dtype=np.int8))
        self.assertEqual(len(calls),2)
        self.assertTrue(np.all(device.tx[5:]==0xa5))

    def test_strided_and_reversed_branches_preserve_nchw_bytes(self):
        first=np.arange(96,dtype=np.int8).reshape(1,3,4,8)[:,:,1:3,1:7:2]
        second=np.arange(-24,0,dtype=np.int8).reshape(1,2,3,4)[:,:,::-1,::2]
        expected=first.tobytes()+second.tobytes()
        device=object.__new__(TileAccelerator)
        device.tx=np.full(128,0xa5,dtype=np.uint8); device._tx_view=device.tx
        device.config={'mode':4}; device.events=[]
        seen=[]
        device.dma=SimpleNamespace(send=lambda buffer,n:seen.append(buffer[:n].tobytes()))
        device.core=SimpleNamespace(read=lambda addr:len(expected))
        device._begin=lambda command:None
        device._wait=lambda received,sent:{'core_cycles':0,'compute_cycles':0}
        device._guard=lambda operation:operation()
        device._load_parts(2,(first,second),len(expected))
        self.assertEqual(seen,[expected])
        self.assertTrue(np.all(device.tx[len(expected):]==0xa5))
        self.assertEqual(device.events[0]['bytes'],len(expected))
        with self.assertRaisesRegex(ValueError,'payload length'):
            device._load_parts(2,(first,second),len(expected)-1)
        self.assertEqual(len(seen),1)


if __name__=='__main__': unittest.main()
