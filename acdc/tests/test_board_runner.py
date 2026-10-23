from types import SimpleNamespace as S
import unittest
import numpy as np

from acdc.board_verify import recovery_checks
from acdc.contract import ContractError


class Fault(Exception):
    def __init__(self,message,error):
        super().__init__(message);self.recovered=True;self.state={'core':{'error':error}}


class FakeIO:
    def __init__(self):
        self.timeout=5;self.tx=np.zeros(16,np.uint8);self.rx=np.zeros(65536,np.uint8)
        self.reset=False;self.dma=S(active=False,submit=self.submit);self.core=S(read=self.read)
    def configure(self,*args,**kwargs):return {'output_bytes':50400}
    def _guard(self,fn):fn()
    def _execute(self,*args):raise Fault('short packet',2)
    def _begin(self,*args):pass
    def _wait(self,*args):raise Fault('core completion timeout',0)
    def load_parameters(self,*args):pass
    def load_input(self,*args):pass
    def submit(self,**kwargs):self.dma.active=True
    def read(self,address):return 64 if address==0x58 else 0
    def snapshot(self):return {'core':{'status':1,'sent':0}}
    def recover(self):self.dma.active=False;self.reset=True


class BoardRunnerTest(unittest.TestCase):
    def test_recovery_requires_three_fresh_complete_heads(self):
        io=FakeIO();calls=[];head=np.zeros((1,4,2,2),np.int8)
        def forward(first):calls.append(first);return {'head':head}
        layer=S(weight=np.zeros((128,128,3,3),np.int8),bias=np.zeros(128,np.int32),multipliers=np.ones(128,np.int32))
        backend=S(graph=S(device=S(io=io),forward=forward),model=S(layers=[layer]*15))
        rows=[];recovery_checks(backend,head,head,rows,fault_type=Fault)
        self.assertEqual(len(calls),3);self.assertEqual(len(rows),3)
        self.assertEqual(io.timeout,5);self.assertTrue(io.reset)
        with self.assertRaises(ContractError):recovery_checks(backend,head,head+1,[],fault_type=Fault)
