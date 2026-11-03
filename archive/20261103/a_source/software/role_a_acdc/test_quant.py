import unittest
import numpy as np
import torch
from .model import ACDCUNet,LAYER_NAMES
from .quantize import IntegerModel,requant_torch
from .contract import round_q24
from software.role_a_camus_int8 import integer as kernel

class QuantTests(unittest.TestCase):
    def test_signed_q24_torch_matches_numpy(self):
        values=np.array([-2147483647,-1000,-5,-3,-1,0,1,3,5,1000,2147483647],np.int64)
        for m in (1,1<<23,1<<24,2**31-1):
            expected=np.clip(round_q24(values,m),-127,127).astype(np.int8)
            actual=requant_torch(torch.from_numpy(values),torch.tensor(m)).numpy()
            np.testing.assert_array_equal(actual,expected)

    def test_integer_network_zero_and_signed_head(self):
        torch.set_num_threads(2);m=ACDCUNet()
        convs=dict(m.named_modules());layers=[]
        for i,n in enumerate(LAYER_NAMES):
            conv=convs[n];w=np.zeros(tuple(conv.weight.shape),np.int8)
            b=np.array([-40,-20,20,40],np.int32) if i==14 else np.zeros(conv.out_channels,np.int32)
            mult=np.full(conv.out_channels,1<<24,np.int32);bound=kernel.require_bounds(w,b)
            layers.append(kernel.QuantLayer(n,w,b,mult,1/127 if i==0 else .01,.01,conv.padding[0],conv.kernel_size[0],bound,True))
        cats=[{'skip_multiplier_q24':1<<24,'deep_multiplier_q24':1<<24} for _ in range(3)]
        engine=IntegerModel(layers,cats)
        head=engine.quantized(torch.zeros(1,3,8,8)).numpy()
        self.assertEqual(head.shape,(1,4,8,8));self.assertLess(head[0,0,0,0],0);self.assertGreater(head[0,3,0,0],0)
        for i,c in enumerate([-40,-20,20,40]):np.testing.assert_array_equal(head[0,i],np.full((8,8),c,np.int8))

    def test_independent_oracle_first_middle_head(self):
        rng=np.random.default_rng(77);torch.manual_seed(77);m=ACDCUNet();convs=dict(m.named_modules())
        for i in (0,7,8,14):
            n=LAYER_NAMES[i];layer=kernel.build_layer(convs[n],n,input_scale=.02,output_scale=.03)
            q=rng.integers(-127,128,(1,layer.q_weight.shape[1],8,8),dtype=np.int8)
            report=kernel.verify_against_float64_graph(layer,kernel.Int8Feature(q,.02),relu=i<14)
            self.assertTrue(report['matches_oracle']);self.assertEqual(report['mismatch_count'],0)

if __name__=='__main__':unittest.main()
