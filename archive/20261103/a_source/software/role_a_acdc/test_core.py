import unittest,tempfile,json
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
from .model import ACDCUNet,LAYER_NAMES
from .geometry import stack,plane,letterbox,undo_letterbox,put_plane,prepare_volume,probabilities_to_native,physical_affine,ACDC_GEOMETRY_POLICY
from .contract import *
from .train import get_rng,set_rng,seed_all,torch_save,loss_fn

class ContractTests(unittest.TestCase):
    def test_topology_and_cat_order(self):
        torch.set_num_threads(2);m=ACDCUNet();self.assertEqual(len(m.contract()),15)
        self.assertEqual(sum(p.numel() for p in m.parameters()),487348)
        self.assertEqual(m(torch.zeros(1,3,32,32)).shape,(1,4,32,32))
        x=torch.zeros(1,3,32,32)
        with torch.no_grad():
            for p in m.parameters():p.zero_()
            m.head.bias.copy_(torch.tensor([-2.,-1.,1.,2.]))
            y=m(x)
        self.assertTrue(torch.equal(y[0,:,0,0],torch.tensor([-2.,-1.,1.,2.])))

    def test_input_boundary_and_layout(self):
        x=np.zeros(INPUT_SHAPE);x[:,0]=-1;x[:,2]=1
        q=quantize_input(x);self.assertEqual(q.nbytes,196608)
        self.assertEqual((q[0,:,0,0]).tolist(),[-127,0,127])
        with self.assertRaises(ValueError):quantize_input(x+1)

    def test_signed_ties_and_saturation(self):
        q=round_q24(np.array([-5,-3,-1,1,3,5]),1<<23)
        np.testing.assert_array_equal(q,[-2,-2,0,0,2,2])
        s=np.full((1,4,4),127,np.int8);d=np.full((1,2,2),-127,np.int8)
        out=cat_mode4(s,d,2<<24,2<<24);self.assertEqual(out.min(),-127);self.assertEqual(out.max(),127)

    def test_cat_spatial_and_channel_order(self):
        skip=np.full((2,4,4),9,np.int8);deep=np.arange(4,dtype=np.int8).reshape(1,2,2)
        out=cat_mode4(skip,deep,1<<24,2<<24)
        np.testing.assert_array_equal(out[:2],skip)
        np.testing.assert_array_equal(out[2],[[0,0,2,2],[0,0,2,2],[4,4,6,6],[4,4,6,6]])

    def test_channel_bank_capacity(self):
        self.assertEqual(conv_channel_chunks(64,128,3),[(0,64),(64,64)])
        self.assertEqual(conv_channel_chunks(128,128,3),[(0,56),(56,56),(112,16)])
        self.assertEqual(conv_channel_chunks(192,64,3),[(0,32),(32,32)])

    def test_parameter_endianness(self):
        b=parameter_blob(np.array([[[[3]]]],np.int8),np.array([0x10203]),np.array([1<<24]))
        self.assertEqual(b,b'\x03\x03\x02\x01\x00\x00\x00\x00\x01')

class GeometryTests(unittest.TestCase):
    def test_acdc_direction_only_sform_is_explicit(self):
        x=np.ones((8,8,8),np.float32);aff=np.diag([-1.,-1.,1.,1.]);image=nib.Nifti1Image(x,aff)
        image.header.set_zooms((1.5625,1.5625,10));image.set_qform(image.get_qform(),0);image.set_sform(aff,2)
        with self.assertRaises(ValueError):physical_affine(image)
        corrected,why=physical_affine(image,ACDC_GEOMETRY_POLICY)
        np.testing.assert_allclose(nib.affines.voxel_sizes(corrected),[1.5625,1.5625,10])
        self.assertAlmostEqual(abs(np.linalg.det(corrected[:3,:3])),24.4140625)
    def test_neighbors_and_axes(self):
        x=np.arange(4*5*6).reshape(4,5,6).astype(np.float32)/120
        for direction,axis in [('axial',2),('coronal',1),('sagittal',0)]:
            out,t=stack(x,direction,0,size=32)
            np.testing.assert_array_equal(out[0],out[1])
            self.assertFalse(np.array_equal(out[1],out[2]))
            a=np.zeros_like(x);put_plane(a,axis,1,plane(x,axis,1))
            np.testing.assert_array_equal(plane(a,axis,1),plane(x,axis,1))

    def test_letterbox_inverse_probability(self):
        a=np.ones((4,15,31),np.float32)*.25
        boxed,t=letterbox(a);restored=undo_letterbox(boxed,t)
        np.testing.assert_allclose(restored,a,atol=1e-6)

    def test_oblique_physical_landmark(self):
        a=np.array([[0,-2,0,20],[2,0,0,-15],[0,0,5,3],[0,0,0,1]],float)
        x=np.arange(12*13*7,dtype=np.float32).reshape(12,13,7)+1
        y=np.zeros(x.shape,np.uint8);y[4:8,4:8,2:5]=3
        work,labels,m=prepare_volume(nib.Nifti1Image(x,a),nib.Nifti1Image(y,a))
        self.assertEqual(m['working_axes'],'RAS');self.assertEqual(work.shape,labels.shape)
        prob=np.stack([labels==c for c in range(4)]).astype(np.float32)
        native=probabilities_to_native(prob,m).argmax(0)
        self.assertTrue(np.array_equal(native,y))
        self.assertAlmostEqual(m['native_voxel_volume_mm3'],20)

    def test_reject_label_geometry_mismatch(self):
        x=np.arange(512,dtype=np.float32).reshape(8,8,8)+1
        with self.assertRaises(ValueError):prepare_volume(nib.Nifti1Image(x,np.eye(4)),nib.Nifti1Image(np.zeros_like(x),np.diag([2,1,1,1])))

class RecoveryTests(unittest.TestCase):
    def test_optimizer_rng_exact_cpu_resume(self):
        torch.set_num_threads(2);seed_all(42)
        model=torch.nn.Conv2d(3,4,1);opt=torch.optim.AdamW(model.parameters(),lr=.01)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,10)
        def step(m,o,s):
            x=torch.rand(1,3,8,8);y=torch.randint(0,4,(1,8,8));o.zero_grad();loss_fn(m(x),y).backward();o.step();s.step()
        step(model,opt,sched)
        with tempfile.TemporaryDirectory() as t:
            file=Path(t)/'state.pt';torch_save(file,{'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'rng':get_rng()})
            step(model,opt,sched);expected={k:v.clone() for k,v in model.state_dict().items()}
            state=torch.load(file,weights_only=False);m2=torch.nn.Conv2d(3,4,1);o2=torch.optim.AdamW(m2.parameters(),lr=.01)
            s2=torch.optim.lr_scheduler.CosineAnnealingLR(o2,10);m2.load_state_dict(state['model']);o2.load_state_dict(state['optimizer']);s2.load_state_dict(state['scheduler']);set_rng(state['rng'])
            step(m2,o2,s2)
            for k,v in m2.state_dict().items():self.assertTrue(torch.equal(v,expected[k]))

if __name__=='__main__':unittest.main()
