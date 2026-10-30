"""Compare the portable image kernels to the independent frozen libraries."""
import numpy as np
import pytest

from acdc.appliance.native import load_native
from acdc.contract import ContractError


@pytest.fixture(scope='module')
def kernel():return load_native()


def test_four_channel_affine_retains_each_channel_order(kernel):
    from scipy.ndimage import affine_transform
    rng=np.random.default_rng(2381)
    source=rng.random((4,19,17,13),dtype=np.float32)
    sampler=kernel.prepare_affine4(source)
    for matrix,offset in ((np.diag([.7,1.3,.9]),[-.3,.5,.2]),
                          (np.array([[.7,.1,0],[-.1,.8,.1],[.1,0,.9]]),[.2,.1,-.2])):
        actual=sampler.apply((15,14,12),matrix,offset)
        expected=np.stack([affine_transform(c,matrix,offset,output_shape=(15,14,12),order=1,mode='constant',cval=0) for c in source])
        np.testing.assert_array_equal(actual,expected)


def test_foreground_bounds_and_empty_volume(kernel):
    labels=np.zeros((17,21,9),np.uint8)
    assert kernel.foreground_bounds(labels) is None
    labels[2:8,4:19,1:6]=2;labels[12,3,7]=3
    points=np.argwhere(labels!=0)
    np.testing.assert_array_equal(kernel.foreground_bounds(labels),[points.min(axis=0),points.max(axis=0)])
    labels[0,0,0]=4
    with pytest.raises(ContractError):kernel.foreground_bounds(labels)


def test_normalize_and_quantization(kernel):
    from acdc.volume import normalize
    rng=np.random.default_rng(20261021)
    for n in (8,9,101,102,1001):
        a=rng.integers(-300,600,size=(1,1,n)).astype(np.float32)
        a.ravel()[0]=0
        if np.count_nonzero(a)<8:continue
        expected,meta=normalize(a);actual,got=kernel.normalize(a)
        assert got==meta
        np.testing.assert_array_equal(actual,expected)
    a=np.arange(-255,256,dtype=np.float32)/np.float32(254)
    expected=np.clip(np.rint(a.astype(np.float64)*127),-127,127).astype(np.int8)
    np.testing.assert_array_equal(kernel.quantize(a),expected)
    for bad in (np.zeros((2,2,2)),np.ones((2,2,2)),np.full((2,2,2),np.nan)):
        with pytest.raises(ContractError):kernel.normalize(bad)


@pytest.mark.parametrize('shape,target',[
    ((1,17,23),(256,207)),((4,211,171),(125,256)),
    ((4,7,11),(7,11)),((1,1,9),(3,7)),((4,256,207),(211,171)),
    ((1,9,1),(7,3)),((4,1,1),(5,3)),
])
def test_resize_matches_torch(kernel,shape,target):
    from acdc.volume import resize_torch
    a=np.random.default_rng(120).normal(size=shape).astype(np.float32)
    expected=resize_torch(a[None],target)[0]
    actual=kernel.resize(a,target)
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('kind',['identity','scale','oblique','flip','singleton'])
def test_affine_matches_scipy(kernel,kind):
    from scipy.ndimage import affine_transform
    rng=np.random.default_rng(42)
    a=rng.normal(size=(11,13,7) if kind!='singleton' else (1,13,7)).astype(np.float32)
    matrix=np.eye(3);offset=np.zeros(3);shape=(9,12,10)
    if kind=='scale':matrix=np.diag([.83,1.1,.8]);offset=np.array([-.1,.23,.9])
    if kind=='oblique':matrix=np.array([[.87,.13,-.2],[.22,.96,.13],[.01,.3,.6]])
    if kind=='flip':matrix[0,0]=-1;offset[0]=10
    expected=affine_transform(a,matrix,offset,output_shape=shape,order=1,mode='constant',cval=-1)
    actual=kernel.affine(a,shape,matrix,offset,-1)
    np.testing.assert_array_equal(actual,expected)


def test_softmax_uses_frozen_float_products(kernel):
    from acdc.appliance.reconstruct import make_exp_table
    scale=.123456789
    q=np.random.default_rng(18).integers(-127,128,(1,4,256,256),dtype=np.int16).astype(np.int8)
    values=q[0].astype(np.float32)*np.float32(scale)
    values-=values.max(axis=0,keepdims=True)
    expected=np.exp(values);expected/=expected.sum(axis=0,keepdims=True)
    actual=kernel.softmax(q,make_exp_table(scale))
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-7)
    np.testing.assert_array_equal(actual.argmax(axis=0),expected.argmax(axis=0))


def test_block_restoration_matches_reference_for_oblique_geometry(kernel):
    from types import SimpleNamespace
    from acdc.appliance.reconstruct import Collector,make_exp_table
    from acdc.reconstruct import Collector as Reference
    affine=np.array([[1.3,.2,0,-1],[.1,1.7,0,-2],[0,.04,2.2,0],[0,0,0,1.]])
    volume=SimpleNamespace(native=np.zeros((17,13,7),np.float32),affine=affine,
        working=np.zeros((15,14,10),np.float32),working_shape=(15,14,10),geometry_sha='geometry',
        geometry={'working_affine':np.diag([2.,2.,2.,1.]).tolist()})
    scale=.23
    actual=Collector(volume,kernel,make_exp_table(scale),block_x=3)
    expected=Reference(volume,'model',scale,'ED')
    rng=np.random.default_rng(62)
    box=dict(top=0,left=0,resized_hw=[256,256],original_hw=[14,15])
    with pytest.raises(ContractError):actual.finish()
    for z in range(10):
        q=rng.integers(-20,30,(1,4,256,256),dtype=np.int16).astype(np.int8)
        actual.add(q,z,box)
        expected.add(q,dict(center_index=z,orientation='axial',phase='ED',model_sha256='model',
                           head_scale=scale,geometry_sha256='geometry'),box)
    with pytest.raises(ContractError):actual.add(q,9,box)
    p1,l1=actual.finish(save_probabilities=True);p2,l2=expected.finish()
    np.testing.assert_allclose(p1,p2,rtol=0,atol=1e-6)
    np.testing.assert_array_equal(l1,l2)
