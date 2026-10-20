import ctypes as ct
import numpy as np
import pytest
from acdc.optimization.layout import pack_feature
from acdc.optimization.native_layout import load_native_layout


@pytest.fixture(scope='module')
def native():
    return load_native_layout()


def test_real_boundary_layout_exact_and_independent(native):
    rng=np.random.default_rng(20102026)
    x=rng.integers(-127,128,(1,3,256,256),dtype=np.int16).astype(np.int8)
    guarded=np.full(524288+128,0xa5,np.uint8)
    packed=guarded[64:-64]
    native.pack(packed,x)
    assert packed.tobytes()==pack_feature(x[0]).tobytes()
    assert np.all(guarded[:64]==0xa5) and np.all(guarded[-64:]==0xa5)
    head=rng.integers(-127,128,(4,256,256),dtype=np.int16).astype(np.int8)
    physical=pack_feature(head).view(np.uint8).ravel()
    actual=native.head(physical)
    assert np.array_equal(actual[0],head)
    physical[:]=0
    assert np.array_equal(actual[0],head)


def test_reject_values_padding_capacity_and_alias(native):
    x=np.zeros((1,3,256,256),np.int8);dst=np.zeros(524288,np.uint8)
    for channel in range(3):
        x[:]=0;x[0,channel,255,255]=-128
        with pytest.raises(ValueError):native.pack(dst,x)
    for lane,value in [(0,128),(3,128),(4,1),(7,255)]:
        dst[:]=0;dst.reshape(-1,8)[-1,lane]=value
        with pytest.raises(ValueError):native.head(dst)
    with pytest.raises(ValueError):native.pack(dst,x[...,::-1])
    alias=dst[:196608].view(np.int8).reshape(x.shape)
    with pytest.raises(ValueError):native.pack(dst,alias)
    guarded=np.full(524288,0xa5,np.uint8);x[:]=0
    assert native.library.acdc_pack_input(guarded.ctypes.data,1,x.ctypes.data,x.nbytes)==1
    assert np.all(guarded==0xa5)
    assert native.library.acdc_unpack_head(ct.c_void_p(),262144,dst.ctypes.data,dst.nbytes)==1
