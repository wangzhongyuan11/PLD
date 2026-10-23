"""Physical MPR pixels and label display must be computed on the server."""
import io

import numpy as np
import pytest
from PIL import Image

from acdc.contract import atomic_json,ContractError
from acdc.appliance.native import load_native
from acdc.appliance.mpr import MPRVolume
from acdc.appliance.png import rgba_png


def test_native_mpr_gray_overlay_and_contour():
    k=load_native()
    gray=np.indices((4,4,4)).sum(axis=0).astype(np.float32)
    labels=np.ones(gray.shape,np.uint8)
    args=(gray,labels,[1,1,1],[1,0,0],[0,1,0],(2,2))
    result=k.mpr(*args,level=4,window=4,alpha=0)
    expected=np.rint((np.array([[3,4],[4,5]])-2)/4*255).astype(np.uint8)
    np.testing.assert_array_equal(result[:,:,0],expected)
    np.testing.assert_array_equal(result[:,:,1],expected)
    overlay=k.mpr(*args,level=4,window=4,alpha=1)
    np.testing.assert_array_equal(overlay[:,:,:3],np.broadcast_to([255,0,0],(2,2,3)))
    contour=k.mpr(*args,level=4,window=4,alpha=1,outline=True)
    np.testing.assert_array_equal(contour[:,:,:3],np.repeat(expected[:,:,None],3,axis=2))
    with pytest.raises(ContractError):k.mpr(*args,level=4,window=0)


def test_mpr_volume_preserves_ras_extent_and_png(tmp_path):
    gray=np.indices((5,7,3)).sum(axis=0).astype(np.float32)
    labels=np.zeros(gray.shape,np.uint8);labels[1:4,2:5,1]=3
    np.save(tmp_path/'gray.npy',gray);np.save(tmp_path/'labels.npy',labels)
    affine=np.diag([-2.,3.,8.,1.]);affine[0,3]=10
    atomic_json(tmp_path/'geometry.json',dict(shape=list(gray.shape),affine=affine.tolist()))
    volume=MPRVolume(tmp_path,load_native())
    np.testing.assert_array_equal(volume.bounds,[[2,0,0],[10,18,16]])
    for plane in ('axial','coronal','sagittal'):
        png,meta=volume.render(plane,[6,9,8],shape=(80,100),level=4,window=8)
        image=Image.open(io.BytesIO(png))
        assert image.size==(100,80) and image.mode=='RGBA'
        assert meta['generator']=='pynq_ps' and meta['position']==[6,9,8]
    with pytest.raises(ContractError):volume.render('axial',[11,9,8])
    with pytest.raises(ContractError):volume.render('axial',[6,9,8],shape=(0,10))


def test_cached_mpr_pixels_match_frozen_general_sampler():
    import itertools
    k=load_native();rng=np.random.default_rng(831)
    gray=rng.uniform(-40,300,(19,17,13)).astype(np.float32)
    labels=rng.integers(0,4,gray.shape,dtype=np.uint8)
    sampler=k.prepare_mpr(gray,labels)
    assert not sampler.gray.flags.writeable and not sampler.labels.flags.writeable
    mappings=[]
    for u,v in itertools.permutations(range(3),2):
        dx=np.zeros(3);dy=np.zeros(3);dx[u]=.37;dy[v]=-.29
        mappings.append((np.array([-.5,16.,6.499999999999999]),dx,dy))
    mappings.append((np.array([1.,2.,3.]),np.array([.23,.01,0.]),np.array([0.,-.21,.01])))
    for origin,dx,dy in mappings:
        for alpha,visible,outline in ((0.,14,False),(.45,14,False),(1.,6,True)):
            expected=np.empty((63,79,4),np.uint8)
            args=[gray,labels,expected,np.array(gray.shape,np.int64),origin,dx,dy]
            k.check(k.library.acdc_mpr(*(a.ctypes.data for a in args),63,79,80.,220.,alpha,visible,int(outline)))
            actual=sampler.render(origin,dx,dy,(63,79),level=80.,window=220.,alpha=alpha,visible=visible,outline=outline)
            np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('shape',[(1,1),(7,11),(400,400)])
def test_fixed_sub_png_preserves_all_rgba_bytes(shape):
    rng=np.random.default_rng(891)
    pixels=rng.integers(0,256,(*shape,4),dtype=np.uint8)[:,::-1]
    encoded=rgba_png(pixels)
    with Image.open(io.BytesIO(encoded)) as image:image.verify()
    with Image.open(io.BytesIO(encoded)) as image:
        assert image.mode=='RGBA'
        np.testing.assert_array_equal(np.asarray(image),pixels)
