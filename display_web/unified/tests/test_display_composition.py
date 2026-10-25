from types import SimpleNamespace
import numpy as np
from scipy import ndimage
from prompt_seg.service import SegmentationService


def test_display_lookup_preserves_every_byte_and_original_rounding():
    # All possible input byte values in each channel, with selected/background
    # regions and boundaries. The reference is the previous display equation.
    values=np.arange(256,dtype=np.uint8)
    image=np.stack(list(np.meshgrid(values,values,indexing='ij'))+[np.broadcast_to(values[:,None],(256,256))],axis=-1)
    image=np.pad(image,((4,4),(4,4),(0,0)),mode='wrap')
    for target in (1,2,3):
        native=np.full(image.shape[:2],target,np.uint8);native[100:110,100:110]=0
        selected=native==target;color=np.array(((0,0,0),(25,174,114),(232,150,43),(98,107,213))[target],float)
        expected=image.astype(float);expected[selected]=.6*expected[selected]+.4*color
        expected[ndimage.binary_dilation(selected & ~ndimage.binary_erosion(selected))]=color
        expected=np.rint(expected).clip(0,255).astype(np.uint8)
        request=SimpleNamespace(target_class=target,roi=None)
        actual,mask,stats=SegmentationService(object())._compose(image,request,native,native,4)
        assert np.array_equal(actual,expected)
        assert np.array_equal(mask,selected.astype(np.uint8)*255)
        assert stats['target_pixels']==int(selected.sum()) and stats['area_ratio']==float(selected.mean())
