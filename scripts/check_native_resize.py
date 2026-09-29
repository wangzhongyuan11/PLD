"""Directed scalar/NEON RGB tests against Pillow plus native capacity guards."""
import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference.native_resize import NativeBilinear


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library',type=Path)
    parser.add_argument('--pillow-rows',action='store_true')
    args=parser.parse_args()
    filters=getattr(Image,'Resampling',Image)
    rng=np.random.default_rng(2710)
    cases=0
    candidate=NativeBilinear
    if args.pillow_rows:
        from reference.pillow_resize import NativePillowBilinear
        candidate=NativePillowBilinear
    with candidate(args.library) as resize:
        for h,w in ((1,1),(1,39),(37,1),(39,53),(256,256),(411,617)):
            for pixels in (rng.integers(0,256,(h,w,3),dtype=np.uint8),np.full((h,w,3),255,dtype=np.uint8)):
                image=Image.fromarray(pixels)
                for size in ((1,1),(256,191),(191,256),(w,h),(23,31)):
                    expected=np.asarray(image.resize(size,filters.BILINEAR))
                    actual=np.asarray(resize.resize(image,size))
                    np.testing.assert_array_equal(actual,expected,err_msg=str((h,w,size)))
                    cases+=1
        source=np.arange(72,dtype=np.uint8).reshape(2,12,3)
        output=np.full((2,5,3),0xa5,dtype=np.uint8)
        bounds,weights,stride=resize.coeff(12,5)
        for src_bytes,dst_bytes in ((source.nbytes-1,output.nbytes),(source.nbytes,output.nbytes-1)):
            code=resize.horizontal(source.ctypes.data,src_bytes,output.ctypes.data,dst_bytes,
                                   12,2,5,bounds.ctypes.data,weights.ctypes.data,stride)
            assert code==1 and np.all(output==0xa5)
        bad_bounds=bounds.copy(); bad_bounds[0,0]=12
        code=resize.horizontal(source.ctypes.data,source.nbytes,output.ctypes.data,output.nbytes,
                               12,2,5,bad_bounds.ctypes.data,weights.ctypes.data,stride)
        assert code==1 and np.all(output==0xa5)
        if args.pillow_rows:
            rgba=np.zeros((2,12,4),dtype=np.uint8);rgba[:,:,:3]=source
            rows=np.array([rgba[y].ctypes.data for y in range(2)],dtype=np.uintp)
            for capacity,start,end in ((output.nbytes-1,0,2),(output.nbytes,1,1),(output.nbytes,0,3)):
                code=resize.horizontal_rows(rows.ctypes.data,output.ctypes.data,capacity,12,2,5,
                        start,end,bounds.ctypes.data,weights.ctypes.data,stride)
                assert code==1 and np.all(output==0xa5)
            code=resize.horizontal_rows(rows.ctypes.data,output.ctypes.data,output.nbytes,12,2,5,
                    0,1,bounds.ctypes.data,weights.ctypes.data,stride)
            assert code==0 and np.all(output[1]==0xa5)
            np.testing.assert_array_equal(output[:1],np.asarray(Image.fromarray(source[:1]).resize((5,1),filters.BILINEAR)))
    print(f'NATIVE_RGB_PASS cases={cases} random_white_identity_odd_edges_tails_capacity_bounds')


if __name__=='__main__': main()
