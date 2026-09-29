"""Independent numeric and graph regressions; no PYNQ dependency."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.tinyunet import apply_lut, conv2d, maxpool2, requant_q24, rescale_lut, upsample2
from scripts.gen_requant_vectors import expected
from reference.tiling import choose_tile, conv2d_tiled, INPUT_BANK_BYTES, OUTPUT_BANK_BYTES


class NumericTests(unittest.TestCase):
    def test_signed_ties_saturation_and_int32_edges(self):
        values = np.array([-(1 << 31), -257, -255, -5, -3, -1, 0, 1, 3, 5, 255, 257, (1 << 31)-1])
        for multiplier in (0, 1, 1 << 23, (1 << 24)+1, (1 << 31)-1):
            want = [expected(int(a), multiplier) for a in values]
            np.testing.assert_array_equal(requant_q24(values, multiplier), want)

    def test_convolution_multichannel_border_bias(self):
        rng = np.random.default_rng(44)
        for kernel in (1, 3):
            x = rng.integers(-127, 128, (1, 3, 5, 6), dtype=np.int8)
            weights = rng.integers(-127, 128, (2, 3, kernel, kernel), dtype=np.int8)
            bias = np.array([417, -739], dtype=np.int32)
            multipliers = [1711, 9221]
            got = conv2d(x, weights, bias, multipliers, kernel//2)
            want = np.zeros(got.shape, dtype=np.int8)
            for o in range(2):
                for y in range(5):
                    for z in range(6):
                        acc = int(bias[o])
                        for c in range(3):
                            for ky in range(kernel):
                                for kx in range(kernel):
                                    iy, ix = y+ky-kernel//2, z+kx-kernel//2
                                    if 0 <= iy < 5 and 0 <= ix < 6:
                                        acc += int(x[0,c,iy,ix]) * int(weights[o,c,ky,kx])
                        want[0,o,y,z] = expected(acc, multipliers[o])
            np.testing.assert_array_equal(got, want)

    def test_pool_and_nearest_preserve_channel_order(self):
        x = np.arange(32, dtype=np.int8).reshape(1,2,4,4)
        pooled = maxpool2(x)
        np.testing.assert_array_equal(pooled, [[[[5,7],[13,15]],[[21,23],[29,31]]]])
        expanded = upsample2(pooled)
        np.testing.assert_array_equal(expanded[:,:,::2,::2], pooled)
        np.testing.assert_array_equal(expanded[:,:,1::2,1::2], pooled)

    def test_lut_float32_order_all_values(self):
        for source, target in ((0.016195047558761957,0.009966377198226808),(0.031,0.127),(0.25,0.01)):
            x = np.arange(-127,128,dtype=np.int16).astype(np.int8)
            table = rescale_lut(source,target)
            want=[]
            for q in x:
                real=np.float32(max(int(q),0))*np.float32(source)
                want.append(np.clip(np.rint(real/np.float32(target)),-127,127))
            np.testing.assert_array_equal(apply_lut(x,table),want)

    def test_reject_out_of_range_accumulator(self):
        with self.assertRaises(OverflowError):
            requant_q24(np.array([1 << 31]),1)

    def test_halo_and_partial_tiles(self):
        rng = np.random.default_rng(48)
        x = rng.integers(-127,128,(1,3,7,11),dtype=np.int8)
        weights = rng.integers(-127,128,(5,3,3,3),dtype=np.int8)
        bias = np.array([-711,0,117,300,-400])
        multipliers = [701,311,1171,1723,503]
        whole = conv2d(x,weights,bias,multipliers,1)
        tiled = conv2d_tiled(x,weights,bias,multipliers,1,tile_shape=(3,4))
        np.testing.assert_array_equal(tiled,whole)

    def test_tile_plan_accounts_for_padded_channel_banks(self):
        for ci, co, h, w, k in ((3,8,256,256,3),(96,32,64,64,3),(8,1,256,256,1)):
            th,tw=choose_tile(ci,co,h,w,k)
            self.assertLessEqual(((ci+3)//4)*(th+k-1)*(tw+k-1),INPUT_BANK_BYTES)
            self.assertLessEqual(((co+7)//8)*th*tw,OUTPUT_BANK_BYTES)


if __name__ == "__main__":
    unittest.main()
