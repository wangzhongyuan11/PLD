import unittest

import numpy as np

from camus.integer import (
    apply_relu_int8,
    conv2d_int32,
    maxpool2_int8,
    nearest_upsample2_int8,
    quantize_letterbox_input,
    requantize_int8,
    rescale_int8,
    round_divide_even,
)


class IntegerMathTests(unittest.TestCase):
    def test_q24_positive_negative_ties_and_saturation(self):
        products = np.asarray([3 * (1 << 23), 5 * (1 << 23), -(3 * (1 << 23)), 127 * (1 << 24)], dtype=np.int64)
        self.assertEqual(round_divide_even(products).tolist(), [2, 2, -2, 127])
        acc = np.zeros((1, 1, 2, 2), dtype=np.int64)
        acc[0, 0] = np.asarray(
            [[(1 << 24) - 1, -(1 << 24) + 1],
             [127 * (1 << 24), -127 * (1 << 24)]], dtype=np.int64,
        )
        out = requantize_int8(acc, np.asarray([1], dtype=np.int32))
        self.assertEqual(out[0, 0].flatten().tolist(), [1, -1, 127, -127])

    def test_relu_does_not_change_negative_result_dtype(self):
        values = np.asarray([-127, -1, 0, 1, 127], dtype=np.int8)
        self.assertEqual(apply_relu_int8(values).tolist(), [0, 0, 0, 1, 127])

    def test_conv_negative_and_bias_exact(self):
        x = np.asarray([[[[-127, 0, 10], [0, 1, -20], [3, -4, 5]]]], dtype=np.int8)
        w = np.asarray([[[[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]]], dtype=np.int8)
        b = np.asarray([-200], dtype=np.int32)
        acc = conv2d_int32(x, w, b, padding=0)
        self.assertEqual(acc.dtype, np.int32)
        self.assertEqual(acc[0, 0, 0, 0], -68)

    def test_pool_upsample_rescale(self):
        x = np.asarray([[[[1, -2], [3, 4]]]], dtype=np.int8)
        self.assertEqual(maxpool2_int8(x)[0, 0].tolist(), [[4]])
        up = nearest_upsample2_int8(x)
        self.assertEqual(up.shape, (1, 1, 4, 4))
        self.assertEqual(up[0, 0, 0, :2].tolist(), [1, 1])
        self.assertEqual(rescale_int8(x, 1 << 24)[0, 0].tolist(), x[0, 0].tolist())

    def test_letterbox_quantization_exact_zero_and_full(self):
        canvas = np.zeros((2, 2), dtype=np.float32)
        canvas[0, 0] = 1.0
        quantized = quantize_letterbox_input(canvas)
        self.assertEqual(quantized.tolist(), [[127, 0], [0, 0]])
        self.assertEqual(quantized.dtype, np.int8)


if __name__ == "__main__":
    unittest.main()
