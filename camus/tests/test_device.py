import unittest

import numpy as np

from camus.device import (
    DeviceContractError,
    EmulatedTileAccelerator,
    choose_camus_tile,
    input_bank_usage,
    output_bank_usage,
    output_channel_chunks,
    weight_bank_usage,
)


class DeviceContractTests(unittest.TestCase):
    def test_expected_output_channel_chunks(self):
        self.assertEqual(output_channel_chunks(128, 128, 3), (56, 56, 16))
        self.assertEqual(output_channel_chunks(192, 64, 3), (32, 32))
        self.assertEqual(weight_bank_usage(56, 128, 3), 2016)
        self.assertEqual(weight_bank_usage(16, 128, 3), 576)
        self.assertEqual(weight_bank_usage(32, 192, 3), 1728)

    def test_rejects_old_modes_and_invalid_relu(self):
        device = EmulatedTileAccelerator()
        for mode in (5, 6):
            with self.assertRaises(DeviceContractError):
                device.configure(mode)
        with self.assertRaises(DeviceContractError):
            device.configure(1, relu=True)

    def test_rejects_shape_and_capacity_before_run(self):
        device = EmulatedTileAccelerator()
        x = np.zeros((1, 193, 8, 8), dtype=np.int8)
        w = np.zeros((1, 193, 1, 1), dtype=np.int8)
        b = np.zeros(1, dtype=np.int32)
        m = np.ones(1, dtype=np.int32)
        with self.assertRaises(DeviceContractError):
            device.conv(x, w, b, m, padding=0, relu=False)
        with self.assertRaises(DeviceContractError):
            output_channel_chunks(193, 8, 3)
        with self.assertRaises(DeviceContractError):
            choose_camus_tile(20000, 64, 1, 1, 3)

    def test_tile_capacity_and_boundary_halo_semantics(self):
        tile_h, tile_w = choose_camus_tile(192, 32, 64, 64, 3)
        self.assertLessEqual(input_bank_usage(192, tile_h, tile_w, 3), 32768)
        self.assertLessEqual(output_bank_usage(32, tile_h, tile_w), 8192)
        device = EmulatedTileAccelerator()
        x = np.zeros((1, 4, 8, 8), dtype=np.int8)
        x[0, :, 3:5, 3:5] = 1
        w = np.zeros((4, 4, 3, 3), dtype=np.int8)
        w[:, 0, 1, 1] = 1
        bias = np.zeros(4, dtype=np.int32)
        mult = np.full(4, 1 << 24, dtype=np.int32)
        out = device.conv(x, w, bias, mult, padding=1, relu=True)
        self.assertEqual(out.shape, (1, 4, 8, 8))
        self.assertTrue(np.array_equal(out, x))
        self.assertEqual(out.flags["WRITEABLE"], True)


if __name__ == "__main__":
    unittest.main()
