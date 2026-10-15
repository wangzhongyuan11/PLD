import unittest

import numpy as np

from camus.device import EmulatedTileAccelerator
from camus.graph import CamusGraph
from camus.model import DEFAULT_MODEL_ROOT, load_model, load_golden
from camus.reference import ReferenceNetwork


class FullFrozenNetworkTests(unittest.TestCase):
    """One explicit full-package numerical gate; not repeated by small unit tests."""

    def test_reference_and_tiled_emulator_match_all_layer_goldens(self):
        model = load_model(DEFAULT_MODEL_ROOT)
        first_input = load_golden(model, model.layers[0], "input_full")
        reference = ReferenceNetwork(model).forward(first_input, capture=True)
        for layer in model.layers:
            self.assertTrue(np.array_equal(
                reference["captures"][f"{layer.index:02d}_{layer.name}_input"],
                load_golden(model, layer, "input_full"),
            ))
            self.assertTrue(np.array_equal(
                reference["captures"][f"{layer.index:02d}_{layer.name}_output"],
                load_golden(model, layer, "output_full_golden"),
            ))
        emulator = CamusGraph(model, EmulatedTileAccelerator()).forward(first_input, capture=True)
        for layer in model.layers:
            self.assertTrue(np.array_equal(
                emulator["captures"][f"{layer.index:02d}_{layer.name}_input"],
                load_golden(model, layer, "input_full"),
            ))
            self.assertTrue(np.array_equal(
                emulator["captures"][f"{layer.index:02d}_{layer.name}_output"],
                load_golden(model, layer, "output_full_golden"),
            ))
        self.assertTrue(np.array_equal(reference["head"], emulator["head"]))
        self.assertLess(int(emulator["head"].min()), 0)


if __name__ == "__main__":
    unittest.main()
