"""Final/legacy identity and mask boundaries must survive model migration."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from reference.tinyunet import (Model, DEFAULT_MODEL_ROOT, BASELINE_SPEC_SHA256,
                               FINAL_SPEC_SHA256, threshold_contract)


class ModelVersionTests(unittest.TestCase):
    def test_registered_versions_and_wrong_explicit_identity(self):
        final = Model()
        legacy = Model(DEFAULT_MODEL_ROOT.parents[1]/"role_a_handoff/role_a/int8")
        self.assertEqual(final.spec_hash, FINAL_SPEC_SHA256)
        self.assertEqual(final.threshold["threshold_q"], -1)
        self.assertEqual(legacy.spec_hash, BASELINE_SPEC_SHA256)
        self.assertEqual(legacy.threshold["threshold_q"], 0)
        with self.assertRaisesRegex(ValueError, "hash changed"):
            Model(expected_spec=BASELINE_SPEC_SHA256)
        with self.assertRaisesRegex(ValueError, "another model"):
            Model(manifest=legacy.manifest_path)

    def test_unknown_spec_and_changed_frozen_parameter_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"model_spec.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unregistered"):
                Model(root)
            frozen = json.loads(Model().manifest_path.read_text(encoding="utf-8"))
            frozen["files"]["int8/weights_int8.bin"] = "0"*64
            path = root/"manifest.json"
            path.write_text(json.dumps(frozen), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                Model(manifest=path)

    def test_all_int8_values_match_probability_thresholds(self):
        scale = Model().layers[-1]["output_scale"]
        q = np.arange(-127, 128)
        probability = 1/(1+np.exp(-q.astype(np.float32)*np.float32(scale)))
        for p, expected in zip((.45, .50, .53, .56, .60), (-1, 0, 1, 2, 3)):
            contract = threshold_contract(p, scale)
            self.assertEqual(contract["threshold_q"], expected)
            np.testing.assert_array_equal(q >= expected, probability >= p)
        for p, scale in ((0, 1), (1, 1), (.5, 0), (float("nan"), 1)):
            with self.assertRaises(ValueError):
                threshold_contract(p, scale)


if __name__ == "__main__":
    unittest.main()
