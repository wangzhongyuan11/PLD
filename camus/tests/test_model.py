import json
from pathlib import Path
import shutil
import tempfile
import unittest

from camus.model import DEFAULT_MODEL_ROOT, MODEL_SPEC_SHA256, ModelError, load_model


class FrozenModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = load_model(DEFAULT_MODEL_ROOT)

    def test_frozen_topology_and_bounds(self):
        model = self.original
        self.assertEqual(len(model.layers), 15)
        self.assertEqual(model.layers[-1].weight.shape, (4, 16, 1, 1))
        self.assertEqual(model.head_scale, 0.19834566491795338)
        self.assertEqual([layer.relu for layer in model.layers[:-1]], [True] * 14)
        self.assertFalse(model.layers[-1].relu)

    def test_tampered_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "int8_export"
            shutil.copytree(DEFAULT_MODEL_ROOT, root)
            target = root / model_relative_weight_path(self.original)
            original = target.read_bytes()
            target.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaises(ModelError):
                load_model(root)

    def test_wrong_spec_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "int8_export"
            shutil.copytree(DEFAULT_MODEL_ROOT, root)
            spec = root / "model_spec.json"
            payload = json.loads(spec.read_text(encoding="utf-8"))
            payload["graph"]["convs"] = 14
            spec.write_text(json.dumps(payload) + "\n", encoding="utf-8", newline="\n")
            with self.assertRaises(ModelError):
                load_model(root)

    def test_missing_model_root_is_rejected(self):
        with self.assertRaises(ModelError):
            load_model(Path("does-not-exist-int8-export"))


def model_relative_weight_path(model) -> str:
    layer = model.layers[0]
    return f"{layer.directory}/weight.bin"


if __name__ == "__main__":
    unittest.main()
