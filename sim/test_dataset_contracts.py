import copy
import unittest

from reference.datasets import collection, validate_manifest
from reference.tinyunet import Model, BASELINE_SPEC_SHA256


class DatasetContractTests(unittest.TestCase):
    def manifest(self, name="final_dev638"):
        model = Model()
        selected = collection(name)
        return {"schema_version": 2, "passed": True, "complete_test_set": True,
                "model_spec_sha256": model.spec_hash, "threshold": model.threshold,
                "collection": {k: selected[k] for k in ("name", "role", "primary", "count", "ids_sha256")},
                "records": selected["cases"]}

    def test_complete_primary_and_named_subset_remain_distinct(self):
        primary = self.manifest()
        subset = self.manifest("final_isic2016_dev376")
        self.assertTrue(validate_manifest(primary, Model()))
        self.assertTrue(validate_manifest(subset, Model()))
        self.assertTrue(primary["collection"]["primary"])
        self.assertFalse(subset["collection"]["primary"])
        primary["records"] = subset["records"]
        with self.assertRaisesRegex(ValueError, "completion flag"):
            validate_manifest(primary)
        primary["complete_test_set"] = False
        self.assertFalse(validate_manifest(primary))

    def test_wrong_threshold_source_id_and_duplicate_are_rejected(self):
        original = self.manifest()
        for change in ("threshold", "source", "duplicate", "identity"):
            data = copy.deepcopy(original)
            if change == "threshold": data["threshold"]["threshold_q"] = 0
            elif change == "source": data["records"][0]["source"] = "unverified"
            elif change == "duplicate": data["records"].append(data["records"][0])
            else: data["collection"]["count"] = 376
            with self.assertRaises(ValueError, msg=change):
                validate_manifest(data, Model())

    def test_legacy_379_manifest_cannot_claim_final_model(self):
        data = {"passed": True, "complete_test_set": True, "model_spec_sha256": BASELINE_SPEC_SHA256,
                "records": collection("isic2016_test379")["cases"]}
        self.assertTrue(validate_manifest(data))
        data["model_spec_sha256"] = Model().spec_hash
        with self.assertRaisesRegex(ValueError, "legacy dataset schema"):
            validate_manifest(data)


if __name__ == "__main__": unittest.main()
