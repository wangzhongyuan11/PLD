import unittest

from camus.model import DEFAULT_MODEL_ROOT, load_model
from camus.plan import build_plan


class PlanTests(unittest.TestCase):
    def test_plan_has_exact_workload_and_expected_chunks(self):
        plan = build_plan(load_model(DEFAULT_MODEL_ROOT))
        self.assertEqual(plan["total_mac_count"], 2_656_043_008)
        self.assertEqual(plan["layers"][7]["output_chunks"], [56, 56, 16])
        self.assertEqual(plan["layers"][8]["output_chunks"], [32, 32])
        self.assertEqual(plan["nonconv"]["maxpool2_runs"], 3)
        self.assertEqual(plan["nonconv"]["q24_rescale_runs"], 6)
        self.assertEqual(plan["nonconv"]["concat_runs"], 3)
        for layer in plan["layers"]:
            for chunk in layer["chunk_schedule"]:
                self.assertLessEqual(chunk["input_bank_usage"], chunk["input_bank_limit"])
                self.assertLessEqual(chunk["output_bank_usage"], chunk["output_bank_limit"])
                self.assertLessEqual(chunk["weight_bank_usage"], chunk["weight_bank_limit"])


if __name__ == "__main__":
    unittest.main()
