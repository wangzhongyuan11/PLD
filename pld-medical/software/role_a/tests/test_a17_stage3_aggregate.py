"""A17 阶段 3 聚合函数单测（合成记录，不做 638 前向）。"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from software.role_a import a17_stage3 as s3


def rec(stem, band, source, d_id, d_comb, d5, d10, d20, iou_id=0.7):
    r = {"stem": stem, "band": band, "source": source}
    for rule, d in (("identity", d_id), ("combined", d_comb), ("keep_ratio_5", d5),
                    ("keep_ratio_10", d10), ("keep_ratio_20", d20)):
        r[f"{rule}_dice"] = d
        r[f"{rule}_iou"] = iou_id
    return r


class AggregateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            rec("A", "tiny", "s1", 0.50, 0.10, 0.45, 0.40, 0.20),      # combined 严重退步 -0.40
            rec("B", "medium", "s1", 0.60, 0.65, 0.62, 0.64, 0.66),    # 各规则改善
            rec("C", "large", "s2", 0.70, 0.71, 0.69, 0.70, 0.72),
            rec("D", "tiny", "s2", 0.80, 0.82, 0.81, 0.83, 0.84),
        ]
        self.old13 = ["A", "B"]

    def test_aggregate_has_no_keyerror_and_all_rules(self) -> None:
        out = s3.aggregate(self.records, self.old13)
        self.assertEqual(set(out["table"]), set(s3.RULES))
        self.assertEqual(set(out["old13_improvement"]), set(s3.RULES))

    def test_overall_and_groups(self) -> None:
        out = s3.aggregate(self.records, self.old13)
        self.assertAlmostEqual(out["table"]["identity"]["all"]["dice"], 0.65, places=12)
        self.assertEqual(out["table"]["identity"]["all"]["n"], 4)
        self.assertEqual(set(out["table"]["identity"]["area_groups"]), {"tiny", "medium", "large"})
        self.assertEqual(set(out["table"]["identity"]["sources"]), {"s1", "s2"})
        self.assertEqual(out["table"]["identity"]["area_groups"]["tiny"]["n"], 2)

    def test_severe_and_worst(self) -> None:
        out = s3.aggregate(self.records, self.old13)
        comb = out["table"]["combined"]["vs_identity"]
        self.assertEqual(comb["severe_gt_0.1"], 1)          # A 掉 0.40
        self.assertEqual(comb["severe_stems"], ["A"])
        self.assertAlmostEqual(comb["worst_delta"], -0.40, places=12)
        self.assertEqual(comb["worst_stem"], "A")
        self.assertEqual(out["table"]["keep_ratio_20"]["vs_identity"]["severe_gt_0.1"], 1)  # A 掉 0.30
        self.assertEqual(out["table"]["keep_ratio_5"]["vs_identity"]["severe_gt_0.1"], 0)

    def test_old13_improvement_is_independent_field(self) -> None:
        out = s3.aggregate(self.records, self.old13)
        imp = out["old13_improvement"]
        self.assertEqual(imp["identity"]["mean_delta_vs_identity"], 0.0)
        self.assertAlmostEqual(imp["combined"]["mean_delta_vs_identity"], (0.10 + 0.65)/2 - (0.50 + 0.60)/2, places=12)
        self.assertAlmostEqual(imp["keep_ratio_5"]["min_delta"], min(0.45-0.50, 0.62-0.60), places=12)
        self.assertEqual(imp["combined"]["severe_remaining"], 1)

    def test_jaccard_rule_applied(self) -> None:
        recs = [rec("X", "tiny", "s1", 0.5, 0.5, 0.5, 0.5, 0.5, iou_id=0.5)]   # IoU 0.5 < 0.65 -> 记 0
        out = s3.aggregate(recs, [])
        self.assertAlmostEqual(out["table"]["identity"]["all"]["jaccard_official"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()