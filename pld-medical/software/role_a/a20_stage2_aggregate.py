"""A20 阶段 2 结果聚合：三候选 FP32/INT8 与 soup 基线对比（只读，不重新评估）。

读取每个候选的 development/comparison.json（evaluate_quant_overall 一次评估同时给出
FP32 + INT8 五阈值），输出总体、三来源、三面积组、量化损失、阈值与对 soup INT8 的差值。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SOUP_INT8 = 0.8420508068333175
SOUP_FP32 = 0.8440543675818143
SOUP_INT8_SOURCES = {
    "isic2016_test": 0.8833904491556513,
    "isic2017_validation": 0.7783944688067704,
    "isic2017_exposed_validation": 0.7880889564115374,
}
SOUP_INT8_AREAS = {"tiny": 0.7795547830519749, "medium": 0.8469352656216961,
                   "large": 0.8509083683262442}
THRESHOLDS = ("0.45", "0.5", "0.53", "0.56", "0.6")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    labels = ["interp_a19_a025", "interp_a19_a050", "interp_a19_a075"]
    rows = []
    for label in labels:
        path = args.root / label / "development" / "comparison.json"
        if not path.is_file():
            rows.append({"label": label, "status": "missing", "path": path.as_posix()})
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        fp32, int8 = data["fp32"], data["int8"]
        row = {
            "label": label,
            "status": "ok",
            "identity": data["identity"],
            "fp32": {
                "dice": fp32["primary"]["dice"], "iou": fp32["primary"]["iou"],
                "threshold": fp32["selected_threshold"],
                "threshold_dice": {t: fp32["threshold_table"][t]["dice"] for t in fp32["threshold_table"]},
                "sources": {k: v["all"]["dice"] for k, v in fp32["sources"].items()},
                "areas": {k: v["dice"] for k, v in fp32["area_groups_pooled"].items()},
            },
            "int8": {
                "dice": int8["primary"]["dice"], "iou": int8["primary"]["iou"],
                "threshold": int8["selected_threshold"],
                "threshold_dice": {t: int8["threshold_table"][t]["dice"] for t in int8["threshold_table"]},
                "sources": {k: v["all"]["dice"] for k, v in int8["sources"].items()},
                "areas": {k: v["dice"] for k, v in int8["area_groups_pooled"].items()},
                "empty_prediction": int8["area_groups_pooled"]["all"].get("empty_prediction"),
            },
            "holdout": data["holdout"],
        }
        row["quantization_loss_int8_minus_fp32"] = row["int8"]["dice"] - row["fp32"]["dice"]
        row["int8_minus_soup_int8"] = row["int8"]["dice"] - SOUP_INT8
        row["int8_minus_soup_sources"] = {k: row["int8"]["sources"][k] - SOUP_INT8_SOURCES[k]
                                          for k in SOUP_INT8_SOURCES}
        row["int8_minus_soup_areas"] = {k: row["int8"]["areas"][k] - SOUP_INT8_AREAS[k]
                                        for k in SOUP_INT8_AREAS}
        rows.append(row)

    ok = [r for r in rows if r["status"] == "ok"]
    result = {
        "schema": "a20-stage2-aggregate-v1",
        "soup_reference": {"fp32": SOUP_FP32, "int8": SOUP_INT8,
                           "int8_sources": SOUP_INT8_SOURCES, "int8_areas": SOUP_INT8_AREAS},
        "thresholds": list(THRESHOLDS),
        "candidates": rows,
        "ranking_by_int8_dice": [r["label"] for r in
                                 sorted(ok, key=lambda r: -r["int8"]["dice"])],
        "ranking_by_fp32_dice": [r["label"] for r in
                                 sorted(ok, key=lambda r: -r["fp32"]["dice"])],
        "note": "只报告数值与差值，是否采用由 Codex 裁定；未打开 2018 最终测试掩膜",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")

    print(f"{'label':<18}{'FP32':>12}{'INT8':>12}{'loss':>12}{'dINT8-soup':>13}  thr(fp/int8)")
    for row in rows:
        if row["status"] != "ok":
            print(f"{row['label']:<18}{'MISSING':>12}")
            continue
        print(f"{row['label']:<18}{row['fp32']['dice']:>12.10f}{row['int8']['dice']:>12.10f}"
              f"{row['quantization_loss_int8_minus_fp32']:>12.10f}{row['int8_minus_soup_int8']:>13.10f}"
              f"  {row['fp32']['threshold']:.2f}/{row['int8']['threshold']:.2f}")
    print()
    for row in rows:
        if row["status"] != "ok":
            continue
        print(f"[{row['label']}]")
        print("  INT8 thr:", {t: round(v, 8) for t, v in row["int8"]["threshold_dice"].items()})
        print("  INT8 src:", {k: round(v, 8) for k, v in row["int8"]["sources"].items()})
        print("  INT8 area:", {k: round(v, 8) for k, v in row["int8"]["areas"].items()})
        print("  FP32 src:", {k: round(v, 8) for k, v in row["fp32"]["sources"].items()})
        print("  FP32 area:", {k: round(v, 8) for k, v in row["fp32"]["areas"].items()})
        print("  vs soup INT8 src:", {k: round(v, 8) for k, v in row["int8_minus_soup_sources"].items()})
        print("  vs soup INT8 area:", {k: round(v, 8) for k, v in row["int8_minus_soup_areas"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
