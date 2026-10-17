"""A19 INT8 验收的只读前置门禁（不写任何产物、不训练、不读 2018 留出集）。

与首版不同：本版把每一项门禁做成**硬校验**，任一不符即以非零退出码结束，
不再只打印布尔值。JSON 报告始终打印，最后汇总 ``gate_failures``。

硬门禁清单：
1. best.pt SHA256 == 期望值（默认 A19 best）；
2. 冻结划分的 train / development 摘要 == 冻结值（2309/638）；
3. 校准 96 张：总数 96、逐来源 32/32/32、ID 唯一、**全部属训练集**、
   与开发集**交集为空**、校准 ID 摘要 == 口径基线值；
4. 量化层数 == 15、权重总字节 == 121,760，且层名与参考 spec 完全一致。

退出码：0 = 全部门禁通过；2 = 有门禁失败（明细在 stdout 的 gate_failures）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from software.role_a import quant

EXPECTED = {
    "checkpoint_sha256": "57b78cfc6df14b6f2fb53aef42c617836fe6b22c394960a0bcaf95e5e1bf4b53",
    "train_ids_sha256": "0b4f6ade53314b206fb8bcccf7d7169c90d2a35fc987dbc70404c9fd4fba2bb0",
    "development_ids_sha256": "4ead5256747a87003cb341ce647e2863ec60e5d2d82a737a9572d2963805133d",
    "calibration_count": 96,
    "calibration_per_source": {"isic2016_train": 32, "isic2017_new_train": 32,
                               "isic2017_exposed_train": 32},
    "calibration_ids_sha256": "db22925dae0cda81afc7acdf3834376b8a76488174881341527b3266b3059470",
    "layer_count": 15,
    "weight_bytes_total": 121760,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-images", type=int, default=EXPECTED["calibration_count"])
    parser.add_argument("--reference-spec", type=Path,
                        help="已验收的同构 model_spec.json，用于硬比对层名与层数")
    args = parser.parse_args()

    failures: list[str] = []

    def gate(name: str, passed: bool, detail: str = "") -> None:
        if not passed:
            failures.append(f"{name}: {detail}" if detail else name)

    report: dict = {}
    actual_hash = quant._sha256(args.checkpoint)
    report["checkpoint"] = str(args.checkpoint)
    report["checkpoint_sha256"] = actual_hash
    report["checkpoint_bytes"] = args.checkpoint.stat().st_size
    gate("checkpoint_sha256", actual_hash == EXPECTED["checkpoint_sha256"],
         f"实际 {actual_hash} != 期望 {EXPECTED['checkpoint_sha256']}")

    model, config = quant._load_fp32(args.checkpoint)
    state = model.state_dict()
    finite = all(bool(torch.isfinite(parameter).all()) for parameter in state.values())
    report["config"] = {k: config[k] for k in sorted(config) if not isinstance(config[k], (dict, list))}
    gate("weights_finite", finite, "存在非有限权重")
    gate("base_channels_8", int(config["base_channels"]) == 8, str(config["base_channels"]))

    assembly = quant.load_overall_assembly()
    grouped = quant.group_train_items(assembly)
    items, records = quant.select_calibration_items(grouped, args.calibration_images)
    train_ids = set(assembly.train_ids)
    dev_ids = {stem for ids in assembly.dev_ids.values() for stem in ids}
    stems = [item.stem for item in items]
    stems_digest = hashlib.sha256("\n".join(stems).encode("utf-8")).hexdigest()
    per_source = {row["name"]: int(row["selected"]) for row in records}
    train_digest = str(assembly.split_digest["train_ids"])
    dev_digest = str(assembly.split_digest["development_all"])

    report["calibration"] = {
        "requested": args.calibration_images,
        "selected_total": len(items),
        "per_source": [{"name": row["name"], "train_count": row["train_count"],
                        "selected": row["selected"]} for row in records],
        "unique_ids": len(set(stems)) == len(stems),
        "leaked_ids": sorted({stem for stem in stems if stem in dev_ids}),
        "non_train_ids": sorted({stem for stem in stems if stem not in train_ids}),
        "ids_sha256": stems_digest,
        "first_stem": stems[0],
        "last_stem": stems[-1],
    }
    report["split"] = {"train_total": int(assembly.counts["train_total"]),
                       "development_total": int(assembly.counts["development_total"]),
                       "train_ids_sha256": train_digest,
                       "development_ids_sha256": dev_digest}

    gate("train_total_2309", int(assembly.counts["train_total"]) == 2309,
         str(assembly.counts["train_total"]))
    gate("development_total_638", int(assembly.counts["development_total"]) == 638,
         str(assembly.counts["development_total"]))
    gate("train_ids_digest", train_digest == EXPECTED["train_ids_sha256"], train_digest)
    gate("development_ids_digest", dev_digest == EXPECTED["development_ids_sha256"], dev_digest)
    gate("calibration_count", len(items) == EXPECTED["calibration_count"], str(len(items)))
    gate("calibration_per_source", per_source == EXPECTED["calibration_per_source"], str(per_source))
    gate("calibration_ids_unique", len(set(stems)) == len(stems), "存在重复校准 ID")
    gate("calibration_all_train", not report["calibration"]["non_train_ids"],
         str(report["calibration"]["non_train_ids"][:5]))
    gate("calibration_no_development_leak", not report["calibration"]["leaked_ids"],
         str(report["calibration"]["leaked_ids"][:5]))
    gate("calibration_ids_digest", stems_digest == EXPECTED["calibration_ids_sha256"], stems_digest)

    stats = quant.calibrate(model, quant.ISICDataset(items, root=None, size=int(config["size"])),
                            len(items))
    _quant_model, layers = quant._quantized_model(model, stats)
    layer_names = [layer["name"] for layer in layers]

    def _weight_bytes(layer: dict) -> int:
        size = 1
        for value in layer["weight_shape"]:
            size *= int(value)
        return size

    weight_total = sum(_weight_bytes(layer) for layer in layers)
    report["layers"] = {"count": len(layers), "names": layer_names,
                        "weight_bytes_total": weight_total}
    gate("layer_count", len(layers) == EXPECTED["layer_count"], str(len(layers)))
    gate("weight_bytes_total", weight_total == EXPECTED["weight_bytes_total"], str(weight_total))
    if args.reference_spec is not None:
        reference = json.loads(args.reference_spec.read_text(encoding="utf-8"))
        ref_names = [layer["name"] for layer in reference["layers"]]
        ref_weight_total = sum(_weight_bytes(layer) for layer in reference["layers"])
        report["reference_spec"] = {
            "path": str(args.reference_spec), "layers": len(ref_names),
            "same_layer_names": ref_names == layer_names,
            "same_weight_total": ref_weight_total == weight_total,
            "reference_calibration_images": reference.get("calibration_images"),
            "reference_calibration_mode": reference.get("calibration_mode"),
            "reference_checkpoint_sha256": reference.get("checkpoint_sha256"),
        }
        gate("reference_layer_names", ref_names == layer_names, "层名与参考 spec 不一致")
        gate("reference_weight_total", ref_weight_total == weight_total,
             f"{ref_weight_total} != {weight_total}")
    else:
        report["reference_spec"] = None

    report["gate_failures"] = failures
    report["gate_passed"] = not failures
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        print(f"[GATE-FAIL] {len(failures)} 项门禁未通过，禁止进入 export", file=sys.stderr)
        return 2
    print("[GATE-OK] 全部门禁通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
