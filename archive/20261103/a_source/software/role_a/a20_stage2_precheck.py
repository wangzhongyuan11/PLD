"""A20 阶段 2 前置核实（只读）：候选哈希对阶段 1 清单、冻结划分摘要、输出目录归属。

不写任何产物、不校准、不评估。任一不符即非零退出，避免带病进入导出。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from software.role_a import a20_interp as a20
from software.role_a import quant

EXPECTED = {
    "train_total": 2309,
    "development_total": 638,
    "train_ids_sha256": "0b4f6ade53314b206fb8bcccf7d7169c90d2a35fc987dbc70404c9fd4fba2bb0",
    "development_ids_sha256": "4ead5256747a87003cb341ce647e2863ec60e5d2d82a737a9572d2963805133d",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    failures: list[str] = []
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    source_hashes = {name: entry["sha256"] for name, entry in summary["sources"].items()}

    gates: dict = {}
    candidates = []
    for entry in summary["candidates"]:
        path = Path(entry["checkpoint"])
        actual = a20.sha256(path)
        candidates.append({
            "label": entry["label"], "alpha": entry["alpha_a19_share"],
            "path": path.as_posix(), "listed_sha256": entry["checkpoint_sha256"],
            "actual_sha256": actual, "sha_matches_listing": actual == entry["checkpoint_sha256"],
            "bytes": path.stat().st_size,
        })
        if actual != entry["checkpoint_sha256"]:
            failures.append(f"{entry['label']} 哈希与阶段1清单不符")
        if entry["alpha_a19_share"] not in a20.ALPHAS:
            failures.append(f"{entry['label']} alpha 不在冻结三点内")
    if [c["label"] for c in candidates] != list(a20.CANDIDATE_NAMES):
        failures.append("候选清单与冻结标签不一致")
    if sorted(c["alpha"] for c in candidates) != list(a20.ALPHAS):
        failures.append("alpha 集合与冻结三点不一致")

    assembly = quant.load_overall_assembly()
    split = {
        "train_total": int(assembly.counts["train_total"]),
        "development_total": int(assembly.counts["development_total"]),
        "train_ids_sha256": str(assembly.split_digest["train_ids"]),
        "development_ids_sha256": str(assembly.split_digest["development_all"]),
    }
    for key, expected in EXPECTED.items():
        if split[key] != expected:
            failures.append(f"划分 {key} 不符: {split[key]} != {expected}")

    resolved = args.output_root.resolve()
    in_a20 = "role_a_a20_interp" in resolved.parts
    if not in_a20:
        failures.append(f"输出目录不属于 A20: {resolved}")
    if resolved == Path(a20.PROJECT / "runs"):
        failures.append("输出目录过宽")

    report = {
        "status": "ok" if not failures else "gate_failed",
        "candidate_sha_gate": candidates,
        "frozen_source_hashes": source_hashes,
        "split": split,
        "output_root": resolved.as_posix(),
        "output_root_is_a20_only": bool(in_a20),
        "alphas": list(a20.ALPHAS),
        "failures": failures,
        "wrote_artifacts": False,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        print("[A20-STAGE2-GATE-FAIL] " + "; ".join(failures))
        return 2
    print("[A20-STAGE2-GATE-OK] 候选哈希、冻结划分、A20 输出目录全部通过；未写产物")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
