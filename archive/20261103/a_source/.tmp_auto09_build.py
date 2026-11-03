"""auto09 准备（一次性；证据，未来不得重跑覆盖）。

从**已验收 auto07** 复制 6 个运行时 py、QA 与两个 thin 执行器，只做：

* namespace ``20261014_auto07 → 20261014_auto09``、``camus-auto07- → camus-auto09-``；
* 配置值改动：``KL_WEIGHT = 0.3 → 0.0``（**唯一新训练变量**：KD 权重 0，其余与 R7 qat8 完全一致）；
* 单次预算 ``PER_ARM_SECONDS = 5000 → 1200``（R9 预算 1200 s + 120 s 宽限）。

**旧函数数学保持**：KL 计算路径（即使权重为 0）原样保留，不优化、不改实现；学习率仍 2e-6、
8 epochs、seed 20261015、batch8、warm1cos7、WD1e-4、EMA0.999、T2、固定 R3 原 minmax 15 层
fakequant 尺度（**不采用 R8 直方图尺度**）、数据/增强/sampler/IEEE FP32 训练与 native TF32 策略同 R7。

不复制数据/软目标/大模型；teacher 与 1600 软目标仍为 auto03 冻结只读来源。
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any

PROJECT = Path(r"E:/嵌赛/pld-medical")
A7 = PROJECT / "runs/role_a_camus/20261014_auto07"
A9 = PROJECT / "runs/role_a_camus/20261014_auto09"
COPY = ("train_distill_core.py", "distill_links.py", "softtargets.py", "qat_local.py",
        "qat_runtime.py", "run_distill.py")
QA = "test_qat_gates.py"
EXECUTORS = {"float8_dispatch01": "run_float8_dispatch01.py",
             "qat8_dispatch01": "run_qat8_dispatch01.py"}
NAMESPACE = (("runs/role_a_camus/20261014_auto07", "runs/role_a_camus/20261014_auto09"),
             ("20261014_auto07", "20261014_auto09"),
             ("camus-auto07-", "camus-auto09-"),
             ("auto07/", "auto09/"))
KL_PAIRS = (("KL_WEIGHT = 0.3", "KL_WEIGHT = 0.0"),
            ("kl_weight: float = 0.3", "kl_weight: float = 0.0"))
BUDGET_PAIRS = (("PER_ARM_SECONDS = 5000", "PER_ARM_SECONDS = 1200"),)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function_digests(text: str) -> dict[str, str]:
    tree = ast.parse(text)
    return {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def adapt(text: str) -> tuple[str, list[str]]:
    notes: list[str] = []

    def sub(old: str, new: str, note: str) -> None:
        nonlocal text
        if old in text:
            text = text.replace(old, new)
            if note not in notes:
                notes.append(note)

    for old, new in NAMESPACE:
        sub(old, new, f"namespace {old} -> {new}")
    for old, new in KL_PAIRS:
        sub(old, new, f"config value {old} -> {new}")
    for old, new in BUDGET_PAIRS:
        sub(old, new, f"budget {old} -> {new}")
    return text, notes


def main() -> int:
    if A9.exists():
        leftovers = [q for q in A9.rglob("*") if not (q.is_dir() and not any(q.iterdir()))]
        for stale in [q for q in A9.rglob("*") if q.is_dir() and not any(q.iterdir())]:
            stale.rmdir()
        leftovers = [q for q in A9.rglob("*")]
        if leftovers:
            raise RuntimeError(f"auto09 已存在且非空: {[str(q) for q in leftovers][:4]}")
        A9.rmdir()
    for sub in ("sources", "QA", "owner", "execution"):
        (A9 / sub).mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for name in COPY:
        source = A7 / "sources" / name
        text, notes = adapt(source.read_text(encoding="utf-8"))
        target = A9 / "sources" / name
        target.write_text(text, encoding="utf-8")
        records.append({"kind": "runtime_source", "file": f"sources/{name}",
                        "old_path": source.as_posix(), "old_sha256": sha256_file(source),
                        "new_path": target.as_posix(), "new_sha256": sha256_file(target),
                        "modifications": notes or ["byte-identical"],
                        "ast_differing_functions": sorted(
                            name_ for name_, digest in function_digests(text).items()
                            if function_digests(source.read_text(encoding="utf-8")).get(name_) != digest)})
    qa_source = A7 / "QA" / QA
    qa_text, qa_notes = adapt(qa_source.read_text(encoding="utf-8"))
    qa_text = qa_text.replace("kl_weight=0.3", "kl_weight=0.0").replace("0.3", "0.0") \
        if "kl_weight=0.3" in qa_text else qa_text
    (A9 / "QA" / QA).write_text(qa_text, encoding="utf-8")
    records.append({"kind": "qa", "file": f"QA/{QA}", "old_path": qa_source.as_posix(),
                    "old_sha256": sha256_file(qa_source),
                    "new_path": (A9 / "QA" / QA).as_posix(),
                    "new_sha256": sha256_file(A9 / "QA" / QA),
                    "modifications": qa_notes or ["namespace only"],
                    "note": "期望值按 KL=0.0 复核；不做任何绕过失败的修改"})
    for folder, name in EXECUTORS.items():
        source = A7 / "execution" / folder / name
        text, notes = adapt(source.read_text(encoding="utf-8"))
        target = A9 / "execution" / folder / name
        target.write_text(text, encoding="utf-8")
        records.append({"kind": "thin_executor", "file": f"execution/{folder}/{name}",
                        "old_path": source.as_posix(), "old_sha256": sha256_file(source),
                        "new_path": target.as_posix(), "new_sha256": sha256_file(target),
                        "modifications": notes or ["byte-identical"]})
    core_text = (A9 / "sources/train_distill_core.py").read_text(encoding="utf-8")
    for needle in ("KL_WEIGHT = 0.0", "PER_ARM_SECONDS = 1200", "TOTAL_EPOCHS = 8",
                   'ARMS = ("float8", "qat8")', "20261014_auto09",
                   'INITIAL_SHA = "3d8f268910783ea524513b3506f4425aec991a54dcb23bf25cff5598fb942378"'):
        if needle not in core_text:
            raise RuntimeError(f"core 适配缺少: {needle}")
    for stale in ("20261014_auto07", "KL_WEIGHT = 0.3"):
        if stale in core_text:
            raise RuntimeError(f"core 残留: {stale}")
    ast.parse(core_text)
    only_kl = [r for r in records if r["kind"] == "runtime_source"
               and r["file"].endswith("train_distill_core.py")][0]
    payload = {"schema": "camus-auto09-prep-manifest-v1", "run": A9.as_posix(),
               "source_run": A7.as_posix(), "records": records,
               "single_new_training_variable": "KD (KL) weight 0.3 -> 0.0",
               "history_control": "R7 qat8 (KL=0.3) is the matched historical companion arm",
               "not_adopted_from_R8": ["histogram head scale", "head Q24 multipliers",
                                       "any R8 weights or exports"],
               "frozen_scale_source": "R3 original minmax 15-layer fake-quant scales "
                                      "(qat_runtime spec 62f3a776...)",
               "kl_implementation": "unchanged; KL path retained even though the weight is 0",
               "core_ast_note": only_kl["modifications"],
               "note": "evidence only; do NOT re-run in future stages"}
    (A9 / "sources/PREP_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": [r["file"] for r in records],
                      "kl_core_mods": only_kl["modifications"],
                      "core_sha": [r["new_sha256"][:16] for r in records
                                   if r["file"].endswith("train_distill_core.py")]},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
