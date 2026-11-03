"""auto07 准备（一次性；证据，未来不得重跑覆盖）。

从**已验收 auto05** 复制 6 个运行时 py、`QA/test_qat_gates.py` 与 `execution/cpu04/run_cpu04.py`，
只做新 run（auto07）路径/schema/identity 命名适配，并**唯一**改变训练学习率：

* ``ARM_LR`` 两个值 **1e-5 → 2e-6**；``TrainConfig``/``AutoConfig`` 默认 ``learning_rate``
  **1e-5 → 2e-6**（降低 5 倍）；
* 其余全部保留：arms ``float8``/``qat8``、seed **20261015**、**8 epochs**、
  batch8、1 warmup + 7 cosine、WD 1e-4、EMA 0.999、KD 0.3 / T=2、原同初始化/教师/
  1600 软标签/32 人 roster/冻结量化尺度/256/15Conv 与数据/增强/训练/恢复/STOP/native 算法；
* 不复制任何旧 arms/permits/输出/权重，不执行任何旧 builder/prepare/apply 准备器。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

PROJECT = Path(r"E:/嵌赛/pld-medical")
A5 = PROJECT / "runs/role_a_camus/20261013_auto05"
A7 = PROJECT / "runs/role_a_camus/20261014_auto07"
SOURCES = ("train_distill_core.py", "distill_links.py", "softtargets.py", "qat_local.py",
           "qat_runtime.py", "run_distill.py")
QA = "test_qat_gates.py"
CPU = "run_cpu04.py"

OLD_PATHS = ("runs/role_a_camus/20261013_auto05", "20261013_auto05", "camus-auto05-")
NEW_PATHS = ("runs/role_a_camus/20261014_auto07", "20261014_auto07", "camus-auto07-")
LEARNING_RATE_PAIRS = (
    ('ARM_LR = {"float8": 1e-5, "qat8": 1e-5}', 'ARM_LR = {"float8": 2e-6, "qat8": 2e-6}'),
    ("learning_rate: float = 1e-5", "learning_rate: float = 2e-6"),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapt(text: str) -> tuple[str, list[str]]:
    notes: list[str] = []

    def record(note: str) -> None:
        if note not in notes:
            notes.append(note)

    for old, new in zip(OLD_PATHS, NEW_PATHS):
        if old in text:
            text = text.replace(old, new)
            record(f"namespace {old} -> {new}")
    for old, new in LEARNING_RATE_PAIRS:
        if old in text:
            text = text.replace(old, new)
            record(f"learning rate {old.split('=')[1].strip() if '=' in old else old} -> "
                   f"{new.split('=')[1].strip() if '=' in new else new}")
    return text, notes


def main() -> int:
    if A7.exists():
        leftovers = [q for q in A7.rglob("*") if not (q.is_dir() and not any(q.iterdir()))]
        for stale in [q for q in A7.rglob("*") if q.is_dir() and not any(q.iterdir())]:
            stale.rmdir()   # 仅清理本次失败留下的空目录
        if leftovers:
            raise RuntimeError(f"auto07 已存在且非空: {[str(q) for q in leftovers][:5]}")
        else:
            A7.rmdir()   # 仅清理本次失败留下的空目录，重新证明不存在

    (A7 / "sources").mkdir(parents=True)
    (A7 / "QA").mkdir(parents=True)
    (A7 / "execution/cpu01").mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for name in SOURCES:
        source = A5 / "sources" / name
        text, notes = adapt(source.read_text(encoding="utf-8"))
        target = A7 / "sources" / name
        target.write_text(text, encoding="utf-8")
        records.append({"kind": "runtime_source", "file": f"sources/{name}",
                        "old_path": source.as_posix(), "old_sha256": sha256_file(source),
                        "new_sha256": sha256_file(target), "modifications": notes})
    qa_source = A5 / "QA" / QA
    qa_text, qa_notes = adapt(qa_source.read_text(encoding="utf-8"))
    (A7 / "QA" / QA).write_text(qa_text, encoding="utf-8")
    records.append({"kind": "qa", "file": f"QA/{QA}", "old_path": qa_source.as_posix(),
                    "old_sha256": sha256_file(qa_source),
                    "new_sha256": sha256_file(A7 / "QA" / QA), "modifications": qa_notes})
    cpu_source = A5 / "execution/cpu04" / CPU
    cpu_text, cpu_notes = adapt(cpu_source.read_text(encoding="utf-8"))
    cpu_text = cpu_text.replace("cpu04", "cpu01").replace("CPU04", "CPU01")
    cpu_notes.append("execution task directory / script name cpu04 -> cpu01")
    target_cpu = A7 / "execution/cpu01" / "run_cpu01.py"
    target_cpu.write_text(cpu_text, encoding="utf-8")
    records.append({"kind": "execution", "file": "execution/cpu01/run_cpu01.py",
                    "old_path": cpu_source.as_posix(), "old_sha256": sha256_file(cpu_source),
                    "new_sha256": sha256_file(target_cpu), "modifications": cpu_notes})
    core = (A7 / "sources/train_distill_core.py").read_text(encoding="utf-8")
    for needle in ('ARM_LR = {"float8": 2e-6, "qat8": 2e-6}', "learning_rate: float = 2e-6",
                   'ARMS = ("float8", "qat8")', "TOTAL_EPOCHS = 8", "seed: int = 20261015"):
        if needle not in core:
            raise RuntimeError(f"core 适配缺少: {needle}")
    if "1e-5" in core:
        raise RuntimeError("仍存在 1e-5 学习率残留")
    cpu_text_final = target_cpu.read_text(encoding="utf-8")
    if "20261013_auto05" in cpu_text_final or "cpu04" in cpu_text_final:
        raise RuntimeError("cpu01 wrapper 仍指向旧 run 或旧命名")
    payload = {"schema": "camus-auto07-prep-manifest-v1", "run": A7.as_posix(),
               "source_run": A5.as_posix(),
               "single_algorithm_change": "learning rate lowered 5x: 1e-5 -> 2e-6 (both arms and "
                                          "the config default)",
               "records": records,
               "not_copied": ["arms/", "permits", "outputs", "weights", "checkpoints",
                              "old builders/prepare/apply scripts"],
               "note": "evidence only; do NOT re-run in future stages"}
    (A7 / "sources/PREP_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": [r["file"] for r in records],
                      "modifications": {r["file"]: r["modifications"] for r in records}},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
