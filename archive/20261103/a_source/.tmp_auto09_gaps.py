"""auto09 缺口补齐（一次性；只补缺失产物，**不重跑**任何 builder/CPU/训练）。

补齐：

* ``sources/PROTOCOL.json``（复用 R7 协议结构；历史对照 = R7 qat8 KL=0.3；本次实际拟训练仅
  ``qat8`` KL=0；``PER_ARM_SECONDS=1200``；固定损失 hard CE + 3fg Dice + 0·T²·KL，**KL 计算路径
  仍照原实现执行**，不优化）；
* ``README.md``；
* ``owner/PREPARATION.json`` 与 ``sources/SOURCE_PREP_MANIFEST.json``（逐文件原/新 SHA 与差异）；
* 两个必要的 thin 执行器：``execution/cpu01/run_cpu01.py``（CPU17，180+120、CPU2、fresh CPU 目录）与
  ``execution/qat8_dispatch01/run_qat8_dispatch01.py``（namespace 适配、``MAX_SECONDS 5000→1200``、
  日志 ``kl_weight`` 由 .3 改 0）。

实际 config 由 ``C.asdict(cfg)`` 生成，source/data 身份由真实元数据生成。
不签许可、不创建 ``arms/qat8``、不创建/清除 STOP。
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(r"E:/嵌赛/pld-medical")
A7 = PROJECT / "runs/role_a_camus/20261014_auto07"
A9 = PROJECT / "runs/role_a_camus/20261014_auto09"
SOURCES = A9 / "sources"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def function_digests(text: str) -> dict[str, str]:
    tree = ast.parse(text)
    return {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


NAPERS = (("runs/role_a_camus/20261014_auto07", "runs/role_a_camus/20261014_auto09"),
          ("20261014_auto07", "20261014_auto09"),
          ("camus-auto07-", "camus-auto09-"),
          ("auto07/", "auto09/"))


def adapt(text: str, *, budget: bool = False) -> tuple[str, list[str]]:
    notes: list[str] = []
    for old, new in NAPERS:
        if old in text:
            text = text.replace(old, new)
            notes.append(f"namespace {old} -> {new}")
    if budget and "MAX_SECONDS = 5000" in text:
        text = text.replace("MAX_SECONDS = 5000", "MAX_SECONDS = 1200")
        notes.append("budget MAX_SECONDS 5000 -> 1200")
    if "kl_weight=.3" in text:
        text = text.replace("kl_weight=.3", "kl_weight=0.0")
        notes.append("logged fixed config kl_weight .3 -> 0.0")
    if '"kl_weight": 0.3' in text:
        text = text.replace('"kl_weight": 0.3', '"kl_weight": 0.0')
        notes.append("logged fixed config kl_weight .3 -> 0.0")
    return text, notes


def main() -> int:
    for path in (str(SOURCES),):
        if path not in sys.path:
            sys.path.insert(0, path)
    import train_distill_core as C                                # noqa: PLC0415
    from dataclasses import asdict

    arms = {}
    for arm in C.ARMS:
        config = C.arm_config(arm, out_dir=A9 / "arms" / arm, stop_file=A9 / "STOP_REQUEST")
        arms[arm] = {"config": asdict(config), "config_digest": config.digest()}
    real = {arm: data for arm, data in arms.items()}
    qat = arms["qat8"]
    if abs(float(qat["config"]["kl_weight"]) - 0.0) > 0.0:
        raise RuntimeError("qat8 的 KL 权重必须为 0.0")
    if abs(float(qat["config"]["learning_rate"]) - 2e-6) > 0.0:
        raise RuntimeError("qat8 学习率必须为 2e-6")
    if int(qat["config"]["epochs"]) != 8:
        raise RuntimeError("epochs 必须为 8")
    data_identity = None
    try:
        data_identity = C.data_identity()
    except Exception as error:  # noqa: BLE001 - 只记录，不掩盖
        data_identity = {"error": f"{type(error).__name__}: {error}"}
    source_identity = None
    try:
        source_identity = C.full_identity()
    except Exception as error:  # noqa: BLE001
        source_identity = {"error": f"{type(error).__name__}: {error}"}
    protocol = json.loads((A7 / "sources/PROTOCOL.json").read_text(encoding="utf-8"))
    protocol.update({
        "schema": "camus-auto09-distill-protocol-v1", "utc": utc_now(), "run": A9.as_posix(),
        "status": "prepared_not_executed",
        "source_run": A7.as_posix(),
        "single_new_training_variable": "KD (KL) weight 0.3 -> 0.0",
        "historical_companion": "R7 qat8 trained with KL=0.3 (independently accepted); "
                               "R7 float8 is NOT part of this comparison",
        "arms_declared": list(C.ARMS),
        "arms_actually_released_for_training": ["qat8"],
        "inherited_for_synthetic_checks_only": ["float8"],
        "actual_config_from_code": real,
        "fixed": {"loss": "hard CE + 3fg Dice + 0.0 * T^2 * KL (the original KL computation path is "
                          "still executed unchanged; only the weight is 0)",
                  "kl_implementation_optimised": False,
                  "epochs": C.TOTAL_EPOCHS, "seed": 20261015, "batch_size": 8,
                  "learning_rate": 2e-6, "schedule": "1 warmup + 7 cosine", "weight_decay": 1e-4,
                  "ema_decay": C.EMA_DECAY, "temperature": C.TEMPERATURE,
                  "fake_quant_scales": "R3 original minmax 15-layer scales (spec 62f3a776...); "
                                       "R8 histogram head scale is NOT adopted",
                  "precision": "IEEE FP32 training; native dev temporarily TF32=true then restored",
                  "init": "auto03 best7 online SHA 3d8f2689...42378",
                  "topology": "15 Conv / 3 concat / shared 4-class head"},
        "budget": {"max_seconds_per_arm": C.PER_ARM_SECONDS, "grace_seconds": 120,
                   "run_max_bytes": 3_000_000_000, "project_max_bytes": 40_000_000_000},
        "stop_protection": {"root_stop": (A9 / "STOP_REQUEST").as_posix(),
                            "policy": "if absent, continue directly and record evidence; if present, "
                                      "refuse. This preparation creates/clears no STOP."},
        "final_comparison": "pre-declared final online epoch8; best/EMA recorded only, never used to "
                            "pick the comparison point",
        "executed": {"cpu_test_run": False, "model_forward": False, "calibration_run": False,
                     "export_run": False, "training_run": False, "development_evaluation": False,
                     "tests_run": False, "deployment_run": False, "permit_created": False},
        "cumulative_accepted_training_epochs": 364,
        "source_identity": source_identity, "data_identity": data_identity,
    })
    write_json(SOURCES / "PROTOCOL.json", protocol)

    # thin 执行器
    executor_records: list[dict[str, Any]] = []
    cpu_src = A7 / "execution/cpu01/run_cpu01.py"
    cpu_text, cpu_notes = adapt(cpu_src.read_text(encoding="utf-8"))
    cpu_text = cpu_text.replace("MAX_SECONDS = 300", "MAX_SECONDS = 180")
    cpu_notes.append("budget MAX_SECONDS 300 -> 180 (180 + 120 grace)")
    (A9 / "execution/cpu01").mkdir(parents=True, exist_ok=True)
    cpu_target = A9 / "execution/cpu01/run_cpu01.py"
    cpu_target.write_text(cpu_text, encoding="utf-8")
    executor_records.append({"file": "execution/cpu01/run_cpu01.py", "kind": "cpu17_executor",
                             "old_path": cpu_src.as_posix(), "old_sha256": sha256(cpu_src),
                             "new_path": cpu_target.as_posix(), "new_sha256": sha256(cpu_target),
                             "modifications": cpu_notes})
    qat_src = A7 / "execution/qat8_dispatch01/run_qat8_dispatch01.py"
    qat_text, qat_notes = adapt(qat_src.read_text(encoding="utf-8"), budget=True)
    (A9 / "execution/qat8_dispatch01").mkdir(parents=True, exist_ok=True)
    qat_target = A9 / "execution/qat8_dispatch01/run_qat8_dispatch01.py"
    qat_target.write_text(qat_text, encoding="utf-8")
    executor_records.append({"file": "execution/qat8_dispatch01/run_qat8_dispatch01.py",
                             "kind": "qat8_executor", "old_path": qat_src.as_posix(),
                             "old_sha256": sha256(qat_src), "new_path": qat_target.as_posix(),
                             "new_sha256": sha256(qat_target), "modifications": qat_notes})
    ast.parse(cpu_text), ast.parse(qat_text)

    # 源差异（AST）
    source_records: list[dict[str, Any]] = []
    for name in ("train_distill_core.py", "distill_links.py", "softtargets.py", "qat_local.py",
                 "qat_runtime.py", "run_distill.py"):
        old_text = (A7 / "sources" / name).read_text(encoding="utf-8")
        new_text = (SOURCES / name).read_text(encoding="utf-8")
        old_f, new_f = function_digests(old_text), function_digests(new_text)
        differing = sorted(k for k in set(old_f) & set(new_f) if old_f[k] != new_f[k])
        source_records.append({"file": name, "old_sha256": sha256(A7 / "sources" / name),
                               "new_sha256": sha256(SOURCES / name),
                               "identical": old_text == new_text,
                               "differing_functions": differing,
                               "changed_non_function_lines": [
                                   line.strip() for line in new_text.splitlines()
                                   if line.strip() in ("KL_WEIGHT = 0.0", "PER_ARM_SECONDS = 1200")
                                   or "20261014_auto09" in line or "camus-auto09-" in line]})
    qa_old = (A7 / "QA/test_qat_gates.py").read_text(encoding="utf-8")
    qa_new = (A9 / "QA/test_qat_gates.py").read_text(encoding="utf-8")
    qa_old_f, qa_new_f = function_digests(qa_old), function_digests(qa_new)
    qa_diff = sorted(k for k in set(qa_old_f) & set(qa_new_f) if qa_old_f[k] != qa_new_f[k])
    manifest = {"schema": "camus-auto09-source-prep-manifest-v1", "utc": utc_now(),
                "run": A9.as_posix(), "source_run": A7.as_posix(),
                "sources": source_records, "executors": executor_records,
                "qa": {"old_sha256": sha256(A7 / "QA/test_qat_gates.py"),
                       "new_sha256": sha256(A9 / "QA/test_qat_gates.py"),
                       "differing_functions": qa_diff,
                       "expected_value_fix": "line 298 expected kl_weight 0.3 -> 0.0 only; "
                                             "the other 17 checks keep their original meaning",
                       "other_0_3_occurrences": [line.strip() for line in qa_new.splitlines()
                                                 if "0.3" in line]},
                "not_rerun": [".tmp_auto09_build.py", "any earlier builder", "any CPU or training"],
                "created_no_permit": True, "created_arms_dir": False,
                "executed": {"cpu_test_run": False, "training_run": False, "model_forward": False,
                             "development_evaluation": False, "permit_created": False}}
    write_json(SOURCES / "SOURCE_PREP_MANIFEST.json", manifest)
    preparation = {"schema": "camus-auto09-preparation-v1", "utc": utc_now(), "run": A9.as_posix(),
                   "status": "prepared_not_executed_pending_owner_review",
                   "actual_config_from_code": real,
                   "source_identity": source_identity, "data_identity": data_identity,
                   "sources": source_records, "executors": executor_records,
                   "qa_sha256": sha256(A9 / "QA/test_qat_gates.py"),
                   "protocol_sha256": sha256(SOURCES / "PROTOCOL.json"),
                   "manifest_sha256": sha256(SOURCES / "SOURCE_PREP_MANIFEST.json"),
                   "stages": ["preparation (done)", "owner review", "CPU17 synthetic gate",
                              "training permit", "qat8 8-epoch training",
                              "final online8 FP32 evaluation", "INT8 export/verify",
                              "INT8 dev200"],
                   "open_gates": ["no approved permit exists", "no arms/qat8 created",
                                  "STOP not created/cleared by this preparation",
                                  "fresh CPU directory required before CPU17",
                                  "baseline epoch0 200-frame evidence to be re-verified by owner"],
                   "executed": {"cpu_test_run": False, "training_run": False, "model_forward": False,
                                "calibration_run": False, "export_run": False,
                                "development_evaluation": False, "tests_run": False,
                                "deployment_run": False, "permit_created": False},
                   "cumulative_accepted_training_epochs": 364}
    write_json(A9 / "owner/PREPARATION.json", preparation)
    readme = f"""# auto09：单因素 QAT 训练（KD 权重 0）——**仅准备，未执行**

**唯一新训练变量**：KD（KL）权重 **0.3 → 0.0**。其余与已验收 R7 qat8 完全一致：auto03 best7 online
初始化（SHA `3d8f268910783ea524513b3506f4425aec991a54dcb23bf25cff5598fb942378`）、seed 20261015、
8 epochs、batch8、lr 2e-6、1 warmup + 7 cosine、WD 1e-4、EMA 0.999、T=2、**R3 原 minmax 15 层
fakequant 尺度**（**不采用 R8 直方图头尺度**）、数据/增强/sampler 同 R7、IEEE FP32 训练、
native 评估 TF32 策略同 R7、15 Conv / 3 concat / 共享 4 类头。

**历史对照**：R7 **qat8（KL=0.3）** 为同条件历史臂；R7 float8 不参与本对照。

**损失**：hard CE + 3fg Dice + **0 · T²·KL**；**原 KL 计算路径仍按原实现执行**（不做零权重短路优化）。

**数据**：原 400 训练 / 1600 帧、50 dev / 200 native；原 32 训练患者 / 128 帧校准；
teacher 与 1600 软目标仍为 auto03 冻结只读来源；不复制数据/软目标/大模型。

**比较点**：预声明 **final online epoch8**；best/EMA 仅记录，不用来选点。

**阶段**：准备（已完成）→ owner 审核 → CPU17 合成门禁 → 训练许可 → qat8 8 轮训练 → final online8
FP32 评估 → INT8 导出/verify → INT8 dev200。**当前全部 `executed=false`，无批准许可。**

**预算**：单次 **1200 s + 120 s**；R9 合计 ≤3 GB / 项目 ≤40 GB；STOP 守护（本准备不创建/不清除 STOP）。

**预算与许可**：拟放行实际训练 **仅 qat8 一个 arm**；继承的 float8 配置仅用于合成检查。
"""
    (A9 / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"protocol": sha256(SOURCES / "PROTOCOL.json")[:16],
                      "manifest": sha256(SOURCES / "SOURCE_PREP_MANIFEST.json")[:16],
                      "preparation": sha256(A9 / "owner/PREPARATION.json")[:16],
                      "cpu_executor": sha256(A9 / "execution/cpu01/run_cpu01.py")[:16],
                      "qat_executor": sha256(A9 / "execution/qat8_dispatch01/run_qat8_dispatch01.py")[:16],
                      "qa": sha256(A9 / "QA/test_qat_gates.py")[:16],
                      "qat8_digest": qat["config_digest"][:16]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
