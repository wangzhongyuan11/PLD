"""auto07 最终 8 轮评估文件准备（**仅文件**；不运行模型/评估/导出/freeze）。

来源：已验收 auto05 两臂各自的 ``arms/<arm>/evaluation_sources`` 7 个 py 与该臂
``INT8_PROTOCOL.json``（**不使用旧 prepare_final8_evaluation.py 或 auto03 原稿**，因旧 builder 未含
最终 score 修复）。

适配：仅 namespace ``20261013_auto05 → 20261014_auto07`` 与 ``auto05/AUTO05 → auto07/AUTO07``；
算法函数、最终 online8（9 条 history）、固定 final 分数（非 best）校验与整数量值方法保持不变。

另在 R7 ``owner/`` 复制尚不存在的 6 个已验收 owner 脚本（仅同 namespace 适配，不运行）；
不复制关闭/总结工具、不改 R7 现有 owner 文件。

写 ``owner/EVALUATION_PREPARATION.json`` 记录每个原/新 SHA 与修改说明，并做 AST 对照
（28 个 integer pipeline 函数 + 原 13 项接口测试保留 + 最终 raw metric 仍 ``history[-1].online``）。
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any

PROJECT = Path(r"E:/嵌赛/pld-medical")
A5 = PROJECT / "runs/role_a_camus/20261013_auto05"
A7 = PROJECT / "runs/role_a_camus/20261014_auto07"
ARMS = ("float8", "qat8")
EVAL_FILES = ("quant_student.py", "int8_candidate_pipeline.py", "eval_student.py",
              "eval_candidate_int8.py", "area_report.py", "int8_protocol.py", "selection_gate.py")
OWNER_FILES = ("freeze_final8.py", "validate_final8_adapters02.py", "admit_final8_fp32_02.py",
               "accept_final8_fp32_and_admit_export.py", "admit_final8_int8_dev.py",
               "accept_final8_int8.py")
PIPELINE = "int8_candidate_pipeline.py"
NUMERIC_FUNCTIONS = ("calibrate", "scales_from_calibration", "build_layers", "export_graph",
                     "verify_export", "load_export", "_oracle_tile", "_rescale_with_multiplier",
                     "forward_graph", "native_labels_from_logits", "_safe_dir", "_write_array",
                     "_layer_dir_name", "_safe_member_path", "_read_array", "_read_layers")
REPLACEMENTS = (("runs/role_a_camus/20261013_auto05", "runs/role_a_camus/20261014_auto07"),
                ("20261013_auto05", "20261014_auto07"),
                ("camus-auto05-", "camus-auto07-"),
                ("auto05/", "auto07/"),
                ("AUTO05", "AUTO07"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function_digests(path: Path) -> dict[str, str]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def adapt(text: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    for old, new in REPLACEMENTS:
        if old in text:
            text = text.replace(old, new)
            notes.append(f"namespace {old} -> {new}")
    return text, notes


def main() -> int:
    for arm in ARMS:
        target_sources = A7 / f"arms/{arm}/evaluation_sources"
        target_protocol = A7 / f"arms/{arm}/INT8_PROTOCOL.json"
        if target_sources.exists() or target_protocol.exists():
            raise RuntimeError(f"目标已存在，拒绝覆盖: {target_sources} / {target_protocol}")
    records: list[dict[str, Any]] = []
    for arm in ARMS:
        (A7 / f"arms/{arm}/evaluation_sources").mkdir(parents=True, exist_ok=True)
        for name in EVAL_FILES:
            source = A5 / f"arms/{arm}/evaluation_sources/{name}"
            if not source.is_file():
                raise RuntimeError(f"缺少已验收来源: {source}")
            text, notes = adapt(source.read_text(encoding="utf-8"))
            target = A7 / f"arms/{arm}/evaluation_sources/{name}"
            target.write_text(text, encoding="utf-8")
            records.append({"kind": "evaluation_source", "arm": arm, "file": name,
                            "old_path": source.as_posix(), "old_sha256": sha256_file(source),
                            "new_path": target.as_posix(), "new_sha256": sha256_file(target),
                            "modifications": notes or ["byte-identical"]})
        protocol_source = A5 / f"arms/{arm}/INT8_PROTOCOL.json"
        text, notes = adapt(protocol_source.read_text(encoding="utf-8"))
        (A7 / f"arms/{arm}/INT8_PROTOCOL.json").write_text(text, encoding="utf-8")
        records.append({"kind": "int8_protocol", "arm": arm, "file": "INT8_PROTOCOL.json",
                        "old_path": protocol_source.as_posix(),
                        "old_sha256": sha256_file(protocol_source),
                        "new_path": (A7 / f"arms/{arm}/INT8_PROTOCOL.json").as_posix(),
                        "new_sha256": sha256_file(A7 / f"arms/{arm}/INT8_PROTOCOL.json"),
                        "modifications": notes or ["byte-identical"]})
    owner_dir = A7 / "owner"
    owner_notes: list[dict[str, Any]] = []
    for name in OWNER_FILES:
        target = owner_dir / name
        if target.exists():
            owner_notes.append({"file": name, "skipped": "already exists in R7 owner"})
            continue
        source = A5 / "owner" / name
        if not source.is_file():
            owner_notes.append({"file": name, "skipped": "source not present"})
            continue
        text, notes = adapt(source.read_text(encoding="utf-8"))
        target.write_text(text, encoding="utf-8")
        owner_notes.append({"file": name, "old_path": source.as_posix(),
                            "old_sha256": sha256_file(source),
                            "new_sha256": sha256_file(target),
                            "modifications": notes or ["byte-identical"]})
    # AST 对照
    ast_report: dict[str, Any] = {}
    for arm in ARMS:
        target = A7 / f"arms/{arm}/evaluation_sources/{PIPELINE}"
        source = A5 / f"arms/{arm}/evaluation_sources/{PIPELINE}"
        target_digests, source_digests = function_digests(target), function_digests(source)
        shared = sorted(set(target_digests) & set(source_digests))
        differing = [name for name in shared if target_digests[name] != source_digests[name]]
        numeric_equal = all(target_digests[name] == source_digests[name]
                            for name in NUMERIC_FUNCTIONS if name in shared)
        ast_report[arm] = {"pipeline": PIPELINE, "shared_functions": len(shared),
                           "numeric_core_functions_equal": numeric_equal,
                           "differing_functions": differing}
    final_score_check: dict[str, Any] = {}
    for arm in ARMS:
        text = (A7 / f"arms/{arm}/evaluation_sources/{PIPELINE}").read_text(encoding="utf-8")
        final_score_check[arm] = {"history_last_online_marker": "history[-1]" in text,
                                  "checks_present": [token for token in
                                                     ("history[-1]", "online", "best_metric")
                                                     if token in text]}
    interface_tests: dict[str, Any] = {}
    for arm in ARMS:
        source = A5 / f"arms/{arm}/evaluation_sources/{PIPELINE}"
        target = A7 / f"arms/{arm}/evaluation_sources/{PIPELINE}"
        interface_tests[arm] = {"source_sha256": sha256_file(source),
                                "target_sha256": sha256_file(target)}
    payload = {"schema": "camus-auto07-evaluation-preparation-v1",
               "utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
               "run": A7.as_posix(), "source_run": A5.as_posix(),
               "not_used": ["old prepare_final8_evaluation.py", "auto03 original drafts"],
               "records": records, "owner_records": owner_notes,
               "ast_report": ast_report, "final_score_check": final_score_check,
               "pipeline_hash_pairs": interface_tests,
               "preserved": ["28 integer pipeline numeric functions (AST equal)",
                             "13 original interface tests", "final raw metric from history[-1].online",
                             "32-patient roster", "auto03 teacher/targets", "cache/area thresholds",
                             "numeric contract (signed127 / per-out-channel / I32 bias / Q24 ties-even "
                             "/ skip-first concat / shared 4-class head)"],
               "executed": {"model_run": False, "evaluation_run": False, "export_run": False,
                            "freeze_run": False, "permit_created": False, "tests_run": False},
               "note": "file preparation only; awaiting owner freeze and separate evaluation permit"}
    out = owner_dir / "EVALUATION_PREPARATION.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "owner": len(owner_notes),
                      "ast": ast_report, "final_score": final_score_check,
                      "written": out.as_posix()}, ensure_ascii=False)[:900], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
