"""A 组最终训练的断点门禁：源码/配置/输入哈希不一致时**硬拒绝** resume。

设计要点（对应 Codex 阶段 0 审查）：
* 不是"只记录哈希"，而是恢复前重新计算并与断点内记录逐项比对，任一不符立即抛错。
* 校验面覆盖：驱动与依赖源码、配置（含 config_hash）、输入（起点权重、冻结划分摘要、
  校准 ID 摘要、训练/校准数据清单摘要）。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

__all__ = ["sha256_file", "hash_files", "build_guard", "verify_guard", "GuardMismatch"]


class GuardMismatch(RuntimeError):
    """断点门禁不通过：源码/配置/输入哈希与断点记录不一致。"""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_files(paths: Mapping[str, Path | str]) -> dict[str, str]:
    """把 {逻辑名: 路径} 映射为 {逻辑名: SHA256}；文件缺失即报错（不静默跳过）。"""
    return {name: sha256_file(path) for name, path in paths.items()}


def build_guard(*, code_paths: Mapping[str, Path | str], config_hash: str,
                inputs: Mapping[str, str]) -> dict[str, Any]:
    """构造写入断点的门禁记录。``inputs`` 为 [逻辑名 -> 摘要]（如划分摘要、起点权重哈希）。"""
    if not code_paths:
        raise ValueError("code_paths 不能为空")
    if not inputs:
        raise ValueError("inputs 不能为空")
    return {"schema": "a-final-resume-guard-v1",
            "code_hashes": hash_files(code_paths),
            "config_hash": str(config_hash),
            "input_hashes": {str(name): str(value) for name, value in inputs.items()}}


def verify_guard(record: Mapping[str, Any], *, code_paths: Mapping[str, Path | str],
                 config_hash: str, inputs: Mapping[str, str]) -> None:
    """比对当前状态与断点记录；任一不符抛 ``GuardMismatch``（硬拒绝）。"""
    if record.get("schema") != "a-final-resume-guard-v1":
        raise GuardMismatch(f"断点门禁 schema 不支持: {record.get('schema')!r}")

    current_code = hash_files(code_paths)
    recorded_code = dict(record.get("code_hashes", {}))
    missing = sorted(set(recorded_code) - set(current_code))
    extra = sorted(set(current_code) - set(recorded_code))
    if missing or extra:
        raise GuardMismatch(f"源码清单不一致: 缺失 {missing}，多出 {extra}")
    changed = sorted(name for name in current_code if current_code[name] != recorded_code[name])
    if changed:
        raise GuardMismatch("源码哈希变化: " + ", ".join(
            f"{name} {recorded_code[name][:12]}->{current_code[name][:12]}" for name in changed))

    if str(record.get("config_hash")) != str(config_hash):
        raise GuardMismatch(f"配置哈希变化: {record.get('config_hash')} != {config_hash}")

    recorded_inputs = {str(k): str(v) for k, v in dict(record.get("input_hashes", {})).items()}
    current_inputs = {str(k): str(v) for k, v in inputs.items()}
    missing = sorted(set(recorded_inputs) - set(current_inputs))
    extra = sorted(set(current_inputs) - set(recorded_inputs))
    if missing or extra:
        raise GuardMismatch(f"输入清单不一致: 缺失 {missing}，多出 {extra}")
    changed = sorted(name for name in current_inputs
                     if current_inputs[name] != recorded_inputs[name])
    if changed:
        raise GuardMismatch("输入哈希变化: " + ", ".join(
            f"{name} {recorded_inputs[name][:12]}->{current_inputs[name][:12]}" for name in changed))
