"""Independent 2-epoch repair trial (v3) starting from the original baseline.

Provenance / 与首轮试验的关系
---------------------------
* 起点是原始基线 ``runs/role_a_tinyunet/best.pt``，区别于 v1/v2 从首轮
  研究候选出发；此前候选在已暴露的 2016 开发集上有大病灶退步。
* v3 在新的 ``runs/role_a_from_baseline_v3/`` 中重新训练，保留独立证据。
* 训练集排除规则、两个开发集门槛、采样与教师蒸馏权重沿用 v2；
  损失函数直接复用 v1 的
  :func:`software.role_a.refine_preserve_large.loss_with_retention`，避免两份实现漂移。
* **2017 官方 Task 1 测试集始终隔离**：本脚本只枚举它之外的目录，从不读取、
  从不用于训练或选模型。

可恢复性（真实限制，不夸大）
---------------------------
优化器状态、轮数、采样器状态与主进程 Python/NumPy/PyTorch 随机状态都会随检查点
保存并在 ``--resume`` 时恢复；但 DataLoader 使用 ``num_workers=2`` +
``persistent_workers=True``，工作进程内部的随机增强状态无法完整保存，因此**只能从
已完成轮次继续，恢复后的随机增强不保证逐位一致**。详细说明见 :data:`RESUME_SCOPE`。

用法::

    py -m software.role_a.refine_from_baseline_v3            # 全新开始
    py -m software.role_a.refine_from_baseline_v3 --resume   # 从最后一个完整轮次继续
"""
from __future__ import annotations

import argparse
import collections.abc
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from software.role_a import refine_small as base
from software.role_a.data import enumerate_isic_pairs
from software.role_a.refine_conservative import evaluate_grid, set_blend
from software.role_a.refine_full2017 import EXCLUSIONS, OUT as FIRST_OUT
from software.role_a.refine_preserve_large import loss_with_retention

__all__ = [
    "OUT",
    "START",
    "CHECKPOINT_NAME",
    "COMPLETION_NAME",
    "CHECKPOINT_SCHEMA",
    "COMPLETION_SCHEMA",
    "EPOCHS",
    "LR",
    "BLENDS",
    "THRESHOLDS",
    "SEED",
    "RESUME_SCOPE",
    "ResumeError",
    "TrialStateError",
    "sha256_file",
    "config_digest",
    "build_config",
    "current_input_hashes",
    "atomic_write_json",
    "atomic_torch_save",
    "capture_rng_state",
    "restore_rng_state",
    "sampler_state_payload",
    "restore_sampler_state",
    "plan_mode",
    "validate_resume",
    "load_checkpoint",
    "restore_model_state",
    "run",
    "main",
]

OUT = base.PROJECT / "runs" / "role_a_from_baseline_v3"
START = base.OLD
CHECKPOINT_NAME = "checkpoint.pt"
COMPLETION_NAME = "selection.json"
CHECKPOINT_SCHEMA = "role-a-from-baseline-v3-checkpoint-v1"
COMPLETION_SCHEMA = "role-a-from-baseline-v3-completion-v1"

EPOCHS = 2
LR = 0.00001
BLENDS = (0.1, 0.25, 0.5, 1.0)
THRESHOLDS = (0.45, 0.5, 0.53, 0.56, 0.6)
SEED = base.SEED + 4
MICRO_TVERSKY_WEIGHT = 0.2
SAMPLER_SPEC = "2016 large 3, 2016 other 2, 2017 tiny 3, 2017 other 0.5"
TEACHER_SPEC = "2016 nonmicro 2.0, 2017 nonmicro 0.5"
BATCH_SIZE = 4
NUM_WORKERS = 2
PERSISTENT_WORKERS = True

RESUME_SCOPE = (
    "只能从已完成轮次继续：检查点在每个完整轮次结束后先写临时文件再原子替换，"
    "轮次中途中断时该轮从头重跑，磁盘上始终保留最后一个有效检查点。"
    "优化器、轮数、采样器状态与主进程 Python/NumPy/PyTorch 随机状态会随检查点保存并恢复；"
    "但 DataLoader 使用 num_workers=2 与 persistent_workers=True，工作进程内部的随机增强"
    "（ISICDataset 中的翻转）状态无法完整保存，因此恢复后的后续轮次与不中断的连续训练"
    "不保证逐位一致，随机增强不保证逐位一致。"
)


class ResumeError(ValueError):
    """检查点与当前配置/输入不符，或缺少可恢复状态。"""


class TrialStateError(RuntimeError):
    """试验目录状态不允许本次运行（已完成或存在未完成检查点）。"""


# ------------------------------------------------------------------ 哈希/配置
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """分块计算文件 sha256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def config_digest(config: Mapping[str, Any]) -> str:
    """配置的稳定摘要（键排序、紧凑分隔符），用于 ``--resume`` 一致性校验。"""
    payload = json.dumps(dict(config), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_config() -> dict[str, Any]:
    """本试验的完整配置；任何一项变化都会改变 :func:`config_digest`。"""
    return {
        "schema": "role-a-from-baseline-v3",
        "start_checkpoint": str(START),
        "teacher_checkpoint": str(base.OLD),
        "exclusions_file": str(EXCLUSIONS),
        "reference_trial": str(OUT.parent / "role_a_preserve_large_v2"),
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "blends": list(BLENDS),
        "thresholds": list(THRESHOLDS),
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "persistent_workers": PERSISTENT_WORKERS,
        "augment": True,
        "size": base.SIZE,
        "micro_training_area_upper": base.MICRO_TRAIN_MAX,
        "sampler": SAMPLER_SPEC,
        "teacher_bce": TEACHER_SPEC,
        "micro_tversky_weight": MICRO_TVERSKY_WEIGHT,
        "loss": "refine_preserve_large.loss_with_retention (shared with v1)",
        "acceptance": ("external 2017 validation and exposed 2016 development: tiny Dice > baseline "
                       "and medium/large/all Dice >= baseline"),
        "final_holdout": "official ISIC 2017 Task 1 test, never used for training or selection",
        "resume_scope": RESUME_SCOPE,
    }


def current_input_hashes() -> dict[str, str]:
    """训练所依赖的权重、来源记录、代码与基线产物的 sha256。"""
    return {
        "start_checkpoint": sha256_file(START),
        "teacher_checkpoint": sha256_file(base.OLD),
        "exclusions": sha256_file(EXCLUSIONS),
        "first_trial_protocol": sha256_file(FIRST_OUT / "protocol.json"),
        "development_baseline": sha256_file(FIRST_OUT / "official_test_comparison.json"),
        "training_provenance": sha256_file(base.SUPPLEMENT / "provenance.json"),
        "validation_provenance": sha256_file(base.EXTERNAL / "provenance.json"),
        "isic2016_provenance": sha256_file(base.PROJECT / "data" / "raw" / "isic2016" / "provenance.json"),
        "trial_code": sha256_file(Path(__file__)),
        "loss_code": sha256_file(base.PROJECT / "software" / "role_a" / "refine_preserve_large.py"),
        "base_code": sha256_file(base.PROJECT / "software" / "role_a" / "refine_small.py"),
        "evaluation_code": sha256_file(base.PROJECT / "software" / "role_a" / "refine_conservative.py"),
        "dataset_code": sha256_file(base.PROJECT / "software" / "role_a" / "data.py"),
        "metrics_code": sha256_file(base.PROJECT / "software" / "role_a" / "metrics.py"),
        "model_code": sha256_file(base.PROJECT / "software" / "role_a" / "model.py"),
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# -------------------------------------------------------------- 原子写入
def _fsync_best_effort(path: Path) -> None:
    try:
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    except OSError:  # 落盘尽力而为；原子性由 os.replace 保证
        pass


def _replace_with_retry(temporary: Path, target: Path, attempts: int = 5,
                        delay: float = 0.05) -> None:
    """``os.replace`` 重试：Windows 上目标文件被杀毒/索引短暂占用时仍能替换。"""
    last: OSError | None = None
    for attempt in range(max(1, attempts)):
        try:
            os.replace(temporary, target)
            return
        except OSError as error:
            last = error
            if attempt + 1 < attempts:
                time.sleep(delay * (attempt + 1))
    raise OSError(f"无法原子替换 {target}: {last}")


def _atomic_write(path: Path, writer: Callable[[Path], None], attempts: int = 5) -> None:
    """写临时文件 -> fsync -> 原子替换；失败时清理临时文件并保留旧文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.",
                                        suffix=".tmp")
    os.close(descriptor)
    temporary = Path(name)
    try:
        writer(temporary)
        _fsync_best_effort(temporary)
        _replace_with_retry(temporary, path, attempts)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    """原子写 JSON（UTF-8）。"""
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write(Path(path), lambda target: target.write_bytes(data))


def atomic_torch_save(payload: Any, path: Path,
                      serializer: Callable[[Any, Path], None] = torch.save,
                      attempts: int = 5) -> None:
    """原子写 torch 检查点（先临时文件，再替换）。"""
    _atomic_write(Path(path), lambda target: serializer(payload, target), attempts)


# ---------------------------------------------------------------- 随机状态
def _numpy_state_payload(state: tuple[Any, ...]) -> dict[str, Any]:
    name, keys, position, has_gauss, cached_gaussian = state
    return {
        "name": str(name),
        "keys": [int(value) for value in keys],
        "pos": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def capture_rng_state() -> dict[str, Any]:
    """快照主进程 Python / NumPy / PyTorch(CPU+CUDA) 随机状态（全部可 JSON 序列化）。"""
    version, internal_state, gauss_next = random.getstate()
    return {
        "python": {"version": int(version), "state": [int(v) for v in internal_state],
                   "gauss_next": gauss_next},
        "numpy": _numpy_state_payload(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().tolist(),
        "torch_cuda": ([state.tolist() for state in torch.cuda.get_rng_state_all()]
                       if torch.cuda.is_available() else []),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """恢复 :func:`capture_rng_state` 的快照（仅主进程随机源）。"""
    python_state = state["python"]
    random.setstate((int(python_state["version"]),
                     tuple(int(value) for value in python_state["state"]),
                     python_state["gauss_next"]))
    numpy_state = state["numpy"]
    np.random.set_state((str(numpy_state["name"]),
                         np.array([int(value) for value in numpy_state["keys"]], dtype=np.uint32),
                         int(numpy_state["pos"]), int(numpy_state["has_gauss"]),
                         float(numpy_state["cached_gaussian"])))
    torch.set_rng_state(torch.tensor([int(value) for value in state["torch_cpu"]],
                                     dtype=torch.uint8))
    cuda_state = state.get("torch_cuda") or []
    if cuda_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.tensor([int(v) for v in values], dtype=torch.uint8)
                                      for values in cuda_state])


def sampler_state_payload(sampler: WeightedRandomSampler) -> dict[str, Any]:
    """采样器状态。只有显式 ``generator`` 的采样器才可保存，否则拒绝。"""
    generator = getattr(sampler, "generator", None)
    if generator is None:
        raise ResumeError("采样器没有显式 generator，无法保存/恢复顺序")
    return {
        "generator": generator.get_state().tolist(),
        "num_samples": int(sampler.num_samples),
        "replacement": bool(sampler.replacement),
    }


def restore_sampler_state(sampler: WeightedRandomSampler, payload: Mapping[str, Any]) -> None:
    """恢复采样器 generator 状态；规格不符时抛 :class:`ResumeError`。"""
    if int(sampler.num_samples) != int(payload["num_samples"]) or \
            bool(sampler.replacement) != bool(payload["replacement"]):
        raise ResumeError("检查点采样器规格（num_samples/replacement）与当前配置不符")
    generator = getattr(sampler, "generator", None)
    if generator is None:
        raise ResumeError("采样器没有显式 generator，无法恢复顺序")
    generator.set_state(torch.tensor([int(value) for value in payload["generator"]],
                                     dtype=torch.uint8))


# ------------------------------------------------------------ 状态判定/校验
def plan_mode(out_dir: Path, resume: bool) -> str:
    """返回 ``fresh`` / ``resume`` / ``complete`` / ``missing-checkpoint`` / ``partial-without-resume``。"""
    out_dir = Path(out_dir)
    if (out_dir / COMPLETION_NAME).exists():
        return "complete"
    if (out_dir / CHECKPOINT_NAME).exists():
        return "resume" if resume else "partial-without-resume"
    return "missing-checkpoint" if resume else "fresh"


REQUIRED_CHECKPOINT_FIELDS = ("schema", "epoch", "phase", "config_hash", "input_hashes",
                              "model_state", "optimizer_state", "sampler_state", "rng", "history")


def validate_resume(checkpoint: Mapping[str, Any], *, config_hash: str,
                    input_hashes: Mapping[str, str], epochs: int = EPOCHS) -> tuple[int, str]:
    """校验检查点与当前配置/输入一致，返回 ``(epoch, phase)``。

    任何不一致都抛 :class:`ResumeError`，绝不"尽力而为"地继续训练。
    """
    if not isinstance(checkpoint, collections.abc.Mapping):
        raise ResumeError("检查点不是字典结构")
    missing = [name for name in REQUIRED_CHECKPOINT_FIELDS if name not in checkpoint]
    if missing:
        raise ResumeError(f"检查点缺少字段: {missing}")
    if checkpoint["schema"] != CHECKPOINT_SCHEMA:
        raise ResumeError(f"检查点 schema 不符: {checkpoint['schema']!r}")
    if checkpoint["config_hash"] != config_hash:
        raise ResumeError("检查点配置哈希与当前脚本配置不符（超参数/路径已变化）")
    stored_inputs = checkpoint["input_hashes"]
    if not isinstance(stored_inputs, collections.abc.Mapping):
        raise ResumeError("检查点 input_hashes 不是字典")
    for name, digest in input_hashes.items():
        if stored_inputs.get(name) != digest:
            raise ResumeError(f"输入 {name} 的哈希与当前文件不符（权重/排除清单/基线已变化）")
    epoch = checkpoint["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= int(epochs):
        raise ResumeError(f"检查点轮数 {epoch!r} 不在 1..{epochs} 内")
    phase = checkpoint["phase"]
    if phase not in ("trained", "evaluated"):
        raise ResumeError(f"检查点阶段 {phase!r} 无法识别")
    return int(epoch), str(phase)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """``weights_only=True`` 加载检查点（只允许张量与基本类型）。"""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ResumeError(f"检查点结构异常: {path}")
    return checkpoint


def restore_model_state(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """严格恢复权重；结构不符时抛 :class:`ResumeError`。"""
    try:
        model.load_state_dict(dict(state), strict=True)
    except RuntimeError as error:
        raise ResumeError(f"检查点权重与当前模型结构不匹配: {error}") from error


def _append_json_log(path: Path, entry: Mapping[str, Any]) -> None:
    path = Path(path)
    entries: list[Any] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                entries = existing
        except json.JSONDecodeError:
            entries = []
    entries.append(dict(entry))
    atomic_write_json(path, entries)


# ------------------------------------------------------------------ 训练主体
def _clean_items(items: list[Any], excluded: set[str]) -> list[Any]:
    return [item for item in items if item.stem not in excluded]


def run(resume: bool = False) -> int:
    """执行（或继续）2 轮修复试验。返回进程退出码。"""
    mode = plan_mode(OUT, resume)
    if mode == "complete":
        if resume:
            print(f"[v3] 试验已完成（{OUT / COMPLETION_NAME} 存在）；--resume 不重复训练，"
                  "也不会覆盖既有产物。")
            return 0
        raise TrialStateError(
            f"{OUT} 已存在完成标记 {COMPLETION_NAME}；拒绝静默覆盖已完成试验。"
            "如需重跑，请先把该目录改名或另建输出目录。")
    if mode == "missing-checkpoint":
        raise ResumeError(f"--resume 需要 {OUT / CHECKPOINT_NAME}，但该文件不存在。")
    if mode == "partial-without-resume":
        raise TrialStateError(
            f"{OUT} 存在未完成的检查点；请显式使用 --resume 继续，脚本不会静默覆盖。")

    config = build_config()
    config_hash = config_digest(config)
    hashes = current_input_hashes()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "fresh":
        base.seed_everything(SEED)

    # ---- 沿用已审查的排除规则与数据装配 ----
    exclusions = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))
    external_excluded = set(exclusions["external_validation"])
    train2016_excluded = set(exclusions["isic2016_training"])
    supplement_excluded = set(exclusions["supplemental_training"])
    test_excluded = set(exclusions["official_test_clean_subset_exclusions"])
    external_items, external_areas = base.external_items()
    external_items = _clean_items(external_items, external_excluded)
    external_areas = {stem: area for stem, area in external_areas.items()
                      if stem not in external_excluded}
    external_cuts = base.cuts(external_areas)
    test_items = enumerate_isic_pairs(base.TEST / "images", base.TEST / "masks", require_mask=True)
    if len(test_items) != 379:
        raise ValueError("ISIC 2016 development set changed")
    test_items = _clean_items(test_items, test_excluded)
    test_areas = base.area_map(test_items)
    test_cuts = base.cuts(test_areas)
    first_protocol = json.loads((FIRST_OUT / "protocol.json").read_text(encoding="utf-8"))
    external_baseline = first_protocol["baseline"]
    development_baseline = json.loads((FIRST_OUT / "official_test_comparison.json")
                                      .read_text(encoding="utf-8"))["clean_subset_summaries"]["original"]
    original_items = enumerate_isic_pairs(base.TRAIN / "images", base.TRAIN / "masks", require_mask=True)
    supplement_items = enumerate_isic_pairs(base.SUPPLEMENT / "images", base.SUPPLEMENT / "masks",
                                            require_mask=True)
    if len(original_items) != 900 or len(supplement_items) != 949:
        raise ValueError("source training datasets changed")
    original_items = _clean_items(original_items, train2016_excluded)
    supplement_items = _clean_items(supplement_items, supplement_excluded)
    if len(original_items) != 897 or len(supplement_items) != 946:
        raise ValueError("reviewed overlap exclusions changed")
    original_ids = {item.stem for item in original_items}
    train_items = original_items + supplement_items
    train_areas = base.area_map(train_items)
    weights = []
    for item in train_items:
        area = train_areas[item.stem]
        if item.stem in original_ids:
            weights.append(3.0 if area > 0.10 else 2.0)
        else:
            weights.append(3.0 if area <= 0.02 else 0.5)

    generator = torch.Generator().manual_seed(SEED)
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(train_items),
                                    replacement=True, generator=generator)
    batches = base.loader(train_items, augment=True, sampler=sampler)

    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    checkpoint_epoch = 0
    pending_eval_epoch: int | None = None

    if mode == "fresh":
        student, source_checkpoint = base.load_model(START, device)
        start_state = {name: value.detach().clone() for name, value in student.state_dict().items()}
        optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    else:
        checkpoint = load_checkpoint(OUT / CHECKPOINT_NAME)
        checkpoint_epoch, phase = validate_resume(checkpoint, config_hash=config_hash,
                                                  input_hashes=hashes)
        student, source_checkpoint = base.load_model(START, device)
        start_state = {name: value.detach().clone() for name, value in student.state_dict().items()}
        restore_model_state(student, checkpoint["model_state"])
        optimizer = torch.optim.Adam(student.parameters(), lr=LR)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        restore_sampler_state(sampler, checkpoint["sampler_state"])
        history = list(checkpoint["history"])
        best = checkpoint.get("best")
        pending_eval_epoch = checkpoint_epoch if phase == "trained" else None
        _append_json_log(OUT / "resume_log.json", {
            "resumed_at": _timestamp(),
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_phase": phase,
            "pending_eval_epoch": pending_eval_epoch,
            "config_hash": config_hash,
            "resume_scope": RESUME_SCOPE,
        })
        print(f"[v3] --resume: 检查点 epoch={checkpoint_epoch} phase={phase}; "
              f"下一轮={checkpoint_epoch + 1}; {RESUME_SCOPE}", flush=True)

    teacher, _ = base.load_model(base.OLD, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    # load_model() constructs a randomly initialized network before loading weights.
    # Restore the saved RNG only after both networks have been constructed.
    if mode == "resume":
        restore_rng_state(checkpoint["rng"])

    if mode == "fresh":
        atomic_write_json(OUT / "protocol.json", {
            **config,
            "config_hash": config_hash,
            "input_hashes": hashes,
            "train_count": len(train_items),
            "external_count": len(external_items),
            "development_count": len(test_items),
            "external_baseline": external_baseline,
            "development_baseline": development_baseline,
            "disclosure": ("ISIC 2016 test was exposed before this repair trial and is "
                           "development feedback; v3 starts afresh from the original baseline"),
            "final_holdout": "official ISIC 2017 Task 1 test, never used for training or selection",
        })

    def persist(epoch: int, phase_name: str, trained: dict[str, torch.Tensor]) -> None:
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "created_at": _timestamp(),
            "trial": "role_a_from_baseline_v3",
            "epoch": int(epoch),
            "phase": phase_name,
            "config": config,
            "config_hash": config_hash,
            "input_hashes": hashes,
            "model_state": {name: value.detach().cpu().clone() for name, value in trained.items()},
            "optimizer_state": optimizer.state_dict(),
            "optimizer_class": "torch.optim.Adam",
            "sampler_state": sampler_state_payload(sampler),
            "rng": capture_rng_state(),
            "history": list(history),
            "best": best,
            "start_checkpoint": str(START),
            "start_checkpoint_sha256": hashes["start_checkpoint"],
            "resume_scope": RESUME_SCOPE,
        }
        atomic_torch_save(payload, OUT / CHECKPOINT_NAME)
        print(f"[v3] 原子检查点已保存: epoch={epoch} phase={phase_name} -> {OUT / CHECKPOINT_NAME}",
              flush=True)

    def train_one_epoch(epoch: int) -> tuple[dict[str, torch.Tensor], float]:
        student.train()
        total_loss = 0.0
        for batch in batches:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            is_original = torch.tensor([Path(name).stem in original_ids
                                        for name in batch["filename"]], device=device)
            micro = torch.tensor([train_areas[Path(name).stem] <= base.MICRO_TRAIN_MAX
                                  for name in batch["filename"]], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(image)
            logits = student(image)
            loss = loss_with_retention(logits, target, valid, micro, is_original, teacher_logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
        trained = {name: value.detach().clone() for name, value in student.state_dict().items()}
        return trained, total_loss / len(batches)

    def evaluate_candidates(epoch: int, trained: dict[str, torch.Tensor]) -> None:
        nonlocal best
        history[:] = [row for row in history if int(row["epoch"]) != int(epoch)]
        for alpha in BLENDS:
            set_blend(student, trained, start_state, alpha)
            path = OUT / "candidates" / f"epoch{epoch:02d}_alpha{alpha:.2f}.pt"
            candidate = {
                "model_state": {name: value.detach().cpu().clone()
                                for name, value in student.state_dict().items()},
                "config": source_checkpoint["config"],
                "epoch": int(epoch),
                "alpha": alpha,
                "source_checkpoint": str(START),
                "trial": "role_a_from_baseline_v3",
            }
            atomic_torch_save(candidate, path)
            external_grid = evaluate_grid(student, external_items, external_areas, external_cuts,
                                          device, THRESHOLDS)
            development_grid = evaluate_grid(student, test_items, test_areas, test_cuts,
                                             device, THRESHOLDS)
            for threshold in THRESHOLDS:
                external_metrics, external_records = external_grid[threshold]
                development_metrics, development_records = development_grid[threshold]
                accepted = (base.acceptance(external_metrics, external_baseline) and
                            base.acceptance(development_metrics, development_baseline))
                score = min(external_metrics["tiny"]["dice"] - external_baseline["tiny"]["dice"],
                            development_metrics["tiny"]["dice"] - development_baseline["tiny"]["dice"])
                history.append({
                    "epoch": int(epoch), "alpha": alpha, "threshold": threshold,
                    "candidate": str(path), "accepted": accepted, "score": score,
                    "external": external_metrics, "development2016": development_metrics,
                })
                atomic_write_json(OUT / "history.json", history)
                print(f"epoch={epoch} alpha={alpha} threshold={threshold} "
                      f"ext tiny={external_metrics['tiny']['dice']:.5f} "
                      f"large={external_metrics['large']['dice']:.5f} "
                      f"dev tiny={development_metrics['tiny']['dice']:.5f} "
                      f"large={development_metrics['large']['dice']:.5f} accepted={accepted}",
                      flush=True)
                if accepted and (best is None or score > float(best["score"])):
                    best = {"epoch": int(epoch), "alpha": alpha, "threshold": threshold,
                            "score": score, "candidate": str(path),
                            "external": external_metrics, "development2016": development_metrics}
                    atomic_torch_save({**candidate, "threshold": threshold,
                                       "selection": ("dual development gate; 2017 test remains "
                                                     "untouched")},
                                      OUT / "best.pt")
                    atomic_write_json(OUT / "best_metrics.json", {
                        "epoch": int(epoch), "alpha": alpha, "threshold": threshold,
                        "score": score, "external": external_metrics,
                        "external_records": external_records,
                        "development2016": development_metrics,
                        "development_records": development_records,
                    })
        set_blend(student, trained, start_state, 1.0)

    if pending_eval_epoch is not None:
        trained = {name: value.detach().clone() for name, value in student.state_dict().items()}
        evaluate_candidates(pending_eval_epoch, trained)
        persist(pending_eval_epoch, "evaluated", trained)

    for epoch in range(checkpoint_epoch + 1, EPOCHS + 1):
        trained, train_loss = train_one_epoch(epoch)
        print(f"[v3] epoch={epoch} 训练完成 train_loss={train_loss:.6f}", flush=True)
        persist(epoch, "trained", trained)
        evaluate_candidates(epoch, trained)
        persist(epoch, "evaluated", trained)

    atomic_write_json(OUT / COMPLETION_NAME, {
        "schema": COMPLETION_SCHEMA,
        "trial": "role_a_from_baseline_v3",
        "completed_at": _timestamp(),
        "completed_epochs": EPOCHS,
        "accepted": best is not None,
        "best": ({"epoch": best["epoch"], "alpha": best["alpha"], "threshold": best["threshold"],
                  "score": best["score"], "candidate": best["candidate"]} if best else None),
        "config_hash": config_hash,
        "rule": ("external and exposed 2016 development tiny gain; all other area-group Dice "
                 "nondecreasing"),
        "resume_scope": RESUME_SCOPE,
        "final_holdout": ("official ISIC 2017 Task 1 test, never touched by this trial"),
    })
    print(f"[v3] 试验完成: accepted={best is not None} -> {OUT / COMPLETION_NAME}", flush=True)
    return 0


# --------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="refine_from_baseline_v3.py", description=__doc__)
    parser.add_argument("--resume", action="store_true",
                        help="从最后一个完整轮次的原子检查点继续；只接受与当前配置/输入一致的检查点")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(resume=bool(args.resume))
    except (ResumeError, TrialStateError, FileNotFoundError) as error:
        print(f"[错误] {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
