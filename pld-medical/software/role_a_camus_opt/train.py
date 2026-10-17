"""CAMUS 优化臂训练（control20 λ0 / boundary20 λ0.05）：复用原成熟接口，仅改损失/初始化/身份。

设计：

* **不修改、不 monkeypatch** 原 ``software/role_a_camus``；本模块通过显式 import 复用其
  ``TrainConfig`` 字段、``LazyFrameBatches``、``native_dev_macro_dice``、断点与恢复、原子写与 STOP 语义；
* 损失：``CE + 三前景 soft Dice + λ·boundary``（λ=0 时与训练基线 ``multiclass_loss`` **逐值一致**）；
  卷积可 AMP，**CE/softmax/Dice/求和/边界项全部 FP32**；
* 初始化：从 `FROZEN_BEST.pt` **只取 weights**，fresh AdamW / GradScaler / cosine scheduler，
  并把「初始权重 SHA + 原 94 轮身份」写入断点（初始模型作为 epoch0 候选参与选模）；
* 配置：control20 λ=0、boundary20 λ=0.05；lr 3e-5、20 epoch、batch 8、seed 20261012、256、base16、
  3 pool、patience 21；
* 断点必需含 optimizer/scheduler/AMP/RNG/config(含 λ)/原基础源+opt 源码身份/data 身份/epoch/patience/
  history/initial_SHA；STOP 中途保存**完整 last**（partial 仅取证，不可冒充无损恢复）；
* 数据仍是原 `cache04` 惰性加载（不复制、不改缓存、不扩大增强因素）。
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

PROJECT = Path(r"E:/嵌赛/pld-medical")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from software.role_a_camus import geometry as base_geo          # noqa: E402
from software.role_a_camus import metrics as base_metrics        # noqa: E402
from software.role_a_camus import model as base_model            # noqa: E402
from software.role_a_camus import train as base_train            # noqa: E402
from software.role_a_camus_opt import boundary_loss as opt_loss  # noqa: E402

BASE_PACKAGE = PROJECT / "software/role_a_camus"
OPT_PACKAGE = PROJECT / "software/role_a_camus_opt"
BASE_DEV_NATIVE_FRAMES = 200
TRAIN_SPLIT = "training"
DEV_SPLIT = "validation"
ARMS = {"control20": 0.0, "boundary20": 0.05}
INITIAL_SHA_EXPECTED = "dcfe16a59cdde4821180561aa2d8c72393f1f53e6c8bbbcdade0541413a1e616"
FROZEN_BEST = PROJECT / "runs/role_a_camus/20261012/final_selection/FROZEN_BEST.pt"


class OptTrainError(RuntimeError):
    """配置/初始化/恢复不满足要求。"""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def base_source_identity() -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(BASE_PACKAGE.glob("*.py"))}


def opt_source_identity() -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(OPT_PACKAGE.glob("*.py"))}


@dataclass
class OptTrainConfig:
    """在基础配置上增加 λ 与身份字段（其余训练因素与原基线完全一致）。"""

    arm: str = "control20"
    boundary_weight: float = 0.0
    ce_weight: float = 1.0
    foreground_dice_weight: float = 1.0
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    patience: int = 21
    amp: bool = True
    size: int = 256
    base_channels: int = base_model.DEFAULT_BASE_CHANNELS
    num_pools: int = base_model.DEFAULT_NUM_POOLS
    seed: int = 20261012
    out_dir: str | None = None
    stop_file: str | None = None
    data_manifest: str | None = None
    initial_checkpoint: str = str(FROZEN_BEST)
    max_seconds: int = 7200

    def digest(self) -> str:
        payload = {key: value for key, value in sorted(asdict(self).items())
                   if key not in ("out_dir", "stop_file", "data_manifest")}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
                              ).hexdigest()


def resolve_arm(arm: str) -> float:
    if arm not in ARMS:
        raise OptTrainError(f"未知臂: {arm!r}（可选 {sorted(ARMS)}）")
    return ARMS[arm]


def initial_weights(model: nn.Module, checkpoint: Path = FROZEN_BEST) -> dict[str, Any]:
    """只取冻结权重初始化（不取 optimizer/RNG），并核验权重 SHA。"""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise OptTrainError(f"初始权重不存在: {checkpoint}")
    digest = sha256_file(checkpoint)
    if digest != INITIAL_SHA_EXPECTED:
        raise OptTrainError(f"初始权重 SHA 不符: {digest}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    return {"initial_SHA256": digest, "initial_epoch": payload.get("epoch"),
            "initial_best_epoch": payload.get("best_epoch"),
            "initial_best_metric": payload.get("best_metric"),
            "initial_optimizer_reset": True, "initial_scheduler_reset": True,
            "initial_scaler_reset": True}


def make_optimizer(model: nn.Module, config: OptTrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                             weight_decay=config.weight_decay)


def make_scheduler(optimizer: torch.optim.Optimizer, config: OptTrainConfig) -> Any:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))


def make_scaler(config: OptTrainConfig, device: torch.device) -> Any:
    enabled = bool(config.amp and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def run_epoch_opt(model: nn.Module, batches: Sequence[tuple[Tensor, Tensor, Tensor]],
                  optimizer: torch.optim.Optimizer | None, device: torch.device,
                  config: OptTrainConfig, scaler: Any | None = None, *,
                  stop_file: str | Path | None = None) -> dict[str, float]:
    """一个 epoch：使用 λ 感知损失；每 batch 检查 STOP（有界）。"""
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "ce": 0.0, "foreground_dice": 0.0, "boundary": 0.0}
    for image, label, valid in batches:
        if training and (base_train.stop_requested(stop_file) or time.monotonic() >= getattr(config, "_runtime_deadline", float("inf"))):
            raise base_train.StopRequested("batch 边界检测到 STOP_REQUEST")
        image = image.to(device)
        label = label.to(device)
        valid = valid.to(device)
        with torch.set_grad_enabled(training):
            use_amp = scaler is not None and config.amp and device.type == "cuda"
            if use_amp:
                with torch.autocast(device_type="cuda", enabled=True):
                    logits = model(image)
                loss, breakdown = opt_loss.multiclass_loss_for_arm(
                    logits, label, valid, boundary_weight=config.boundary_weight, size=config.size,
                    ce_weight=config.ce_weight, dice_weight=config.foreground_dice_weight)
            else:
                logits = model(image)
                loss, breakdown = opt_loss.multiclass_loss_for_arm(
                    logits, label, valid, boundary_weight=config.boundary_weight, size=config.size,
                    ce_weight=config.ce_weight, dice_weight=config.foreground_dice_weight)
            if not bool(torch.isfinite(loss)):
                raise OptTrainError("nonfinite loss; first error stop")
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        for key in totals:
            totals[key] += float(breakdown.get(key, 0.0))
    count = max(1, len(batches))
    return {key: value / count for key, value in totals.items()}


REQUIRED_KEYS = ("model_state", "optimizer_state", "scheduler_state", "amp_scaler_state", "rng_state",
                 "config", "config_digest", "boundary_weight", "input_spec", "base_source_identity",
                 "opt_source_identity", "data_identity", "epoch", "best_metric", "best_epoch",
                 "stale", "patience", "history", "initial_SHA256", "status")


def save_checkpoint_opt(path: Path, *, epoch: int, model: nn.Module,
                        optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any,
                        config: OptTrainConfig, history: Sequence[Mapping[str, Any]],
                        best_metric: float | None, best_epoch: int, stale: int,
                        data_identity: Mapping[str, Any], status: str,
                        initial_info: Mapping[str, Any],
                        max_seconds: int | None = None,
                        elapsed_seconds: float | None = None) -> dict[str, Any]:
    payload = {"format": "camus-opt-checkpoint-v1", "epoch": int(epoch), "status": status,
               "arm": config.arm, "boundary_weight": float(config.boundary_weight),
               "config": asdict(config), "config_digest": config.digest(),
               "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
               "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
               "amp_scaler_state": scaler.state_dict() if scaler is not None else None,
               "rng_state": base_train.capture_rng_state(), "history": list(history),
               "best_metric": best_metric, "best_epoch": int(best_epoch), "stale": int(stale),
               "patience": config.patience, "max_seconds": max_seconds,
               "elapsed_seconds": elapsed_seconds,
               "base_source_identity": base_source_identity(),
               "opt_source_identity": opt_source_identity(),
               "data_identity": dict(data_identity), **dict(initial_info),
               "input_spec": {"in_channels": 1, "num_classes": 4, "size": config.size,
                              "letterbox": True, "label_nearest": True,
                              "ignore_index": opt_loss.IGNORE_INDEX,
                              "normalization": "clip0_255_div255"},
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    base_train.atomic_torch_save(payload, Path(path))
    return payload


def save_partial_opt(path: Path, *, epoch: int, model: nn.Module,
                     optimizer: torch.optim.Optimizer, reason: str,
                     data_identity: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"format": "camus-opt-partial-v1", "epoch": int(epoch), "reason": reason,
               "recoverable": False,
               "note": "epoch 中途快照，仅取证；恢复请用上一个完整 epoch 的 last.pt",
               "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
               "rng_state": base_train.capture_rng_state(),
               "data_identity": dict(data_identity),
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    base_train.atomic_torch_save(payload, Path(path))
    return payload


def load_checkpoint_opt(path: Path, *, config: OptTrainConfig,
                        data_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """恢复：缺项 / config / 源码 / 数据身份不符一律拒绝。"""
    path = Path(path)
    if not path.is_file():
        raise OptTrainError(f"断点不存在: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing = [key for key in REQUIRED_KEYS if key not in payload or (payload[key] is None and not (key == "best_metric" and payload.get("epoch") == 0))]
    if missing:
        raise OptTrainError(f"断点缺少必需项: {missing}")
    if payload.get("format") != "camus-opt-checkpoint-v1":
        raise OptTrainError(f"断点格式不可用于恢复: {payload.get('format')!r}")
    if payload.get("config_digest") != config.digest():
        raise OptTrainError("config 摘要不符，拒绝恢复")
    if abs(float(payload.get("boundary_weight", -1)) - config.boundary_weight) > 1e-12:
        raise OptTrainError("λ 与断点不符，拒绝恢复")
    if payload.get("base_source_identity") != base_source_identity():
        raise OptTrainError("原基础源码身份不符，拒绝恢复")
    if payload.get("opt_source_identity") != opt_source_identity():
        raise OptTrainError("opt 源码身份不符，拒绝恢复")
    if data_identity:
        recorded = payload.get("data_identity") or {}
        for key, value in data_identity.items():
            if recorded.get(key) != value:
                raise OptTrainError(f"数据身份不符: {key}")
    return payload


def restore_opt_state(payload: Mapping[str, Any], *, model: nn.Module,
                      optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any) -> dict[str, Any]:
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("amp_scaler_state") is not None:
        scaler.load_state_dict(payload["amp_scaler_state"])
    base_train.restore_rng_state(payload["rng_state"])
    return {"epoch": int(payload["epoch"]), "best_metric": payload.get("best_metric"),
            "best_epoch": int(payload.get("best_epoch") or 0), "stale": int(payload.get("stale") or 0)}


def fit_arm(batches: Any, dev_frames: Sequence[Any], config: OptTrainConfig, model: nn.Module,
            device: torch.device | str = "cpu", *, data_identity: Mapping[str, Any] | None = None,
            resume_from: Path | None = None, base_identity: Mapping[str, Any] | None = None,
            expected_dev_frames: int = BASE_DEV_NATIVE_FRAMES) -> dict[str, Any]:
    """Paired fine tuning with an eligible epoch0 and byte-preserved complete checkpoints."""
    device = torch.device(device)
    model.to(device)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    scaler = make_scaler(config, device)
    out_dir = Path(config.out_dir or "camus_opt_arm")
    if resume_from is None and out_dir.exists() and any(out_dir.iterdir()):
        raise OptTrainError("fresh training output must be empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    initial_info = dict(base_identity or {})
    data_id = dict(data_identity or {})
    history = []
    best = None
    best_epoch = stale = 0
    start_epoch = 1
    if resume_from is not None:
        payload = load_checkpoint_opt(Path(resume_from), config=config, data_identity=data_id)
        restored = restore_opt_state(payload, model=model, optimizer=optimizer,
                                     scheduler=scheduler, scaler=scaler)
        history = list(payload['history'])
        best, best_epoch, stale = restored['best_metric'], restored['best_epoch'], restored['stale']
        initial_info = {key: value for key, value in payload.items() if key.startswith('initial_')}
        start_epoch = restored['epoch'] + 1
    if initial_info.get('initial_SHA256') != INITIAL_SHA_EXPECTED:
        raise OptTrainError("initial frozen model identity is required")
    started = time.monotonic()
    config._runtime_deadline = started + config.max_seconds

    def save(path, epoch, status):
        return save_checkpoint_opt(path, epoch=epoch, model=model, optimizer=optimizer,
                                   scheduler=scheduler, scaler=scaler, config=config, history=history,
                                   best_metric=best, best_epoch=best_epoch, stale=stale,
                                   data_identity=data_id, status=status, initial_info=initial_info,
                                   max_seconds=config.max_seconds, elapsed_seconds=time.monotonic()-started)

    def evaluate():
        result = base_train.native_dev_macro_dice(
            model, dev_frames, device, batch_size=config.batch_size, stop_file=config.stop_file,
            expected_frames=expected_dev_frames, deadline_monotonic=config._runtime_deadline)
        if result[0] is None or not result[2].get('complete'):
            raise OptTrainError('native validation is incomplete or has no metric')
        return result

    def stopped(epoch, reason, partial):
        # Never relabel partially updated weights/Adam/RNG as epoch-1.
        if partial:
            save_partial_opt(out_dir/'partial_state.pt', epoch=epoch, model=model,
                             optimizer=optimizer, reason=reason, data_identity=data_id)
        last = out_dir/'last.pt'
        if not last.is_file():
            raise OptTrainError('no complete checkpoint available at stop')
        committed = torch.load(last, map_location='cpu', weights_only=False)
        result = {'status':'stopped','epoch':committed['epoch'],'best_metric':best,
                  'best_epoch':best_epoch,'reason':reason,'out_dir':out_dir.as_posix(),
                  'resume_checkpoint':last.as_posix(),'last_SHA256':sha256_file(last),
                  'history':history,'elapsed_seconds':time.monotonic()-started}
        base_train.atomic_write_json(result, out_dir/'STATUS.json')
        return result

    if resume_from is None:
        # Initialize a complete recoverable epoch0 before the initial dev pass.
        save(out_dir/'last.pt', 0, 'initialized')
        try:
            metric, records, info = evaluate()
        except base_train.StopRequested:
            return stopped(0, 'STOP/deadline during initial validation', False)
        expected_initial = getattr(config, '_expected_initial_metric', None)
        if expected_initial is not None and abs(metric - expected_initial) > 1e-9:
            raise OptTrainError('initial native dev Dice differs from frozen baseline')
        best, best_epoch = metric, 0
        history.append({'epoch':0,'train':None,'dev_native_macro_dice':metric,
                        'dev_info':info,'source':'initial_frozen_weights'})
        save(out_dir/'best.pt', 0, 'best')
        save(out_dir/'last.pt', 0, 'initialized')
        base_train.atomic_write_json({'epoch':0,'metric':'native_patient_balanced_macro_dice',
                                      'value':metric,'records':records,'dev_info':info},
                                     out_dir/'BEST_DEV_RECORDS.json')
    for epoch in range(start_epoch, config.epochs+1):
        if base_train.stop_requested(config.stop_file) or time.monotonic() >= config._runtime_deadline:
            return stopped(epoch, 'STOP/deadline before epoch', False)
        if hasattr(batches, 'set_epoch'):
            batches.set_epoch(epoch)
        try:
            train_metrics = run_epoch_opt(model, batches, optimizer, device, config, scaler,
                                          stop_file=config.stop_file)
            metric, records, info = evaluate()
        except base_train.StopRequested:
            return stopped(epoch, 'STOP/deadline during training or validation', True)
        scheduler.step()
        improved = best is None or metric > best
        if improved:
            best, best_epoch, stale = metric, epoch, 0
        else:
            stale += 1
        entry = {'epoch':epoch,'train':train_metrics,'dev_native_macro_dice':metric,
                 'dev_info':info,'lr':float(optimizer.param_groups[0]['lr']),
                 'elapsed_seconds':time.monotonic()-started}
        history.append(entry)
        if improved:
            save(out_dir/'best.pt', epoch, 'best')
            base_train.atomic_write_json({'epoch':epoch,'metric':'native_patient_balanced_macro_dice',
                                          'value':metric,'records':records,'dev_info':info},
                                         out_dir/'BEST_DEV_RECORDS.json')
        save(out_dir/'last.pt', epoch, 'running')
        base_train.atomic_write_json({'status':'running','epoch':epoch,'epoch_completed':True,
                                      'dev_native_macro_dice':metric,'best_metric':best,
                                      'best_epoch':best_epoch,'stale_epochs':stale,
                                      'boundary_weight':config.boundary_weight,
                                      'elapsed_seconds':entry['elapsed_seconds']}, out_dir/'STATUS.json')
        if stale >= config.patience:
            result = {'status':'early_stopped','epoch':epoch,'best_metric':best,
                      'best_epoch':best_epoch,'out_dir':out_dir.as_posix(),'history':history}
            base_train.atomic_write_json(result,out_dir/'STATUS.json')
            return result
    result = {'status':'completed','epoch':history[-1]['epoch'] if history else 0,
              'best_metric':best,'best_epoch':best_epoch,'out_dir':out_dir.as_posix(),
              'history':history,'elapsed_seconds':time.monotonic()-started}
    base_train.atomic_write_json(result,out_dir/'STATUS.json')
    return result
