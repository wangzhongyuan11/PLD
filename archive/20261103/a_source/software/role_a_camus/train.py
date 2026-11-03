"""CAMUS 训练骨架（本轮不运行；可真实运行但需 Codex 批准文件与 QC 通过）。

要点：

* ``train.run_epoch`` 统一接口：``multiclass_loss`` 返回 ``{"loss", "ce", "foreground_dice"}``。
* 损失 = CE（补边 ``ignore_index=255`` 忽略）+ 三前景 soft Dice。
* 数据增强为**成对**几何（水平镜像、小角度旋转+缩放），valid 与标签同步；灰度亮度扰动只训练用。
* 每 epoch 用 **native 三前景 Dice（帧等权→患者等权）** 选模；dev loss 仅报告，不参与选模。
* 断点含 model/optimizer/scheduler/scaler/RNG/best/epoch/patience/config/source/data 身份，原子 best/last。
* STOP 每 batch 有界响应：安全保存 + ``stopped`` 状态，**不得声称成功**；异常/ OOM 真实非 0 退出且不重试。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from . import geometry as geo
from . import metrics as camus_metrics
from . import model as camus_model

IGNORE_INDEX = geo.IGNORE_INDEX
TRAIN_SPLIT = "training"
DEV_SPLIT = "validation"
TEST_SPLIT = "testing"


class StopRequested(RuntimeError):
    """检测到 STOP：已安全保存，训练状态为 stopped（非成功）。"""


class ResumeRejected(RuntimeError):
    """断点不完整或身份不匹配，拒绝恢复。"""


class QCFailed(RuntimeError):
    """准入检查失败（缺批准、缺 cache、QC 未通过）。"""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def code_identity(package_dir: Path | None = None) -> dict[str, str]:
    package_dir = Path(package_dir or Path(__file__).parent)
    return {path.name: sha256_file(path)
            for path in sorted(package_dir.glob("*.py")) if path.is_file()}


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 15
    amp: bool = True
    size: int = 256
    base_channels: int = camus_model.DEFAULT_BASE_CHANNELS
    num_pools: int = camus_model.DEFAULT_NUM_POOLS
    seed: int = 20261012
    ce_weight: float = 1.0
    foreground_dice_weight: float = 1.0
    num_workers: int = 2
    out_dir: str | None = None
    stop_file: str | None = None
    approval_file: str | None = None
    data_manifest: str | None = None
    run_name: str = "camus_baseline"

    def digest(self) -> str:
        payload = {k: v for k, v in sorted(asdict(self).items())
                   if k not in ("out_dir", "stop_file", "approval_file", "data_manifest")}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(),
                             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def stop_requested(stop_file: str | Path | None) -> bool:
    return bool(stop_file) and Path(stop_file).exists()


def atomic_torch_save(payload: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_write_json(payload: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def multiclass_loss(logits: Tensor, target: Tensor, valid: Tensor | None = None, *,
                    ce_weight: float = 1.0, dice_weight: float = 1.0,
                    foreground: Sequence[int] = geo.FOREGROUND_CLASSES
                    ) -> tuple[Tensor, dict[str, float]]:
    """CE + 三前景 soft Dice。返回 ``(total_loss, {"loss","ce","foreground_dice"})``。

    接口统一：breakdown 的键与 ``run_epoch`` 累加键完全一致（修正上一版 ``'loss'`` KeyError）。
    """
    target = target.long()
    if valid is None:
        keep = torch.ones_like(target, dtype=torch.bool)
    else:
        keep = valid.bool().squeeze(1)
    masked_target = torch.where(keep, target, torch.full_like(target, IGNORE_INDEX))
    ce = nn.functional.cross_entropy(logits.float(), masked_target, ignore_index=IGNORE_INDEX)
    probabilities = torch.softmax(logits.float(), dim=1)  # FP32 pixel sums: AMP half sums can overflow above 65504
    terms: list[Tensor] = []
    for class_index in foreground:
        pred = probabilities[:, class_index][keep]
        real = (masked_target == class_index)[keep].float()
        intersection = (pred * real).sum()
        denominator = pred.sum() + real.sum()
        terms.append(1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0))
    dice_loss = torch.stack(terms).mean() if terms else torch.zeros((), dtype=logits.dtype)
    total = ce_weight * ce + dice_weight * dice_loss
    breakdown = {"loss": float(total.detach()), "ce": float(ce.detach()),
                 "foreground_dice": float(dice_loss.detach())}
    return total, breakdown


def run_epoch(model: nn.Module, batches: Sequence[tuple[Tensor, Tensor, Tensor]],
              optimizer: torch.optim.Optimizer | None, device: torch.device, config: TrainConfig,
              scaler: Any | None = None, *, stop_file: str | Path | None = None
              ) -> dict[str, float]:
    """一个 epoch；``optimizer=None`` 时仅评估。每个 batch 检查 STOP 并有界退出。"""
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "ce": 0.0, "foreground_dice": 0.0}
    for image, label, valid in batches:
        if training and (stop_requested(stop_file) or time.monotonic() >= getattr(config, "_runtime_deadline", float("inf"))):
            raise StopRequested("batch 边界检测到 STOP_REQUEST")
        image = image.to(device)
        label = label.to(device)
        valid = valid.to(device)
        with torch.set_grad_enabled(training):
            if scaler is not None and config.amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", enabled=True):
                    logits = model(image)
                    loss, breakdown = multiclass_loss(logits, label, valid,
                                                      ce_weight=config.ce_weight,
                                                      dice_weight=config.foreground_dice_weight)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                logits = model(image)
                loss, breakdown = multiclass_loss(logits, label, valid,
                                                  ce_weight=config.ce_weight,
                                                  dice_weight=config.foreground_dice_weight)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
        for key in totals:
            totals[key] += float(breakdown[key])
    count = max(1, len(batches))
    return {key: value / count for key, value in totals.items()}


def augment_pair(image: np.ndarray, label: np.ndarray, valid: np.ndarray, rng: random.Random,
                 *, enable_photometry: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """成对几何增强（水平镜像、小角度旋转+缩放），valid/标签同步；亮度扰动只训练用。"""
    image = np.array(image, dtype=np.float32, copy=True)
    label = np.array(label, copy=True)
    valid = np.array(valid, dtype=np.float32, copy=True)
    if rng.random() < 0.5:
        image, label, valid = image[:, ::-1].copy(), label[:, ::-1].copy(), valid[:, ::-1].copy()
    if rng.random() < 0.5:
        angle = rng.uniform(-10.0, 10.0)
        scale = rng.uniform(0.95, 1.05)
        matrix = torch.tensor([[[math.cos(math.radians(angle)) * scale,
                                 -math.sin(math.radians(angle)) * scale, 0.0],
                                [math.sin(math.radians(angle)) * scale,
                                 math.cos(math.radians(angle)) * scale, 0.0]]])
        image_t = torch.from_numpy(image)[None, None]
        label_t = torch.from_numpy(label.astype(np.float32))[None, None]
        valid_t = torch.from_numpy(valid)[None, None]
        grid = nn.functional.affine_grid(matrix, list(image_t.shape), align_corners=False)
        image = nn.functional.grid_sample(image_t, grid, mode="bilinear",
                                          align_corners=False)[0, 0].numpy()
        label = nn.functional.grid_sample(label_t, grid, mode="nearest",
                                          align_corners=False)[0, 0].numpy().round().astype(np.int64)
        valid = (nn.functional.grid_sample(valid_t, grid, mode="nearest",
                                           align_corners=False)[0, 0].numpy() > 0.5).astype(np.float32)
    if enable_photometry and rng.random() < 0.5:
        image = np.clip(image * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05), 0.0, 1.0)
    return image.astype(np.float32), label, valid


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                             weight_decay=config.weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig) -> Any:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))


REQUIRED_CHECKPOINT_KEYS = ("model_state", "optimizer_state", "scheduler_state", "amp_scaler_state",
                            "rng_state", "config", "config_digest", "input_spec", "source_identity",
                            "data_identity", "epoch", "best_metric", "best_epoch", "stale",
                            "patience", "status")


def save_checkpoint(path: Path, *, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Any, scaler: Any, config: TrainConfig, history: Sequence[Mapping[str, Any]],
                    best_metric: float | None, source_identity: Mapping[str, str],
                    data_identity: Mapping[str, Any], status: str, best_epoch: int = 0,
                    stale: int = 0, max_seconds: int | None = None,
                    elapsed_seconds: float | None = None) -> dict[str, Any]:
    """原子保存完整断点（含 RNG/AMP/scheduler/patience/best_epoch/stale/config/data 身份）。"""
    payload = {"format": "camus-checkpoint-v1", "epoch": int(epoch), "status": status,
               "config": asdict(config), "config_digest": config.digest(),
               "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
               "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
               "amp_scaler_state": scaler.state_dict() if scaler is not None else None,
               "rng_state": capture_rng_state(), "history": list(history),
               "best_metric": best_metric, "best_epoch": int(best_epoch), "stale": int(stale),
               "patience": config.patience, "max_seconds": max_seconds,
               "elapsed_seconds": elapsed_seconds,
               "source_identity": dict(source_identity), "data_identity": dict(data_identity),
               "input_spec": {"in_channels": 1, "num_classes": camus_model.NUM_CLASSES,
                              "size": config.size, "letterbox": True, "label_nearest": True,
                              "ignore_index": IGNORE_INDEX, "normalization": "clip0_255_div255"},
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_torch_save(payload, Path(path))
    return payload


def save_partial_state(path: Path, *, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer,
                       reason: str, source_identity: Mapping[str, str],
                       data_identity: Mapping[str, Any]) -> dict[str, Any]:
    """epoch 中途 STOP/异常时的**取证**快照：明确标注为半个 epoch，不得用于精确恢复。"""
    payload = {"format": "camus-partial-state-v1", "epoch": int(epoch), "reason": reason,
               "recoverable": False,
               "note": "epoch 中途快照，仅用于取证；恢复请使用上一个完整 epoch 的 last.pt",
               "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
               "rng_state": capture_rng_state(), "source_identity": dict(source_identity),
               "data_identity": dict(data_identity),
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_torch_save(payload, Path(path))
    return payload


def load_checkpoint(path: Path, *, config: TrainConfig,
                    source_identity: Mapping[str, str] | None = None,
                    data_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """恢复断点：缺项 / config / 源码 / 数据身份不符一律拒绝（不静默降级）。"""
    path = Path(path)
    if not path.is_file():
        raise ResumeRejected(f"断点不存在: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in payload or (payload[key] is None and not (key == "best_metric" and payload.get("epoch") == 0))]
    if missing:
        raise ResumeRejected(f"断点缺少必需项: {missing}")
    if payload.get("format") != "camus-checkpoint-v1":
        raise ResumeRejected(f"断点格式不可用于恢复: {payload.get('format')!r}")
    if payload.get("config_digest") != config.digest():
        raise ResumeRejected("config 摘要不符，拒绝恢复")
    if source_identity:
        recorded = payload.get("source_identity") or {}
        for name, digest in source_identity.items():
            if recorded.get(name) != digest:
                raise ResumeRejected(f"源码身份不符: {name}")
    if data_identity:
        recorded = payload.get("data_identity") or {}
        for key, value in data_identity.items():
            if recorded.get(key) != value:
                raise ResumeRejected(f"数据身份不符: {key}")
    return payload


def restore_training_state(payload: Mapping[str, Any], *, model: nn.Module,
                           optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any
                           ) -> dict[str, Any]:
    """真实恢复 model/optimizer/scheduler/scaler/RNG 与计数（epoch/best/best_epoch/stale）。"""
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("amp_scaler_state") is not None:
        scaler.load_state_dict(payload["amp_scaler_state"])
    restore_rng_state(payload["rng_state"])
    return {"epoch": int(payload["epoch"]), "best_metric": payload.get("best_metric"),
            "best_epoch": int(payload.get("best_epoch") or 0), "stale": int(payload.get("stale") or 0)}


def verify_approval(approval_path: str | Path | None, config: TrainConfig, *,
                    source_identity: Mapping[str, str], cache_path: str | Path,
                    cache_sha256: str, out_dir: str | Path, train_frames: int,
                    dev_frames: int) -> dict[str, Any]:
    """训练准入：批准文件必须绑定源码身份、cache SHA、config digest、输出目录与 400/50 划分。"""
    if not approval_path:
        raise QCFailed("缺少批准文件参数（--approval）")
    path = Path(approval_path)
    if not path.is_file():
        raise QCFailed(f"批准文件不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = {"approved": payload.get("approved") is True,
              "training_approved": payload.get("training_approved") is True,
              "source_identity": payload.get("source_identity") == dict(source_identity),
              "cache_sha256": payload.get("cache_sha256") == cache_sha256,
              "cache_path": (payload.get("cache_manifest") or "") == Path(cache_path).as_posix(),
              "config_digest": payload.get("config_digest") == config.digest(),
              "out_dir": (payload.get("out_dir") or "") == Path(out_dir).as_posix(),
              "train_frames": payload.get("train_frames") == train_frames,
              "dev_frames": payload.get("dev_frames") == dev_frames}
    failures = sorted(name for name, ok in checks.items() if not ok)
    if failures:
        raise QCFailed(f"批准文件校验失败: {failures}")
    return payload


def require_cache(data_manifest: str | Path | None, *, require_combined: bool = True
                  ) -> dict[str, Any]:
    """训练准入：缓存清单必须是**受审 combined**（含 training 1600 + validation 200）。"""
    if not data_manifest:
        raise QCFailed("缺少缓存清单参数（--data-manifest）")
    path = Path(data_manifest)
    if not path.is_file():
        raise QCFailed(f"缓存清单不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("qc_passed"):
        raise QCFailed("缓存清单 qc_passed != true，拒绝训练")
    if require_combined:
        if payload.get("schema") != "camus-combined-cache-manifest-v1":
            raise QCFailed(f"需要 combined manifest（含 train+dev），收到 schema={payload.get('schema')!r}")
        counts = payload.get("counts") or {}
        if counts.get("training") != 1600 or counts.get("validation") != 200:
            raise QCFailed(f"combined manifest 帧数非法: {counts!r}")
    if payload.get("test_gt_decoded") is True:
        raise QCFailed("缓存清单声称解码过测试 GT，拒绝训练")
    return payload


def fit(train_batches: Any, dev_frames: Sequence[Any], config: TrainConfig, model: nn.Module,
        device: torch.device | str = "cpu", *, data_identity: Mapping[str, Any] | None = None,
        source_identity: Mapping[str, str] | None = None, resume_from: Path | None = None,
        max_seconds: int = 10800) -> dict[str, Any]:
    """训练循环：逐 epoch 惰性取批次、native Dice 选模、顺序正确的原子断点、STOP/超时安全收尾。

    * 断点顺序：先更新 best/best_epoch/stale，**再**保存 ``last.pt``（不落后一轮）。
    * ``--resume`` 真实恢复 model/optimizer/scheduler/scaler/RNG 与 epoch/best/patience。
    * epoch 中途 STOP：保存 ``partial_state.pt`` 取证，并从上一个完整 epoch 的 ``last.pt`` 收尾。
    * 超时或 STOP 退出为 ``stopped``（非 0 成功）；异常/OOM 直接抛出，非 0 退出且不重试。
    """
    device = torch.device(device)
    model.to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = (torch.amp.GradScaler("cuda", enabled=bool(config.amp and device.type == "cuda"))
              if hasattr(torch, "amp")
              else torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda")))
    out_dir = Path(config.out_dir or "camus_run")
    out_dir.mkdir(parents=True, exist_ok=True)
    identity = dict(source_identity or code_identity())
    data_id = dict(data_identity or {})
    history: list[dict[str, Any]] = []
    best: float | None = None
    best_epoch = 0
    stale = 0
    start_epoch = 1
    if resume_from is not None:
        payload = load_checkpoint(Path(resume_from), config=config, source_identity=identity,
                                  data_identity=data_id)
        restored = restore_training_state(payload, model=model, optimizer=optimizer,
                                          scheduler=scheduler, scaler=scaler)
        history = list(payload.get("history") or [])
        best, best_epoch, stale = (restored["best_metric"], restored["best_epoch"],
                                   restored["stale"])
        start_epoch = restored["epoch"] + 1
    started = time.monotonic()
    config._runtime_deadline = started + max_seconds
    if resume_from is None:
        save_checkpoint(out_dir / "last.pt", epoch=0, model=model, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler, config=config, history=history,
                        best_metric=None, source_identity=identity, data_identity=data_id,
                        status="initialized", best_epoch=0, stale=0,
                        max_seconds=max_seconds, elapsed_seconds=0.0)

    def safe_stop(epoch, reason):
        # Retain the already committed full-epoch checkpoint; partial weights
        # belong only to a separate non-resumable evidence artifact.
        save_partial_state(out_dir / "partial_state.pt", epoch=epoch, model=model,
                           optimizer=optimizer, reason=reason,
                           source_identity=identity, data_identity=data_id)
        status={"status":"stopped", "reason":reason, "epoch":epoch-1,
                "best_metric":best, "best_epoch":best_epoch,
                "elapsed_seconds":time.monotonic()-started,
                "partial_state":(out_dir/"partial_state.pt").as_posix(),
                "resume_checkpoint":(out_dir/"last.pt").as_posix()}
        atomic_write_json(status, out_dir/"STATUS.json")
        return dict(status, out_dir=out_dir.as_posix(), history=history)

    for epoch in range(start_epoch, config.epochs + 1):
        if hasattr(train_batches, "set_epoch"):
            train_batches.set_epoch(epoch)          # 每 epoch 重新洗牌/增强
        elapsed = time.monotonic() - started
        if elapsed > max_seconds:
            save_checkpoint(out_dir / "last.pt", epoch=epoch - 1, model=model, optimizer=optimizer,
                            scheduler=scheduler, scaler=scaler, config=config, history=history,
                            best_metric=best, source_identity=identity, data_identity=data_id,
                            status="stopped", best_epoch=best_epoch, stale=stale,
                            max_seconds=max_seconds, elapsed_seconds=elapsed)
            atomic_write_json({"status": "stopped", "reason": "max_seconds exceeded",
                               "epoch": epoch - 1, "best_metric": best,
                               "elapsed_seconds": elapsed, "max_seconds": max_seconds},
                              out_dir / "STATUS.json")
            return {"status": "stopped", "epoch": epoch - 1, "best_metric": best,
                    "out_dir": out_dir.as_posix(), "history": history}
        try:
            train_metrics = run_epoch(model, train_batches, optimizer, device, config, scaler,
                                      stop_file=config.stop_file)
        except StopRequested:
            return safe_stop(epoch, "STOP or deadline during training batch")
        try:
            native_metric, dev_records, dev_info = native_dev_macro_dice(
                model, dev_frames, device, batch_size=config.batch_size, stop_file=config.stop_file,
                deadline_monotonic=config._runtime_deadline,
                expected_frames=max(4, 4 * len({f.patient for f in dev_frames}) if dev_frames else 0))
        except StopRequested:
            return safe_stop(epoch, "STOP or deadline during development evaluation")
        scheduler.step()
        improved = native_metric is not None and (best is None or native_metric > best)
        if improved:
            best, best_epoch, stale = native_metric, epoch, 0
        else:
            stale += 1
        entry = {"epoch": epoch, "train": train_metrics, "dev_loss": None,
                 "dev_native_macro_dice": native_metric, "dev_info": dev_info,
                 "lr": float(optimizer.param_groups[0]["lr"]),
                 "elapsed_seconds": time.monotonic() - started}
        history.append(entry)
        # 先更新计数并保存 best，再保存 last（last 与 best/stale 同步，不落后一轮）
        if improved:
            save_checkpoint(out_dir / "best.pt", epoch=epoch, model=model, optimizer=optimizer,
                            scheduler=scheduler, scaler=scaler, config=config, history=history,
                            best_metric=best, source_identity=identity, data_identity=data_id,
                            status="best", best_epoch=best_epoch, stale=stale,
                            max_seconds=max_seconds, elapsed_seconds=entry["elapsed_seconds"])
            atomic_write_json({"epoch": epoch, "metric": "native_patient_balanced_macro_dice",
                               "value": best, "records": dev_records, "dev_info": dev_info},
                              out_dir / "BEST_DEV_RECORDS.json")
        save_checkpoint(out_dir / "last.pt", epoch=epoch, model=model, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler, config=config, history=history,
                        best_metric=best, source_identity=identity, data_identity=data_id,
                        status="running", best_epoch=best_epoch, stale=stale,
                        max_seconds=max_seconds, elapsed_seconds=entry["elapsed_seconds"])
        atomic_write_json({"status": "running", "epoch": epoch, "epoch_completed": True,
                           "dev_native_macro_dice": native_metric, "best_metric": best,
                           "best_epoch": best_epoch, "stale_epochs": stale,
                           "elapsed_seconds": entry["elapsed_seconds"], "max_seconds": max_seconds},
                          out_dir / "STATUS.json")
        if stale >= config.patience:
            atomic_write_json({"status": "early_stopped", "epoch": epoch, "best_epoch": best_epoch,
                               "best_metric": best, "patience": config.patience,
                               "elapsed_seconds": time.monotonic() - started},
                              out_dir / "STATUS.json")
            return {"status": "early_stopped", "epoch": epoch, "best_metric": best,
                    "best_epoch": best_epoch, "out_dir": out_dir.as_posix(), "history": history}
    return {"status": "completed", "epoch": history[-1]["epoch"] if history else 0,
            "best_metric": best, "best_epoch": best_epoch, "out_dir": out_dir.as_posix(),
            "history": history}


# --------------------------------------------------------------------------- CLI

def frames_to_batches(frames: Sequence[Any], batch_size: int, *, augment: bool, seed: int
                      ) -> list[tuple[Tensor, Tensor, Tensor]]:
    """缓存帧 → batch 列表；训练时按 seed 随机打乱并做**成对**增强。"""
    rng = random.Random(seed)
    order = list(range(len(frames)))
    if augment:
        rng.shuffle(order)
    batches: list[tuple[Tensor, Tensor, Tensor]] = []
    for start in range(0, len(order), batch_size):
        images, labels, valids = [], [], []
        for index in order[start:start + batch_size]:
            frame = frames[index]
            image = np.asarray(frame.image, dtype=np.float32)
            label = np.asarray(frame.label, dtype=np.int64)
            valid = np.asarray(frame.valid, dtype=np.float32)
            if augment:
                image, label, valid = augment_pair(image, label, valid, rng, enable_photometry=True)
            images.append(torch.from_numpy(image)[None])
            labels.append(torch.from_numpy(label))
            valids.append(torch.from_numpy(valid)[None])
        batches.append((torch.stack(images), torch.stack(labels), torch.stack(valids)))
    return batches


def load_cache_frames(manifest_path: Path, split: str) -> list[Any]:
    """从缓存清单读取指定 split 的帧（含原生图像/GT、spacing、affine、官方 quality）。"""
    from . import data as camus_data

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    frames: list[Any] = []
    for record in manifest.get("records", []):
        if record.get("split") != split:
            continue
        payload = camus_data.load_cached_frame(Path(record["cache_file"]))
        frames.append(camus_data.CamusFrame(
            patient=record["patient"], split=split, view=record["view"], phase=record["phase"],
            image=payload["image"], valid=payload["valid"], label=payload["label"],
            meta=payload["meta"], spacing_mm=tuple(payload["spacing_mm"]),
            native_image=payload["native_image"], native_label=payload["native_label"],
            native_shape=tuple(record["native_shape"]), affine=tuple(tuple(row) for row in payload["affine"]),
            source_sha256=record["source_sha256"], quality=record.get("quality", "unknown"),
            quality_source=payload.get("quality_source"), histogram=record.get("histogram", {})))
    return frames


def native_dev_macro_dice(model: nn.Module, frames: Sequence[Any], device: torch.device,
                          *, batch_size: int = 4, stop_file: str | Path | None = None,
                          expected_frames: int | None = None, deadline_monotonic: float | None = None
                          ) -> tuple[float | None, list[dict[str, Any]], dict[str, Any]]:
    """dev 上的**快速** native Dice 选模：只算三前景 Dice/召回聚合，不算 HD95 与体积。

    要求 dev 完整（默认 50 患者 × 4 帧 = 200 帧）；不完整即不给出选模指标。
    """
    from . import metrics as camus_metrics  # 局部导入避免循环

    frames = list(frames)
    required = expected_frames if expected_frames is not None else 200
    patients = {frame.patient for frame in frames}
    complete = len(frames) == required and all(
        sum(1 for frame in frames if frame.patient == patient) == 4 for patient in patients)
    info = {"frames": len(frames), "patients": len(patients), "required_frames": required,
            "complete": bool(complete)}
    if not complete:
        info["reason"] = "dev 不完整（必须 50 患者×4 帧=200），不提供选模指标"
        return None, [], info
    was_training = model.training
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            if stop_requested(stop_file) or (deadline_monotonic is not None and time.monotonic() >= deadline_monotonic):
                raise StopRequested("dev 评估期间检测到 STOP_REQUEST")
            chunk = frames[start:start + batch_size]
            batch = torch.stack([torch.from_numpy(np.asarray(f.image, dtype=np.float32))[None]
                                 for f in chunk]).to(device)
            logits = model(batch)
            for index, frame in enumerate(chunk):
                native_prediction = geo.argmax_on_native_torch(logits[index:index + 1].cpu(),
                                                               frame.meta)[0].numpy()
                if frame.native_label is None:
                    records.append({"patient": frame.patient, "view": frame.view,
                                    "phase": frame.phase, "quality": frame.quality,
                                    "frame_id": f"{frame.patient}_{frame.view}_{frame.phase}",
                                    "per_class": {}, "foreground_macro_dice": None,
                                    "null_classes": ["all"],
                                    "reason": "native GT 缺失，拒绝以 256 标签代替"})
                    continue
                scores = camus_metrics.frame_scores(native_prediction, frame.native_label)
                scores.update({"patient": frame.patient, "view": frame.view, "phase": frame.phase,
                               "quality": frame.quality,
                               "frame_id": f"{frame.patient}_{frame.view}_{frame.phase}"})
                records.append(scores)
    if was_training:
        model.train(True)
    metric = camus_metrics.dice_macro_for_selection(records)
    info["null_frames"] = sum(1 for record in records if record.get("foreground_macro_dice") is None)
    return metric, records, info


class LazyFrameBatches:
    """逐 epoch 按需生成训练批次：每 epoch 用 ``seed+epoch`` 重新洗牌/增强，不常驻全部数据。"""

    def __init__(self, records: Sequence[Mapping[str, Any]], *, batch_size: int, seed: int,
                 start_epoch: int = 1) -> None:
        self.records = list(records)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.start_epoch = int(start_epoch)
        self.epoch = self.start_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return (len(self.records) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.records)))
        rng.shuffle(order)
        from . import data as camus_data

        for start in range(0, len(order), self.batch_size):
            images, labels, valids = [], [], []
            for index in order[start:start + self.batch_size]:
                record = self.records[index]
                # Training needs only the canvas fields. Native arrays stay
                # on disk for evaluation and are never decompressed per epoch.
                with np.load(Path(record["cache_file"]), allow_pickle=False) as payload:
                    image = np.asarray(payload["image"], dtype=np.float32)
                    label = np.asarray(payload["label"], dtype=np.int64)
                    valid = np.asarray(payload["valid"], dtype=np.float32)
                image, label, valid = augment_pair(image, label, valid, rng, enable_photometry=True)
                images.append(torch.from_numpy(image)[None])
                labels.append(torch.from_numpy(label))
                valids.append(torch.from_numpy(valid)[None])
            yield (torch.stack(images), torch.stack(labels), torch.stack(valids))


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAMUS 4 类分割训练（需 Codex 批准 + 缓存 QC）")
    parser.add_argument("--approval", required=True, help="Codex 训练批准 JSON")
    parser.add_argument("--data-manifest", required=True, help="prepare 生成的缓存清单 JSON")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--size", type=int, default=TrainConfig.size)
    parser.add_argument("--base-channels", type=int, default=camus_model.DEFAULT_BASE_CHANNELS)
    parser.add_argument("--num-pools", type=int, default=camus_model.DEFAULT_NUM_POOLS)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stop-file", default=None)
    parser.add_argument("--resume", default=None, help="从 last.pt 真实恢复（含 RNG/AMP/epoch）")
    parser.add_argument("--max-seconds", type=int, default=10800,
                        help="墙钟上限；超时安全保存并以 stopped 退出（非成功）")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def cli_main(argv: Sequence[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv)
    cache = require_cache(args.data_manifest, require_combined=True)
    cache_path = Path(args.data_manifest)
    cache_sha256 = sha256_file(cache_path)
    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                         learning_rate=args.learning_rate, patience=args.patience,
                         amp=not args.no_amp, size=args.size, base_channels=args.base_channels,
                         num_pools=args.num_pools, seed=args.seed, out_dir=args.out_dir,
                         stop_file=args.stop_file, approval_file=args.approval,
                         data_manifest=args.data_manifest)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
        raise QCFailed(f"输出目录已存在且非空（全新目录要求）: {out_dir}")
    source_identity = code_identity()
    train_records = [record for record in cache["records"] if record.get("split") == "training"]
    dev_records = [record for record in cache["records"] if record.get("split") == "validation"]
    verify_approval(args.approval, config, source_identity=source_identity,
                    cache_path=cache_path, cache_sha256=cache_sha256, out_dir=out_dir,
                    train_frames=len(train_records), dev_frames=len(dev_records))
    seed_everything(config.seed)
    dev_frames = load_cache_frames(cache_path, "validation")
    if len(dev_frames) != 200:
        raise QCFailed(f"dev 帧数 {len(dev_frames)} != 200，拒绝训练（native Dice 选模要求完整 dev）")
    batches = LazyFrameBatches(train_records, batch_size=config.batch_size, seed=config.seed)
    model = camus_model.build_model(base_channels=config.base_channels, num_pools=config.num_pools,
                                    size=config.size)
    result = fit(batches, dev_frames, config, model, args.device,
                 data_identity={"cache_manifest": cache_path.as_posix(),
                                "cache_sha256": cache_sha256,
                                "train_frames": len(train_records),
                                "dev_frames": len(dev_frames),
                                "split_digest": (cache.get("splits") or {})
                                .get("training", {}).get("split_digest")},
                 source_identity=source_identity,
                 resume_from=Path(args.resume) if args.resume else None,
                 max_seconds=args.max_seconds)
    print(json.dumps({"status": result["status"], "best_metric": result["best_metric"],
                      "out_dir": result["out_dir"]}, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "completed" else 3 if result["status"] in (
        "stopped", "early_stopped") else 1


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m software.role_a_camus.train`` 入口。"""
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
