"""v2 训练器：8 通道 + 训练期同步 90 度整数旋转 + 按轮余弦学习率。

A09 阶段一设计约束：

* 只复用 ``train_overall_v1`` 的数据装配、模型、指标与检查点工具，不复制整份训练器；
* 部署网络完全不变：不新增任何算子，旋转只发生在训练期增强，验证/量化预处理不旋转；
* 每轮原子保存 ``latest.pt``（模型/优化器/随机状态/学习率调度状态/历史/配置/输入/代码哈希）；
* 断点停在 ``phase='trained'`` 时，恢复只补做该轮评估，绝不重复梯度更新。
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

try:  # 兼容 `py -m software.role_a.train_overall_v2` 与直接运行脚本
    from software.role_a import train_overall_v1 as v1
    from software.role_a.data import DatasetItem
    from software.role_a.model import build_model, describe_graph
except ImportError:  # pragma: no cover - 直接运行脚本时的相对导入
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import train_overall_v1 as v1  # type: ignore[no-redef]
    from data import DatasetItem  # type: ignore[no-redef]
    from model import build_model, describe_graph  # type: ignore[no-redef]


__all__ = [
    "TRIAL_NAME",
    "RunConfig",
    "RotatingTrainDataset",
    "cosine_learning_rate",
    "rotation_quarters",
    "rotate_quarters",
    "code_hashes_snapshot",
    "config_from_args",
    "build_parser",
    "main",
    "train",
]

TRIAL_NAME: str = "role_a_overall_v2"
CONFIG_SCHEMA: str = "role-a-overall-v2-config-v1"
PROTOCOL_SCHEMA: str = "role-a-overall-v2-protocol-v1"
# 检查点在磁盘上的字段布局与 v1 完全一致（仅多出 lr/schedule），因此沿用同一 schema
# 字符串，直接复用 v1 的 validate_resume 守卫；v2/v1 的区分由 config payload 与
# config_hash 承担（v2 的 config_hash 永远不可能等于 v1 的，反之亦然）。
CHECKPOINT_SCHEMA: str = v1.CHECKPOINT_SCHEMA
BEST_SCHEMA: str = "role-a-overall-v2-best-v1"
COMPLETION_SCHEMA: str = "role-a-overall-v2-completion-v1"

# 复用 v1 的冻结口径，避免两套语义漂移
CHECKPOINT_NAME = v1.CHECKPOINT_NAME
BEST_NAME = v1.BEST_NAME
COMPLETION_NAME = v1.COMPLETION_NAME
PROTOCOL_NAME = v1.PROTOCOL_NAME
HISTORY_NAME = v1.HISTORY_NAME
SPLIT_NAME = v1.SPLIT_NAME
DEV_EVIDENCE_NAME = v1.DEV_EVIDENCE_NAME
RESUME_LOG_NAME = v1.RESUME_LOG_NAME
THRESHOLDS = v1.THRESHOLDS
PRIMARY_METRIC = v1.PRIMARY_METRIC
SELECTION_RULE = v1.SELECTION_RULE
FINAL_HOLDOUT = v1.FINAL_HOLDOUT
RESUME_SCOPE = v1.RESUME_SCOPE

ROTATION_QUARTERS: int = 4
ROTATION_SALT: int = 0x9E3779B1
LR_FINAL_DEFAULT: float = 5.0e-6
LR_SCHEDULE_NAME: str = "cosine"

CODE_HASH_FILES: tuple[str, ...] = (
    "train_overall_v2.py", "train_overall_v1.py", "model.py", "data.py", "metrics.py",
)


def rotation_quarters(seed: int, epoch: int, index: int) -> int:
    """每张图由 ``(seed, epoch, 样本下标)`` 确定性派生 0/1/2/3 个 90 度。

    与 v1 增强同一套派生规则（``item_seed``），因此不依赖全局 RNG，
    也不随 DataLoader worker 调度顺序变化。
    """
    derived = int(v1.item_seed(int(seed), int(epoch), int(index))) ^ ROTATION_SALT
    return int(random.Random(derived).randrange(ROTATION_QUARTERS))


def cosine_learning_rate(epoch: int, epochs: int, initial: float, final: float) -> float:
    """按轮余弦下降：``epoch=1`` 取 ``initial``，``epoch=epochs`` 取 ``final``。"""
    epochs = int(epochs)
    epoch = int(epoch)
    if epochs < 1:
        raise v1.ConfigError(f"epochs 必须 >= 1，收到 {epochs}")
    if not 1 <= epoch <= epochs:
        raise v1.ConfigError(f"epoch 必须在 1..{epochs}，收到 {epoch}")
    initial, final = float(initial), float(final)
    if initial <= 0 or final <= 0:
        raise v1.ConfigError(f"initial/final 学习率必须 > 0，收到 {initial}/{final}")
    if final > initial:
        raise v1.ConfigError(f"余弦下降要求 final <= initial，收到 {final} > {initial}")
    if epochs == 1:
        return initial
    progress = (epoch - 1) / (epochs - 1)
    return final + 0.5 * (initial - final) * (1.0 + math.cos(math.pi * progress))


def rotate_quarters(item: DatasetItem, quarters: int) -> DatasetItem:
    """把已 letterbox 的方形张量同步旋转 90 度的整数倍。

    旋转同时作用于 image/mask/valid，三者共用同一个 ``k``，因此在方形画布上
    逐像素保持对齐：``valid`` 之外仍然是补边（mask 恒为 0），前景不可能落到
    ``valid == 0`` 的区域。
    """
    k = int(quarters) % ROTATION_QUARTERS
    if k == 0:
        return item
    rotated_mask = (None if item.mask is None
                    else torch.rot90(item.mask, k, dims=(1, 2)).contiguous())
    return replace(
        item,
        image=torch.rot90(item.image, k, dims=(1, 2)).contiguous(),
        mask=rotated_mask,
        valid=torch.rot90(item.valid, k, dims=(1, 2)).contiguous(),
    )


class RotatingTrainDataset(v1.TrainDataset):
    """v1 训练集之上加一层确定性的同步 90 度整数旋转。"""

    def __init__(self, items: Sequence[Any], size: int = 256, seed: int = 0,
                 augment: bool = True, epoch: int = 0, rotation: bool = True) -> None:
        super().__init__(items, size=size, seed=seed, augment=augment, epoch=epoch)
        self.rotation = bool(rotation)

    def quarters(self, index: int) -> int:
        """该样本本轮的旋转量（关闭旋转时恒为 0）。"""
        if not self.rotation:
            return 0
        return rotation_quarters(self.seed, self.epoch, index)

    def __getitem__(self, index: int) -> DatasetItem:
        return rotate_quarters(super().__getitem__(index), self.quarters(index))


@dataclass(frozen=True)
class RunConfig(v1.RunConfig):
    """v2 配置：在 v1 全部字段之上固定加入旋转增强与余弦学习率。"""

    rotation: bool = True
    lr_final: float = LR_FINAL_DEFAULT
    lr_schedule: str = LR_SCHEDULE_NAME

    def payload(self) -> dict[str, Any]:
        payload = dict(v1.RunConfig.payload(self))
        payload["schema"] = CONFIG_SCHEMA
        payload["trial"] = TRIAL_NAME
        payload["rotation"] = bool(self.rotation)
        payload["lr_final"] = float(self.lr_final)
        payload["lr_schedule"] = str(self.lr_schedule)
        payload["augmentation"] = (
            "v2: letterbox + flips + synchronized 90-degree integer rotations + photometry"
            if self.rotation else "v2 ablation: v1 augmentation without rotation")
        payload["lr_plan"] = (
            f"{LR_SCHEDULE_NAME}: {float(self.lr)} -> {float(self.lr_final)} over "
            f"{int(self.epochs)} epochs, one step per epoch, no restart")
        return payload


def _validate_config(config: RunConfig) -> None:
    v1._validate_config(config)
    if str(config.lr_schedule) != LR_SCHEDULE_NAME:
        raise v1.ConfigError(
            f"v2 只实现按轮余弦学习率（{LR_SCHEDULE_NAME}），收到 {config.lr_schedule!r}")
    if float(config.lr_final) <= 0:
        raise v1.ConfigError(f"lr_final 必须 > 0，收到 {config.lr_final}")
    if float(config.lr_final) > float(config.lr):
        raise v1.ConfigError(
            f"余弦下降要求 lr_final <= lr，收到 {config.lr_final} > {config.lr}")
    # 边界必须可求值：第 1 轮与最后一轮
    cosine_learning_rate(1, int(config.epochs), float(config.lr), float(config.lr_final))
    cosine_learning_rate(int(config.epochs), int(config.epochs),
                         float(config.lr), float(config.lr_final))


def _resolve_checkpoint_lr(checkpoint: Mapping[str, Any], config: RunConfig, epoch: int,
                           optimizer: torch.optim.Optimizer) -> float:
    """核对并返回断点所在轮次实际使用的 lr。

    关键约束：**绝不提前把 LR 切到下一轮**。断点落盘时 ``lr``、``schedule.applied_lr``
    与 ``optimizer_state.param_groups[*].lr`` 必须描述同一个轮次，否则补做评估后再次
    中断会写出内部自相矛盾的检查点。
    """
    planned = cosine_learning_rate(int(epoch), int(config.epochs),
                                   config.lr, config.lr_final)
    stored = checkpoint.get("lr")
    if stored is None:
        raise v1.ResumeError("检查点缺少 lr 字段；v2 拒绝在未知学习率下继续")
    stored = float(stored)
    if abs(stored - float(planned)) > 1e-12:
        raise v1.ResumeError(
            f"检查点记录的 lr={stored} 与余弦计划 {planned} 不符；拒绝在未知调度下继续")
    group_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if not group_lrs or any(abs(value - stored) > 1e-12 for value in group_lrs):
        raise v1.ResumeError(
            f"检查点内部学习率不一致：optimizer_state.param_groups[*].lr={group_lrs} "
            f"与 lr={stored} 不符；拒绝继续")
    schedule = checkpoint.get("schedule")
    if isinstance(schedule, Mapping) and "applied_lr" in schedule:
        if abs(float(schedule["applied_lr"]) - stored) > 1e-12:
            raise v1.ResumeError("检查点 schedule.applied_lr 与 lr 不符；拒绝继续")
    return stored


def code_hashes_snapshot() -> dict[str, str]:
    """本试验依赖的全部代码文件 SHA256（含被复用的 v1 训练器）。"""
    role_a = Path(__file__).resolve().parent
    return {name: v1.sha256_file(role_a / name) for name in CODE_HASH_FILES}


def _schedule_state(config: RunConfig, *,
                    initial_lr: float) -> dict[str, Any]:
    return {"name": LR_SCHEDULE_NAME, "initial": float(config.lr),
            "final": float(config.lr_final), "epochs": int(config.epochs),
            "step": "per-epoch", "restart": False, "applied_lr": float(initial_lr)}


def _set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> float:
    value = float(value)
    for group in optimizer.param_groups:
        group["lr"] = value
    return value


def train(config: RunConfig) -> dict[str, Any]:
    """执行（或继续）一次 v2 训练。返回简短状态字典。"""
    _validate_config(config)
    output_dir = Path(config.output_dir)
    mode = v1.plan_mode(output_dir, config.resume)

    if mode == "complete":
        if config.resume:
            print(f"[overall-v2] {output_dir} 已有完成标记 {COMPLETION_NAME}；"
                  "--resume 不重复训练，也不覆盖既有产物。", flush=True)
            return {"status": "already-complete", "output_dir": output_dir.as_posix()}
        raise v1.TrialStateError(
            f"{output_dir} 已存在完成标记 {COMPLETION_NAME}；拒绝静默覆盖已完成试验。"
            "如需重跑，请另建输出目录。")
    if mode == "partial":
        raise v1.TrialStateError(
            f"{output_dir} 存在未完成检查点 {CHECKPOINT_NAME}；请显式使用 --resume 继续，"
            "脚本不会静默覆盖。")
    if mode == "unknown":
        raise v1.TrialStateError(
            f"{output_dir} 非空但没有 {CHECKPOINT_NAME} / {COMPLETION_NAME}；"
            "无法判断试验状态，拒绝写入。")
    if mode == "missing-checkpoint":
        raise v1.ResumeError(f"--resume 需要 {output_dir / CHECKPOINT_NAME}，但该文件不存在。")

    if config.init in ("baseline", "checkpoint"):
        starting = (config.data_spec.baseline_checkpoint if config.init == "baseline"
                    else config.init_checkpoint)
        if starting is None or not Path(starting).is_file():
            raise v1.ResumeError(f"--init {config.init} 需要已存在的起点权重: {starting}")

    if mode == "fresh":
        v1.seed_everything(config.seed)

    device = v1.resolve_device(config.device)
    assembly = v1.assemble(config.data_spec, verify_frozen=True, verify_images=config.verify_images)
    input_hashes = dict(assembly.hashes)
    if config.init == "checkpoint":
        assert config.init_checkpoint is not None
        input_hashes["init_checkpoint"] = v1.sha256_file(config.init_checkpoint)
    code_hashes = code_hashes_snapshot()
    payload = config.payload()
    config_hash = v1.config_digest(payload)

    model = build_model(base_channels=config.base_channels, num_pools=config.num_pools,
                        input_shape=(3, config.size, config.size)).to(device)
    graph = describe_graph(model)
    if not bool(graph["allowed_only"]):
        raise v1.ConfigError(f"模型含有不允许的算子 {graph['forbidden']}，会破坏 INT8 可映射性")
    estimates = v1.estimate_model(config.base_channels, config.num_pools, config.size)
    if int(estimates["parameters"]) != int(model.num_parameters()):
        raise v1.ConfigError(
            f"静态参数估算 {estimates['parameters']} 与 torch 模型 "
            f"{model.num_parameters()} 不一致；拒绝据此写协议")
    model.assert_shapes(batch_size=1, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=tuple(config.betas),
                                 eps=config.eps, weight_decay=config.weight_decay)

    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    start_epoch = 1
    pending_eval_epoch: int | None = None
    pending_train_metrics: dict[str, float] | None = None
    source_checkpoint: dict[str, Any] | None = None
    current_lr = float(config.lr)

    if mode == "fresh":
        if config.init in ("baseline", "checkpoint"):
            starting_path = (config.data_spec.baseline_checkpoint if config.init == "baseline"
                             else config.init_checkpoint)
            if starting_path is None:
                raise v1.ResumeError("--init baseline 缺少起点权重路径")
            source_checkpoint = v1.load_baseline_weights(
                starting_path, model,
                base_channels=config.base_channels, num_pools=config.num_pools,
                size=config.size)
            print(f"[overall-v2] 从已审查权重起步: {source_checkpoint['source_checkpoint']} "
                  f"(epoch={source_checkpoint['source_epoch']}, "
                  f"sha256={source_checkpoint['source_checkpoint_sha256'][:12]}...)", flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        v1.atomic_write_json(output_dir / PROTOCOL_NAME, {
            "schema": PROTOCOL_SCHEMA,
            "trial": TRIAL_NAME,
            "created_at": v1._timestamp(),
            "config": payload,
            "config_hash": config_hash,
            "input_hashes": input_hashes,
            "code_hashes": code_hashes,
            "source_checkpoint": source_checkpoint,
            "sources": list(assembly.source_records),
            "excluded_ids": {key: list(value) for key, value in assembly.exclusions.items()},
            "counts": dict(assembly.counts),
            "split_digest": dict(assembly.split_digest),
            "model": {**estimates, "parameters_torch": int(model.num_parameters()),
                      "layer_count": len(graph["layers"]),
                      "allowed_only": bool(graph["allowed_only"]),
                      "forbidden": list(graph["forbidden"])},
            "optimizer": {
                "name": "torch.optim.Adam",
                "lr": float(config.lr),
                "lr_final": float(config.lr_final),
                "lr_schedule": {"name": LR_SCHEDULE_NAME, "step": "per-epoch",
                                "restart": False},
                "betas": [float(value) for value in config.betas],
                "eps": float(config.eps),
                "weight_decay": float(config.weight_decay),
                "grad_clip": float(config.grad_clip),
                "bce_weight": float(config.bce_weight),
                "dice_weight": float(config.dice_weight),
                "batch_size": int(config.batch_size),
                "epochs": int(config.epochs),
                "seed": int(config.seed),
                "augment": bool(config.augment),
                "rotation": bool(config.rotation),
                "num_workers": int(config.num_workers),
                "device": str(device),
            },
            "augmentation": {
                "train": payload["augmentation"],
                "rotation_quarters": ROTATION_QUARTERS,
                "rotation_derivation": "random.Random(item_seed(seed, epoch, index) ^ 0x9E3779B1)",
                "evaluation_and_quantization": "no augmentation, no rotation (v1 preprocessing)",
            },
            "thresholds": [float(value) for value in config.thresholds],
            "primary_metric": PRIMARY_METRIC,
            "selection_rule": SELECTION_RULE,
            "resume_scope": RESUME_SCOPE,
            "image_hash_checks": int(assembly.image_hash_checks),
            "final_holdout": FINAL_HOLDOUT,
            "known_limits": [
                "验证集/开发集指标空间是 letterbox 后的 NxN 有效区域，不是原始分辨率",
                "GPU 算子未强制确定性；换设备不承诺逐位一致",
                "params/MAC/峰值特征为算术估算，不是实测资源、时序或功耗",
                "旋转 + 余弦学习率是一个整体方案，改善不能单独归因于其中任一项",
            ],
        })
        v1.atomic_write_json(output_dir / SPLIT_NAME, {
            "train_ids": list(assembly.train_ids),
            "development_ids": {name: list(ids) for name, ids in assembly.dev_ids.items()},
            "digests": dict(assembly.split_digest),
            "note": "训练与开发 ID 全局互斥；开发样本用于每轮评价、统一阈值及最优轮次选择，不参与梯度训练",
        })
        print(f"[overall-v2] 装配完成: 训练 {len(assembly.train_items)}，"
              f"开发 {sum(len(d.items) for d in assembly.dev_sets.values())}，"
              f"config_hash={config_hash[:12]}...", flush=True)
    else:
        checkpoint = v1.load_checkpoint(output_dir / CHECKPOINT_NAME)
        epoch, phase = v1.validate_resume(checkpoint, config_hash=config_hash,
                                          input_hashes=input_hashes, code_hashes=code_hashes,
                                          epochs=config.epochs)
        v1.restore_model_state(model, checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = list(checkpoint["history"])
        stored_best = checkpoint.get("best")
        best = dict(stored_best) if isinstance(stored_best, Mapping) else None
        start_epoch = epoch + 1
        pending_eval_epoch = epoch if phase == "trained" else None
        stored_metrics = checkpoint.get("train_metrics")
        if pending_eval_epoch is not None:
            if not isinstance(stored_metrics, Mapping):
                raise v1.ResumeError(
                    "检查点停在评估前但缺少 train_metrics，无法补做该轮历史；拒绝继续")
            pending_train_metrics = {key: float(stored_metrics[key])
                                     for key in ("loss", "bce", "soft_dice")}
        v1.restore_rng_state(checkpoint["rng"])
        protocol_path = output_dir / PROTOCOL_NAME
        if not protocol_path.is_file():
            raise v1.ResumeError(f"缺少 {protocol_path}，无法核对协议")
        if v1._read_json(protocol_path).get("config_hash") != config_hash:
            raise v1.ResumeError("protocol.json 的配置哈希与当前配置不符")
        stored_source = checkpoint.get("source_checkpoint")
        if stored_source is not None and not isinstance(stored_source, Mapping):
            raise v1.ResumeError("检查点记录的 source_checkpoint 结构异常；拒绝继续")
        source_checkpoint = dict(stored_source) if isinstance(stored_source, Mapping) else None
        protocol_source = v1._read_json(protocol_path).get("source_checkpoint")
        if isinstance(source_checkpoint, Mapping) and isinstance(protocol_source, Mapping):
            if protocol_source.get("source_checkpoint_sha256") != \
                    source_checkpoint.get("source_checkpoint_sha256"):
                raise v1.ResumeError("检查点与 protocol.json 记录的起点权重来源不符；拒绝继续")
        # 断点内 lr/schedule/optimizer_state 必须自洽；下一轮的 lr 只在训练前切换
        current_lr = _resolve_checkpoint_lr(checkpoint, config, epoch, optimizer)
        next_lr = cosine_learning_rate(
            start_epoch if start_epoch <= int(config.epochs) else int(config.epochs),
            int(config.epochs), config.lr, config.lr_final)
        v1._append_json_log(output_dir / RESUME_LOG_NAME, {
            "resumed_at": v1._timestamp(),
            "checkpoint_epoch": int(epoch),
            "checkpoint_phase": str(phase),
            "next_epoch": int(start_epoch),
            "pending_eval_epoch": pending_eval_epoch,
            "restored_lr": float(current_lr),
            "next_lr": float(next_lr),
            "config_hash": config_hash,
            "input_hashes": dict(input_hashes),
            "code_hashes": dict(code_hashes),
            "resume_scope": RESUME_SCOPE,
        })
        print(f"[overall-v2] --resume: 检查点 epoch={epoch} phase={phase}；"
              f"下一轮={start_epoch} restored_lr={current_lr:.8f} "
              f"next_lr={next_lr:.8f}", flush=True)

    def persist(epoch: int, phase: str, lr: float,
                train_metrics: Mapping[str, float] | None = None) -> None:
        v1.atomic_torch_save({
            "schema": CHECKPOINT_SCHEMA,
            "created_at": v1._timestamp(),
            "trial": TRIAL_NAME,
            "epoch": int(epoch),
            "phase": str(phase),
            "lr": float(lr),
            "schedule": _schedule_state(config, initial_lr=lr),
            "config": payload,
            "config_hash": config_hash,
            "input_hashes": dict(input_hashes),
            "code_hashes": dict(code_hashes),
            "device": str(device),
            "init": config.init,
            "source_checkpoint": source_checkpoint,
            "model_state": {name: value.detach().cpu().clone()
                            for name, value in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "optimizer_class": "torch.optim.Adam",
            "rng": v1.capture_rng_state(),
            "history": list(history),
            "best": best,
            "train_metrics": (None if train_metrics is None
                              else {key: float(train_metrics[key])
                                    for key in ("loss", "bce", "soft_dice")}),
            "resume_scope": RESUME_SCOPE,
        }, output_dir / CHECKPOINT_NAME)
        print(f"[overall-v2] 原子检查点已保存: epoch={epoch} phase={phase} lr={float(lr):.8f}",
              flush=True)

    def evaluate_and_maybe_save_best(epoch: int
                                     ) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
        nonlocal best
        records = v1.evaluate_development(model, assembly.dev_sets, size=config.size,
                                          batch_size=config.batch_size, num_workers=0,
                                          thresholds=config.thresholds, device=device)
        report = v1.build_development_report(records, assembly.dev_sets, config.thresholds)
        entry = v1.best_entry(epoch, report, config.thresholds)
        improved = v1._better_than(entry, best)
        if improved:
            best = entry
            v1.atomic_torch_save({
                "schema": BEST_SCHEMA,
                "status": "candidate",
                "completed": False,
                "created_at": v1._timestamp(),
                "trial": TRIAL_NAME,
                "epoch": int(epoch),
                "config": payload,
                "config_hash": config_hash,
                "input_hashes": dict(input_hashes),
                "code_hashes": dict(code_hashes),
                "source_checkpoint": source_checkpoint,
                "base_channels": int(config.base_channels),
                "num_pools": int(config.num_pools),
                "size": int(config.size),
                "model_state": {name: value.detach().cpu().clone()
                                for name, value in model.state_dict().items()},
                "primary_metric": PRIMARY_METRIC,
                "selection_rule": SELECTION_RULE,
                "selected_threshold": float(entry["selected_threshold"]),
                "primary_dice": float(entry["primary_dice"]),
                "primary_iou": float(entry["primary_iou"]),
                "source_dice": dict(entry["source_dice"]),
                "sources": entry["sources"],
                "area_groups_pooled": entry["area_groups_pooled"],
                "development_images": int(entry["development_images"]),
                "threshold_dice": dict(entry["threshold_dice"]),
                "optimizer_state_stored": False,
                "resume_supported": False,
                "note": ("best.pt 由主指标（638 张逐图 Dice 平均）选择，且只是候选；"
                         "唯一完成标记是 completed.json。恢复请用 latest.pt。"),
            }, output_dir / BEST_NAME)
            v1.atomic_write_json(output_dir / DEV_EVIDENCE_NAME, {
                "epoch": int(epoch),
                "selected_threshold": float(entry["selected_threshold"]),
                "primary_dice": float(entry["primary_dice"]),
                "development_images": int(entry["development_images"]),
                "records": v1.dev_evidence(records, entry["selected_threshold"],
                                           config.thresholds),
            })
        return report, records, improved

    if pending_eval_epoch is not None:
        print(f"[overall-v2] 检查点停在 epoch={pending_eval_epoch} 的评估前，补做该轮评估"
              "（不重跑梯度）", flush=True)
        report, _, improved = evaluate_and_maybe_save_best(pending_eval_epoch)
        if pending_train_metrics is None:  # pragma: no cover - 上面已强制校验
            raise v1.ResumeError("检查点停在评估前但缺少 train_metrics，无法补写历史")
        epoch_lr = float(current_lr)  # 该轮训练时真实使用的 lr
        history[:] = [row for row in history if int(row["epoch"]) != int(pending_eval_epoch)]
        row = v1._history_row(pending_eval_epoch, pending_train_metrics, report,
                              config.thresholds, improved=bool(improved))
        row["lr"] = epoch_lr
        history.append(row)
        v1.atomic_write_json(output_dir / HISTORY_NAME, history)
        persist(pending_eval_epoch, "evaluated", epoch_lr, pending_train_metrics)

    for epoch in range(start_epoch, int(config.epochs) + 1):
        current_lr = _set_learning_rate(
            optimizer, cosine_learning_rate(epoch, config.epochs,
                                            config.lr, config.lr_final))
        dataset = RotatingTrainDataset(assembly.train_items, size=config.size,
                                       seed=config.seed, augment=config.augment,
                                       epoch=epoch, rotation=config.rotation)
        loader = v1.make_train_loader(dataset, config.batch_size, config.num_workers,
                                      config.seed, epoch)
        train_metrics = v1.run_train_epoch(model, loader, optimizer, device,
                                           bce_weight=config.bce_weight,
                                           dice_weight=config.dice_weight,
                                           grad_clip=config.grad_clip)
        persist(epoch, "trained", current_lr, train_metrics)
        report, _, improved = evaluate_and_maybe_save_best(epoch)
        history[:] = [row for row in history if int(row["epoch"]) != int(epoch)]
        row = v1._history_row(epoch, train_metrics, report, config.thresholds,
                              improved=bool(improved))
        row["lr"] = float(current_lr)
        history.append(row)
        v1.atomic_write_json(output_dir / HISTORY_NAME, history)
        persist(epoch, "evaluated", current_lr)
        print(f"[overall-v2] epoch {epoch}/{config.epochs} lr={current_lr:.8f} "
              f"train_loss={row['train_loss']:.6f} primary_dice={row['primary_dice']:.6f} "
              f"threshold={row['selected_threshold']:.2f} improved={row['improved']}", flush=True)

    completion = {
        "schema": COMPLETION_SCHEMA,
        "trial": TRIAL_NAME,
        "completed_at": v1._timestamp(),
        "completed_epochs": int(config.epochs),
        "config_hash": config_hash,
        "primary_metric": PRIMARY_METRIC,
        "selection_rule": SELECTION_RULE,
        "lr_schedule": {"name": LR_SCHEDULE_NAME, "initial": float(config.lr),
                        "final": float(config.lr_final), "epochs": int(config.epochs),
                        "step": "per-epoch", "restart": False},
        "rotation": bool(config.rotation),
        "best": (None if best is None else {
            key: best[key] for key in ("epoch", "selected_threshold", "primary_dice",
                                       "primary_iou", "source_dice", "sources",
                                       "area_groups_pooled", "development_images",
                                       "threshold_dice", "selection_rule")
        }),
        "best_checkpoint": (output_dir / BEST_NAME).as_posix(),
        "resume_supported": True,
        "resume_scope": RESUME_SCOPE,
        "final_holdout": FINAL_HOLDOUT,
        "note": ("completed.json 是唯一的完成标记；在此之前 best.pt 的 status 一直是 "
                 "candidate，异常中断不会把它标成已完成。"),
    }
    v1.atomic_write_json(output_dir / COMPLETION_NAME, completion)
    print(f"[overall-v2] 训练完成: best epoch="
          f"{None if best is None else best['epoch']} -> {output_dir / COMPLETION_NAME}",
          flush=True)
    return {"status": "completed", "output_dir": output_dir.as_posix(),
            "best": completion["best"], "completed_epochs": int(config.epochs)}


def build_parser() -> argparse.ArgumentParser:
    parser = v1.build_parser()
    parser.description = "v2 训练器：8 通道 + 训练期同步 90 度整数旋转 + 按轮余弦学习率"
    parser.add_argument("--no-rotation", action="store_true",
                        help="消融用：关闭训练期同步 90 度旋转（默认开启）")
    parser.add_argument("--lr-final", type=float, default=LR_FINAL_DEFAULT,
                        help=f"余弦学习率终点，默认 {LR_FINAL_DEFAULT}")
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    base = v1.config_from_args(args)
    values: dict[str, Any] = dict(base.__dict__)
    values["rotation"] = not bool(args.no_rotation)
    values["lr_final"] = float(args.lr_final)
    values["lr_schedule"] = LR_SCHEDULE_NAME
    return RunConfig(**values)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        config = config_from_args(args)
        train(config)
        return 0
    except (v1.AssemblyError, v1.ConfigError, v1.ResumeError, v1.TrialStateError,
            FileNotFoundError, ValueError) as error:
        print(f"[错误] {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())