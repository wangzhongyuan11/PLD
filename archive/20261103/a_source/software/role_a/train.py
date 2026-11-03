"""A 组 FP32 训练 / 划分 / 测试评估 CLI。

三个互不混淆的子命令：

``prepare-split``
    扫描本地 ISIC 2016 官方**训练集**目录，做固定种子的 train/val 划分，
    把排序后的文件名写入 JSON 与 CSV。
``train``
    只在该划分上训练，按**验证集** Dice 选择最佳 checkpoint。
``evaluate-test``
    显式单独命令，用指定 checkpoint 评估一个**独立**测试目录。它只写评估结果，
    不参与、也不影响最佳模型选择。

设计约定：

* 训练损失只统计 ``valid`` 区域（letterbox 的非补边像素）；
* 划分文件里出现 ``test`` 角色时，会校验它与 train/val 不重叠；
* checkpoint 只存张量与基本类型（``weights_only=True`` 可加载），不依赖 pickle
  反序列化任意对象；
* 本文件不下载数据、不声称任何精度。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:  # 允许 `python software/role_a/train.py` 与 `python -m software.role_a.train` 两种调用方式
    from .data import (
        DEFAULT_SEED,
        DataLayoutError,
        ISICDataset,
        IsicPaths,
        enumerate_isic_pairs,
        export_split,
        export_split_csv,
        isic_collate,
        load_split,
        render_split,
    )
    from .metrics import MetricAccumulator, binary_scores, segmentation_loss
    from .model import DEFAULT_BASE_CHANNELS, TinyUNet, build_model, describe_graph
except ImportError:  # pragma: no cover - 直接把本目录加入 sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data import (  # type: ignore[no-redef]
        DEFAULT_SEED,
        DataLayoutError,
        ISICDataset,
        IsicPaths,
        enumerate_isic_pairs,
        export_split,
        export_split_csv,
        isic_collate,
        load_split,
        render_split,
    )
    from metrics import (  # type: ignore[no-redef]
        MetricAccumulator,
        binary_scores,
        segmentation_loss,
    )
    from model import (  # type: ignore[no-redef]
        DEFAULT_BASE_CHANNELS,
        TinyUNet,
        build_model,
        describe_graph,
    )

__all__ = [
    "main",
    "resolve_device",
    "seed_everything",
    "run_epoch",
    "evaluate",
    "config_to_json",
]

EPOCH_FIELDS: tuple[str, ...] = (
    "epoch",
    "train_loss",
    "train_bce",
    "train_soft_dice",
    "val_loss",
    "val_bce",
    "val_soft_dice",
    "val_dice_mean",
    "val_dice_median",
    "val_iou_mean",
    "val_iou_median",
    "val_dice_mean_non_empty_target",
    "val_empty_target_images",
    "val_empty_prediction_images",
    "is_best",
)


# ------------------------------------------------------------------ 基础设施
def seed_everything(seed: int) -> None:
    """固定 python / numpy / torch（含 CUDA）随机源；GPU 算子不保证逐位复现。"""
    if seed < 0:
        raise ValueError(f"seed 必须 >= 0，收到 {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """解析 ``--device``：``auto`` / ``cpu`` / ``cuda`` / ``cuda:0`` / ``mps``。

    异常:
        ValueError: 设备字符串无法识别，或请求的加速器不可用（不静默降级）。
    """
    name = requested.strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"无法识别的 device: {requested!r}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求了 CUDA 设备，但 torch.cuda.is_available() 为 False")
    if device.type == "mps" and not getattr(torch.backends, "mps", None):
        raise ValueError("请求了 mps 设备，但当前 torch 未提供 mps 后端")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("请求了 mps 设备，但当前机器不可用")
    return device


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def config_to_json(config: Mapping[str, Any]) -> str:
    """把配置（含 Path）序列化成稳定的 JSON 文本。"""
    return json.dumps(_jsonable(config), indent=2, ensure_ascii=False, sort_keys=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False),
                    encoding="utf-8")


# ------------------------------------------------------------------- 数据集
def _items_for(
    split: Mapping[str, Sequence[str]],
    role: str,
    pairs: Sequence[IsicPaths],
    allow_missing_mask: bool,
) -> list[IsicPaths]:
    """按划分角色取出配对的 :class:`IsicPaths`，缺项立即报错。"""
    if role not in split:
        raise DataLayoutError(f"划分中没有 {role} 角色")
    index = {item.stem: item for item in pairs}
    selected: list[IsicPaths] = []
    missing: list[str] = []
    for stem in sorted(split[role]):
        item = index.get(stem)
        if item is None:
            missing.append(stem)
            continue
        if not item.has_mask and not allow_missing_mask:
            raise DataLayoutError(f"{role} 中的 {stem} 缺少掩膜，不能用于训练或验证")
        selected.append(item)
    if missing:
        raise DataLayoutError(
            f"划分 {role} 中有 {len(missing)} 个文件名在数据目录里找不到: {missing[:10]}"
        )
    if not selected:
        raise DataLayoutError(f"划分 {role} 为空")
    return selected


def _worker_init(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's per-worker seed (Windows pickle safe)."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(
    items: Sequence[IsicPaths],
    root: Path,
    size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    augment: bool = False,
) -> DataLoader:
    """构造 DataLoader。``num_workers > 0`` 时用派生种子保证每个 worker 可复现。"""
    if batch_size < 1:
        raise ValueError(f"batch_size 必须 >= 1，收到 {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers 必须 >= 0，收到 {num_workers}")
    dataset = ISICDataset(items, root=root, size=size, augment=augment)
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=isic_collate,
        generator=generator if shuffle else None,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        drop_last=False,
    )


# --------------------------------------------------------------------- 训练
def run_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    grad_clip: float = 5.0,
) -> dict[str, float]:
    """跑一个 epoch。``optimizer`` 为 ``None`` 时是纯推理（验证/测试）。"""
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "bce": 0.0, "soft_dice": 0.0}
    batches = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"]
        if masks is None:
            raise DataLayoutError("训练/验证 batch 缺少掩膜")
        masks = masks.to(device, non_blocking=True)
        valids = batch["valid"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss, parts = segmentation_loss(logits, masks, valids,
                                            bce_weight=bce_weight, dice_weight=dice_weight)
        if training:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["bce"] += parts["bce"]
        totals["soft_dice"] += parts["soft_dice"]
        batches += 1
    if batches == 0:
        raise DataLayoutError("loader 为空，没有产生任何 batch")
    return {key: value / batches for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    threshold: float = 0.5,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    records: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """逐图计算 Dice/IoU 再取均值，同时返回损失分项。"""
    model.eval()
    accumulator = MetricAccumulator()
    totals = {"loss": 0.0, "bce": 0.0, "soft_dice": 0.0}
    batches = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"]
        if masks is None:
            raise DataLayoutError("评估 batch 缺少掩膜；纯图像推理请使用 evaluate 之外的工具")
        masks = masks.to(device, non_blocking=True)
        valids = batch["valid"].to(device, non_blocking=True)
        logits = model(images)
        loss, parts = segmentation_loss(logits, masks, valids,
                                        bce_weight=bce_weight, dice_weight=dice_weight)
        totals["loss"] += float(loss.detach())
        totals["bce"] += parts["bce"]
        totals["soft_dice"] += parts["soft_dice"]
        batches += 1
        for index in range(logits.shape[0]):
            scores = binary_scores(logits[index:index + 1], masks[index:index + 1],
                                   valids[index:index + 1], threshold=threshold)
            scores["filename"] = batch["filename"][index]
            accumulator.update(scores)
            if records is not None:
                records.append({key: scores[key] for key in
                                ("filename", "dice", "iou", "empty_target",
                                 "empty_prediction", "valid_pixels")})
    if batches == 0:
        raise DataLayoutError("loader 为空，没有产生任何 batch")
    metrics: dict[str, float] = {key: value / batches for key, value in totals.items()}
    summary = accumulator.summary()
    metrics.update({key: value for key, value in summary.items() if key != "images"})
    return metrics, summary


# ------------------------------------------------------------------ 子命令
def command_prepare_split(args: argparse.Namespace) -> int:
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    pairs = enumerate_isic_pairs(image_dir, mask_dir, require_mask=True)
    root = Path(args.data_root) if args.data_root else image_dir
    split = render_split(pairs, seed=args.seed, val_fraction=args.val_fraction,
                         stratify_by_area=not args.no_stratify, report_dir=args.output_dir)

    test_stems: list[str] = []
    test_pairs: list[IsicPaths] = []
    test_note: str | None = None
    if args.include_test:
        # 只登记官方测试集文件名用于最终评估；require_mask=False，因为测试标注
        # 是否可得需单独确认。test 角色绝不参与训练或选模型。
        test_pairs = enumerate_isic_pairs(args.test_image_dir, args.test_mask_dir,
                                          require_mask=False)
        test_stems = [item.stem for item in test_pairs]
        missing_masks = sum(1 for item in test_pairs if not item.has_mask)
        test_note = (f"官方测试集 {len(test_stems)} 张，其中 {missing_masks} 张当前没有掩膜；"
                     "evaluate-test 需要掩膜才能算 Dice/IoU")
        split = dict(split)
        split["test"] = test_stems

    output_dir = Path(args.output_dir)
    split_path = export_split(
        split, output_dir / "split.json", seed=args.seed, pairs=[*pairs, *test_pairs],
        extra={
            "image_dir": image_dir.resolve().as_posix(),
            "mask_dir": (mask_dir.resolve().as_posix() if mask_dir else None),
            "val_fraction": args.val_fraction,
            "stratify_by_area": not args.no_stratify,
            "test_image_dir": (Path(args.test_image_dir).resolve().as_posix() if args.include_test else None),
            "test_note": test_note,
        },
    )
    csv_path = export_split_csv(split, output_dir / "split.csv",
                                pairs=[*pairs, *test_pairs], root=root)
    report = {
        "image_dir": image_dir.as_posix(),
        "mask_dir": (mask_dir.as_posix() if mask_dir else image_dir.as_posix()),
        "pairs": len(pairs),
        "train": len(split["train"]),
        "val": len(split["val"]),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "split_json": split_path.as_posix(),
        "split_csv": csv_path.as_posix(),
        "test_note": test_note,
        "note": "只扫描官方训练集目录做 train/val；test 角色只登记文件名，不参与训练。",
    }
    if args.include_test:
        report["test"] = len(test_stems)
    print(json.dumps(_jsonable(report), indent=2, ensure_ascii=False))
    return 0


def build_run_config(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    return {
        "image_dir": Path(args.image_dir).as_posix(),
        "mask_dir": (Path(args.mask_dir).as_posix() if args.mask_dir else Path(args.image_dir).as_posix()),
        "split": Path(args.split).as_posix(),
        "output_dir": Path(args.output_dir).as_posix(),
        "size": args.size,
        "base_channels": args.base_channels,
        "num_pools": args.num_pools,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "device": str(device),
        "num_workers": args.num_workers,
        "threshold": args.threshold,
        "bce_weight": args.bce_weight,
        "dice_weight": args.dice_weight,
        "grad_clip": args.grad_clip,
        "augment": args.augment,
    }


def command_train(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    if args.epochs < 1:
        raise ValueError(f"epochs 必须 >= 1，收到 {args.epochs}")
    if args.lr <= 0:
        raise ValueError(f"lr 必须 > 0，收到 {args.lr}")
    seed_everything(args.seed)

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    pairs = enumerate_isic_pairs(image_dir, mask_dir, require_mask=True)
    split = load_split(args.split)
    if "test" in split:
        raise DataLayoutError(
            "训练用划分文件不允许包含 test 角色；请为测试集单独使用 evaluate-test"
        )
    train_items = _items_for(split, "train", pairs, allow_missing_mask=False)
    val_items = _items_for(split, "val", pairs, allow_missing_mask=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_run_config(args, device)
    (output_dir / "config.json").write_text(config_to_json(config), encoding="utf-8")
    (output_dir / "split_used.json").write_text(
        json.dumps({"seed": args.seed,
                    "train": sorted(split["train"]),
                    "val": sorted(split["val"])}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    _write_json(output_dir / "model_graph.json", describe_graph(
        build_model(base_channels=args.base_channels, num_pools=args.num_pools,
                    input_shape=(3, args.size, args.size))))

    model = build_model(base_channels=args.base_channels, num_pools=args.num_pools,
                        input_shape=(3, args.size, args.size)).to(device)
    model.assert_shapes(batch_size=1, device=device)
    graph = describe_graph(model)
    if not graph["allowed_only"]:
        raise ValueError(f"模型包含不允许的算子 {graph['forbidden']}，会破坏 INT8 可映射性")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = make_loader(train_items, image_dir, args.size, args.batch_size,
                               shuffle=True, seed=args.seed, num_workers=args.num_workers,
                               augment=args.augment)
    val_loader = make_loader(val_items, image_dir, args.size, args.batch_size,
                             shuffle=False, seed=args.seed, num_workers=args.num_workers)

    best_dice = float("-inf")
    best_epoch = 0
    rows: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer,
                                  bce_weight=args.bce_weight, dice_weight=args.dice_weight,
                                  grad_clip=args.grad_clip)
        val_metrics, _ = evaluate(model, val_loader, device, threshold=args.threshold,
                                  bce_weight=args.bce_weight, dice_weight=args.dice_weight)
        is_best = val_metrics["dice_mean"] > best_dice
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_bce": train_metrics["bce"],
            "train_soft_dice": train_metrics["soft_dice"],
            "val_loss": val_metrics["loss"],
            "val_bce": val_metrics["bce"],
            "val_soft_dice": val_metrics["soft_dice"],
            "val_dice_mean": val_metrics["dice_mean"],
            "val_dice_median": val_metrics["dice_median"],
            "val_iou_mean": val_metrics["iou_mean"],
            "val_iou_median": val_metrics["iou_median"],
            "val_dice_mean_non_empty_target": val_metrics["dice_mean_non_empty_target"],
            "val_empty_target_images": val_metrics["empty_target_images"],
            "val_empty_prediction_images": val_metrics["empty_prediction_images"],
            "is_best": int(is_best),
        }
        rows.append(row)
        print(f"epoch {epoch:>3}/{args.epochs} train_loss={train_metrics['loss']:.6f} "
              f"val_loss={val_metrics['loss']:.6f} val_dice={val_metrics['dice_mean']:.6f} "
              f"val_iou={val_metrics['iou_mean']:.6f}{'  (best)' if is_best else ''}")
        if is_best:
            best_dice = float(val_metrics["dice_mean"])
            best_epoch = epoch
            checkpoint = {
                "model_state": {key: value.detach().cpu()
                                for key, value in model.state_dict().items()},
                "epoch": epoch,
                "val_metrics": {key: float(value) for key, value in val_metrics.items()},
                "config": config,
                "split_file": Path(args.split).as_posix(),
                "best_metric": "val_dice_mean",
                "best_value": best_dice,
                "optimizer_state_stored": False,
                "resume_supported": False,
            }
            torch.save(checkpoint, output_dir / "best.pt")
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(EPOCH_FIELDS))
            writer.writeheader()
            for result in rows:
                writer.writerow(result)
    summary = {
        "best_epoch": best_epoch,
        "best_val_dice_mean": best_dice,
        "epochs": args.epochs,
        "checkpoint": (output_dir / "best.pt").as_posix(),
        "config": config,
        "last_epoch": rows[-1],
        "selection_rule": "只按验证集 dice_mean 选最佳；evaluate-test 不参与选择",
    }
    _write_json(output_dir / "metrics.json", summary)
    print(json.dumps(_jsonable({"best_epoch": best_epoch, "best_val_dice_mean": best_dice,
                                "output_dir": output_dir.as_posix()}), indent=2,
                     ensure_ascii=False))
    return 0


def command_evaluate_test(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    seed_everything(args.seed)
    split_path = Path(args.split)
    if not split_path.is_file():
        raise DataLayoutError(
            f"测试划分文件不存在: {split_path}；请先用 prepare-split --include-test 生成"
        )
    split = load_split(split_path)
    if "test" not in split:
        raise DataLayoutError(
            f"{split_path} 中没有 test 角色；evaluate-test 不会偷偷使用 val。"
            "请用 prepare-split --include-test --test-image-dir ... 生成"
        )
    image_dir = Path(args.test_image_dir) if args.test_image_dir else Path(args.image_dir)
    mask_dir = Path(args.test_mask_dir) if args.test_mask_dir else (
        Path(args.mask_dir) if args.mask_dir else None)
    pairs = enumerate_isic_pairs(image_dir, mask_dir, require_mask=True)
    items = _items_for(split, "test", pairs, allow_missing_mask=False)
    loader = make_loader(items, image_dir, args.size, args.batch_size, shuffle=False,
                         seed=args.seed, num_workers=args.num_workers)

    if not Path(args.checkpoint).is_file():
        raise DataLayoutError(f"checkpoint 不存在: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "model_state" not in checkpoint or "config" not in checkpoint:
        raise DataLayoutError(f"checkpoint 缺少 model_state/config 字段: {args.checkpoint}")
    train_config = checkpoint["config"]
    if int(train_config.get("size", args.size)) != args.size:
        raise DataLayoutError(
            f"checkpoint 训练分辨率 {train_config.get('size')} 与当前 --size {args.size} 不一致"
        )
    model = build_model(base_channels=int(train_config.get("base_channels", args.base_channels)),
                        num_pools=int(train_config.get("num_pools", args.num_pools)),
                        input_shape=(3, args.size, args.size)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    records: list[dict[str, Any]] = []
    metrics, summary = evaluate(model, loader, device, threshold=args.threshold,
                                bce_weight=args.bce_weight, dice_weight=args.dice_weight,
                                records=records)
    report = {
        "checkpoint": Path(args.checkpoint).as_posix(),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_selected_by": checkpoint.get("best_metric", "val_dice_mean"),
        "checkpoint_val_value": checkpoint.get("best_value"),
        "test_image_dir": image_dir.as_posix(),
        "test_split": split_path.as_posix(),
        "test_images": summary["images"],
        "size": args.size,
        "threshold": args.threshold,
        "loss": metrics["loss"],
        "dice_mean": metrics["dice_mean"],
        "dice_median": metrics["dice_median"],
        "iou_mean": metrics["iou_mean"],
        "iou_median": metrics["iou_median"],
        "dice_mean_non_empty_target": metrics["dice_mean_non_empty_target"],
        "empty_target_images": metrics["empty_target_images"],
        "empty_prediction_images": metrics["empty_prediction_images"],
        "evaluation_space": f"letterbox 后的 {args.size}x{args.size} 有效区域（非补边像素）",
        "note": ("该结果只用于最终报告，未参与模型选择；与论文数字比较前需核对"
                 "数据版本与评估协议。"),
    }
    if args.output_json:
        _write_json(Path(args.output_json), report)
    if args.per_image_csv:
        path = Path(args.per_image_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(_jsonable(report), indent=2, ensure_ascii=False))
    return 0


def command_describe_model(args: argparse.Namespace) -> int:
    model = build_model(base_channels=args.base_channels, num_pools=args.num_pools,
                        input_shape=(3, args.size, args.size))
    graph = describe_graph(model)
    graph["output_shape"] = list(model.assert_shapes(batch_size=1))
    graph["notes"] = [
        "输出为 logits，sigmoid 只在损失/评估侧",
        "allowed_only 为 True 表示只用了 Conv/ReLU/MaxPool/Upsample/Concat",
        "INT8 量化与 RTL 实现不在本文件范围内",
    ]
    if args.output_json:
        _write_json(Path(args.output_json), graph)
    print(json.dumps(_jsonable(graph), indent=2, ensure_ascii=False))
    return 0


# ------------------------------------------------------------------- argparse
def _add_common_io(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image-dir", type=Path, required=True,
                        help="图像目录（官方训练集或独立测试集）")
    parser.add_argument("--mask-dir", type=Path, default=None,
                        help="掩膜目录；省略时表示掩膜与图像同目录")


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", type=int, default=256, help="letterbox 输出边长，默认 256")
    parser.add_argument("--base-channels", type=int, default=DEFAULT_BASE_CHANNELS)
    parser.add_argument("--num-pools", type=int, default=3, help="2x2 MaxPool 次数，1..3")
    parser.add_argument("--threshold", type=float, default=0.5, help="logits->mask 阈值，默认 0.5")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="ISIC 2016 Tiny U-Net FP32 基线：划分 / 训练 / 独立测试评估",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-split", help="扫描官方训练集并生成固定种子划分")
    _add_common_io(prepare)
    prepare.add_argument("--data-root", type=Path, default=None,
                         help="用于记录相对路径的根目录，默认取 --image-dir")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--val-fraction", type=float, default=0.2)
    prepare.add_argument("--no-stratify", action="store_true",
                         help="关闭按掩膜前景占比的分档，改为纯随机划分")
    prepare.add_argument("--include-test", action="store_true",
                         help="额外记录官方测试集文件名（只读不训练）")
    prepare.add_argument("--test-image-dir", type=Path, default=None)
    prepare.add_argument("--test-mask-dir", type=Path, default=None)
    prepare.set_defaults(handler=command_prepare_split)

    train = subparsers.add_parser("train", help="在 train/val 划分上训练，按 val Dice 选最佳")
    _add_common_io(train)
    _add_model_args(train)
    train.add_argument("--split", type=Path, required=True, help="prepare-split 产出的 split.json")
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/cuda:0/mps")
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--augment", action="store_true", help="训练集同步随机水平/垂直翻转")
    train.add_argument("--bce-weight", type=float, default=1.0)
    train.add_argument("--dice-weight", type=float, default=1.0)
    train.add_argument("--grad-clip", type=float, default=5.0, help="<=0 表示关闭梯度裁剪")
    train.set_defaults(handler=command_train)

    test = subparsers.add_parser("evaluate-test", help="用独立测试集评估指定 checkpoint（不参与选模型）")
    _add_common_io(test)
    _add_model_args(test)
    test.add_argument("--split", type=Path, required=True, help="含 test 角色的 split.json")
    test.add_argument("--checkpoint", type=Path, required=True)
    test.add_argument("--test-image-dir", type=Path, default=None,
                      help="官方测试集图像目录；省略则用 --image-dir")
    test.add_argument("--test-mask-dir", type=Path, default=None)
    test.add_argument("--batch-size", type=int, default=4)
    test.add_argument("--seed", type=int, default=DEFAULT_SEED)
    test.add_argument("--device", type=str, default="auto")
    test.add_argument("--num-workers", type=int, default=0)
    test.add_argument("--bce-weight", type=float, default=1.0)
    test.add_argument("--dice-weight", type=float, default=1.0)
    test.add_argument("--output-json", type=Path, default=None)
    test.add_argument("--per-image-csv", type=Path, default=None,
                      help="逐图 Dice/IoU 和空预测记录，便于错误分析")
    test.set_defaults(handler=command_evaluate_test)

    describe = subparsers.add_parser("describe-model", help="打印网络算子序列与参数量")
    _add_model_args(describe)
    describe.add_argument("--output-json", type=Path, default=None)
    describe.set_defaults(handler=command_describe_model)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:  # pragma: no cover - argparse 已强制子命令
        parser.error("必须指定子命令")
    if getattr(args, "include_test", False):
        if not args.test_image_dir:
            parser.error("--include-test 需要同时给出 --test-image-dir")
    try:
        return int(handler(args))
    except (DataLayoutError, ValueError, FileNotFoundError) as error:
        print(f"[错误] {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
