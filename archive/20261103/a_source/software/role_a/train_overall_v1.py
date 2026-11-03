"""总体精度优先的多来源 Tiny U-Net 训练器（``role_a_overall_v1``）。

目标（对应 ``docs/A03_总体精度优先训练协议.md``）
------------------------------------------------
* 训练集 **2309**：ISIC 2016 原训练筛后 897 + ISIC 2017 新增训练筛后 946 +
  已暴露的 2017 官方测试中冻结划分出的 466；
* 开发集 **638**：2016 官方测试筛后 376 + 2017 官方验证筛后 145 +
  已暴露的 2017 官方测试中冻结留出的 117；
* 主指标固定为三套开发集共 638 张的**逐图 Dice 算术平均**（每张图权重相同），
  只在 ``{0.45, 0.50, 0.53, 0.56, 0.60}`` 五个**全局统一**阈值中选最优；
* 每张图只跑一次网络前向，同时收集五个阈值的结果；
* 逐来源、逐来源自身面积分位（q10/q50）的微小/中等/大三组同时报告 n/Dice/IoU；
  中、大组只报告，**不设硬门槛**；
* 可恢复：每个完整轮次原子写 ``latest.pt``（模型 + 优化器 + 轮数 + 阶段 +
  配置哈希 + 输入/代码哈希 + RNG + 历史），断在轮次中途则该轮从头重跑；
* ``best.pt`` **只由主指标选择**，保存被选中的统一阈值与三来源/面积分组指标，
  并且**永远不是完成标记**（``completed.json`` 才是唯一的完成标记）。

硬边界
------
* 本文件**从不读取** ISIC 2018 的任何图像、掩膜、成绩或排除清单
  （:data:`FINAL_HOLDOUT` 只是文字声明，不构成任何路径）；
* 不产生新的随机划分，不把开发样本用于训练：训练/开发 ID 必须全局互斥，否则直接报错；
* 只写 ``--output-dir`` 下的产物；不修改原始数据、已有权重或其他实验目录；
* 不引入新依赖；沿用 ``data.py`` 的 letterbox/valid 语义与 ``metrics.py`` 的
  "有效像素 BCE + soft Dice" 损失；验证预处理与训练增强完全分离。

可恢复性的真实范围（不夸大）
----------------------------
数据顺序由 ``(seed, epoch)`` 派生的 ``torch.Generator`` 决定，随机增强由
``(seed, epoch, 样本下标)`` 派生的独立 ``random.Random`` 决定，二者都**不依赖**
全局 RNG 或 worker 调度顺序；因此 ``num_workers > 0`` 时恢复后的后续轮次仍与
不中断训练逐位一致。主进程 Python/NumPy/Torch 全局随机状态也一并保存与恢复，
但训练路径本身不消耗它们（无 Dropout 等随机算子）。

已知限制
--------
* 优化器状态与模型状态可精确恢复；但 ``torch`` 的 GPU 算子未强制确定性，
  换设备后不承诺逐位一致；
* 本文件在修改后其代码哈希会改变，``--resume`` 将**拒绝**继续已有试验
  （这是协议要求：来源与代码哈希不符即停）。要继续请另建输出目录；
* 本文件不声称任何精度，也不代表 FPGA 资源、时序或帧率。

用法::

    py -m software.role_a.train_overall_v1 --output-dir runs/role_a_overall_v1 \
        --init baseline --base-channels 8 --epochs 40 --batch-size 4 --lr 0.001 \
        --device auto --num-workers 2

    py -m software.role_a.train_overall_v1 --output-dir runs/role_a_overall_v1 --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:  # 兼容 `py -m software.role_a.train_overall_v1` 与直接运行脚本
    from software.role_a.data import (
        MASK_BINARY_THRESHOLD,
        DatasetItem,
        ISICDataset,
        IsicPaths,
        enumerate_isic_pairs,
        isic_collate,
        letterbox,
    )
    from software.role_a.metrics import segmentation_loss
    from software.role_a.model import build_model, describe_graph
except ImportError:  # pragma: no cover - 直接把本目录加入 sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data import (  # type: ignore[no-redef]
        MASK_BINARY_THRESHOLD,
        DatasetItem,
        ISICDataset,
        IsicPaths,
        enumerate_isic_pairs,
        isic_collate,
        letterbox,
    )
    from metrics import segmentation_loss  # type: ignore[no-redef]
    from model import build_model, describe_graph  # type: ignore[no-redef]

__all__ = [
    # 常量
    "PROJECT",
    "THRESHOLDS",
    "BAND_NAMES",
    "SUMMARY_BANDS",
    "PRIMARY_METRIC",
    "SELECTION_RULE",
    "RESUME_SCOPE",
    "FINAL_HOLDOUT",
    "ALLOWED_BASE_CHANNELS",
    "DEVELOPMENT_TOTAL",
    "TRAIN_TOTAL",
    "EXPOSED_CLEAN_TOTAL",
    "DEFAULT_SEED",
    "CHECKPOINT_NAME",
    "BEST_NAME",
    "COMPLETION_NAME",
    "PROTOCOL_NAME",
    "HISTORY_NAME",
    "SPLIT_NAME",
    "DEV_EVIDENCE_NAME",
    # 异常
    "AssemblyError",
    "ConfigError",
    "ResumeError",
    "IncompatibleWeightsError",
    "TrialStateError",
    # 数据结构
    "SourceSpec",
    "DataSpec",
    "DevSet",
    "Assembly",
    "RunConfig",
    # 工具
    "sha256_file",
    "id_digest",
    "config_digest",
    "atomic_write_json",
    "atomic_torch_save",
    "capture_rng_state",
    "restore_rng_state",
    "area_ratio",
    "area_cuts",
    "band_of",
    "summarize_records",
    "threshold_table",
    "select_threshold",
    "best_entry",
    "dev_evidence",
    "estimate_model",
    "spec_payload",
    "build_default_spec",
    "assemble",
    "frozen_split_provenance",
    "epoch_seed",
    "item_seed",
    "augment_geometry",
    "augment_photometry",
    "augment_arrays",
    "TrainDataset",
    "make_train_loader",
    "run_train_epoch",
    "evaluate_development",
    "build_development_report",
    "load_baseline_weights",
    "restore_model_state",
    "validate_resume",
    "load_checkpoint",
    "plan_mode",
    "resolve_device",
    "seed_everything",
    "code_hashes_snapshot",
    "train",
    "build_parser",
    "config_from_args",
    "main",
]

# --------------------------------------------------------------------- 常量
PROJECT: Path = Path(__file__).resolve().parents[2]

TRIAL_NAME: str = "role_a_overall_v1"
CONFIG_SCHEMA: str = "role-a-overall-v1-config-v1"
PROTOCOL_SCHEMA: str = "role-a-overall-v1-protocol-v1"
CHECKPOINT_SCHEMA: str = "role-a-overall-v1-checkpoint-v1"
BEST_SCHEMA: str = "role-a-overall-v1-best-v1"
COMPLETION_SCHEMA: str = "role-a-overall-v1-completion-v1"

CHECKPOINT_NAME: str = "latest.pt"
BEST_NAME: str = "best.pt"
COMPLETION_NAME: str = "completed.json"
PROTOCOL_NAME: str = "protocol.json"
HISTORY_NAME: str = "history.json"
SPLIT_NAME: str = "split_used.json"
DEV_EVIDENCE_NAME: str = "dev_best.json"
RESUME_LOG_NAME: str = "resume_log.json"

#: 冻结的五个全局统一阈值；不得按图或按来源分别挑选。
THRESHOLDS: tuple[float, ...] = (0.45, 0.50, 0.53, 0.56, 0.60)
BAND_NAMES: tuple[str, ...] = ("tiny", "medium", "large")
SUMMARY_BANDS: tuple[str, ...] = ("all",) + BAND_NAMES
ALLOWED_BASE_CHANNELS: tuple[int, ...] = (8, 12)

PRIMARY_METRIC: str = "development_638_mean_per_image_dice"
SELECTION_RULE: str = (
    "主指标 = 三套开发集共 638 张的逐图 Dice 算术平均；只在 "
    "{0.45, 0.50, 0.53, 0.56, 0.60} 五个全局统一阈值中取主指标最高者。"
    "主指标相等（1e-6 内）时先比较三来源 Dice 均值，再取更小的阈值。"
    "IoU、逐来源与逐面积分组只用于报告，不参与排序。"
)
FINAL_HOLDOUT: str = (
    "官方 ISIC 2018 Task 1 测试集：本脚本从不读取其图像、掩膜、成绩或排除清单，"
    "也不用于训练、早停、阈值选择或可视化选样。"
)
RESUME_SCOPE: str = (
    "latest.pt 在每个完整轮次结束后原子写入（同目录临时文件 -> fsync -> os.replace），"
    "含模型权重、优化器状态、已完成轮数与阶段（trained/evaluated）、配置哈希、"
    "输入/代码哈希、Python/NumPy/Torch 随机状态与历史。断在轮次中途时该轮从头重跑，"
    "磁盘上始终保留最后一个有效状态。数据顺序由 (seed, epoch) 派生的 torch.Generator 决定，"
    "随机增强由 (seed, epoch, 样本下标) 派生的独立 random.Random 决定，二者都不依赖全局 RNG "
    "或 worker 调度，因此 num_workers>0 时恢复仍与不中断训练逐位一致。"
)

TRAIN_TOTAL: int = 2309
DEVELOPMENT_TOTAL: int = 638
EXPOSED_CLEAN_TOTAL: int = 583

DEFAULT_SEED: int = 20260926

# 排除清单里必须存在的四个键（来自 runs/role_a_full2017/overlap_exclusions.json）。
EXCLUSION_KEYS: tuple[str, ...] = (
    "external_validation",
    "supplemental_training",
    "isic2016_training",
    "official_test_clean_subset_exclusions",
)

#: 面积分位至少需要多少张掩膜（真实开发集为 117/145/376）。
MIN_AREA_QUANTILE_IMAGES: int = 3
#: 与冻结 development_baseline 比对面积时的容差（同一掩膜的重复计算）。
AREA_TOLERANCE: float = 1e-6
#: 主指标并列判定的容差（按 6 位小数比较）。
METRIC_TIE_TOLERANCE_DIGITS: int = 6

#: 颜色增强范围（只作用于图像，不作用于掩膜/valid）。
COLOR_GAIN_RANGE: tuple[float, float] = (0.85, 1.15)
COLOR_OFFSET_RANGE: tuple[float, float] = (-12.0, 12.0)
COLOR_APPLY_PROBABILITY: float = 0.5

#: 停止条件里"接近 17 GB"的项目级预算，仅用于自查提示，不做任何下载。
PROJECT_BYTE_BUDGET: int = 17_000_000_000


# --------------------------------------------------------------------- 异常
class AssemblyError(RuntimeError):
    """数据来源、排除清单、冻结划分或 ID 隔离不符合协议。"""


class ConfigError(ValueError):
    """CLI/配置越界或与冻结协议不符。"""


class TrialStateError(RuntimeError):
    """输出目录状态不允许本次运行（已完成 / 未完成 / 状态不明）。"""


class ResumeError(RuntimeError):
    """检查点与当前配置、输入或代码不符，或缺少可恢复状态。"""


class IncompatibleWeightsError(ResumeError):
    """起点权重与目标模型结构不兼容；绝不静默部分加载。"""


# ----------------------------------------------------------------- 基础工具
def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    """分块计算文件 SHA256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """文本的 SHA256（UTF-8）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def id_digest(ids: Sequence[str] | None = None) -> str:
    """样本 ID 列表的稳定摘要（先排序，再连同数量一起哈希）。"""
    ordered = sorted(str(value) for value in (ids or ()))
    return sha256_text(f"{len(ordered)}\n" + "\n".join(ordered))


def _posix(path: Path | str) -> str:
    return Path(path).resolve().as_posix()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"无法序列化为 JSON 的类型: {type(value).__name__}")


def config_digest(payload: Mapping[str, Any]) -> str:
    """配置的稳定摘要（键排序 + 紧凑分隔符），用于 ``--resume`` 一致性校验。"""
    text = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_text(text)


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
        try:
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
        except OSError:  # 落盘尽力而为；原子性由 os.replace 保证
            pass
        last: OSError | None = None
        for attempt in range(max(1, attempts)):
            try:
                os.replace(temporary, path)
                return
            except OSError as error:  # Windows 上被杀毒/索引短暂占用时重试
                last = error
                if attempt + 1 < max(1, attempts):
                    time.sleep(0.05 * (attempt + 1))
        raise OSError(f"无法原子替换 {path}: {last}")
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path | str, payload: Any) -> None:
    """原子写 JSON（UTF-8，缩进 2）。"""
    data = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default).encode("utf-8")
    _atomic_write(Path(path), lambda target: target.write_bytes(data))


def atomic_torch_save(payload: Any, path: Path | str) -> None:
    """原子写 torch 检查点。"""
    _atomic_write(Path(path), lambda target: torch.save(payload, target))


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_file(path: Path | str | None, what: str) -> Path:
    if path is None:
        raise AssemblyError(f"{what} 未在数据规格中声明")
    resolved = Path(path)
    if not resolved.is_file():
        raise AssemblyError(f"{what} 不存在: {resolved}")
    return resolved


# ------------------------------------------------------------------- 随机源
def seed_everything(seed: int) -> None:
    """固定 Python / NumPy / Torch（含 CUDA）随机源。"""
    if int(seed) < 0:
        raise ConfigError(f"seed 必须 >= 0，收到 {seed}")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def resolve_device(requested: str | torch.device) -> torch.device:
    """解析 ``auto``/``cpu``/``cuda``/``cuda:0``/``mps``；不可用时不静默降级。"""
    name = str(requested).strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as error:
        raise ConfigError(f"无法识别的 device: {requested!r}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError("请求了 CUDA 设备，但 torch.cuda.is_available() 为 False")
    if device.type == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise ConfigError("请求了 mps 设备，但当前机器不可用")
    return device


def capture_rng_state() -> dict[str, Any]:
    """快照主进程 Python / NumPy / Torch(CPU+CUDA) 随机状态（全部可序列化）。"""
    version, internal_state, gauss_next = random.getstate()
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "python": {"version": int(version), "state": [int(v) for v in internal_state],
                   "gauss_next": gauss_next},
        "numpy": {"name": str(name), "keys": [int(v) for v in keys], "pos": int(position),
                  "has_gauss": int(has_gauss), "cached_gaussian": float(cached_gaussian)},
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
                         np.array([int(v) for v in numpy_state["keys"]], dtype=np.uint32),
                         int(numpy_state["pos"]), int(numpy_state["has_gauss"]),
                         float(numpy_state["cached_gaussian"])))
    torch.set_rng_state(torch.tensor([int(v) for v in state["torch_cpu"]], dtype=torch.uint8))
    cuda_state = state.get("torch_cuda") or []
    if cuda_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.tensor([int(v) for v in values], dtype=torch.uint8)
                                      for values in cuda_state])


# --------------------------------------------------------------- 面积与分组
def area_ratio(mask_path: Path | str) -> float:
    """掩膜前景占比（原始分辨率，``>=127`` 视为前景）。

    与 ``refine_small.area_ratio`` 的语义一致；只有来源清单没有记录
    ``area_ratio`` 时才回退到这个函数。
    """
    with Image.open(mask_path) as handle:
        values = np.asarray(handle.convert("L"), dtype=np.uint8)
    if values.size == 0:
        raise AssemblyError(f"掩膜为空: {mask_path}")
    return float((values >= MASK_BINARY_THRESHOLD).mean())


def area_cuts(areas: Mapping[str, float]) -> tuple[float, float]:
    """一套开发集自身的前景面积 q10/q50 分位（``numpy.quantile``，线性插值）。"""
    values = np.fromiter((float(value) for value in areas.values()), dtype=np.float64)
    if values.size < MIN_AREA_QUANTILE_IMAGES:
        raise AssemblyError(
            f"面积分位至少需要 {MIN_AREA_QUANTILE_IMAGES} 张掩膜，收到 {values.size}")
    q10, q50 = np.quantile(values, [0.1, 0.5])
    return float(q10), float(q50)


def band_of(area: float, cuts: tuple[float, float]) -> str:
    """按 ``area <= q10`` / ``<= q50`` 切成 tiny/medium/large。"""
    if float(area) <= float(cuts[0]):
        return "tiny"
    if float(area) <= float(cuts[1]):
        return "medium"
    return "large"


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def summarize_records(records: Sequence[Mapping[str, Any]], index: int) -> dict[str, Any]:
    """按 ``record['band']`` 汇总 n/Dice/IoU（每张图权重相同，不做面积加权）。"""
    result: dict[str, Any] = {}
    for band in SUMMARY_BANDS:
        rows = list(records) if band == "all" else [r for r in records if r["band"] == band]
        if not rows:
            result[band] = {"n": 0, "dice": None, "iou": None, "empty_prediction": 0}
            continue
        result[band] = {
            "n": len(rows),
            "dice": _mean_or_none([r["dice"][index] for r in rows]),
            "iou": _mean_or_none([r["iou"][index] for r in rows]),
            "empty_prediction": int(sum(int(bool(r["empty_prediction"][index])) for r in rows)),
        }
    return result


def threshold_table(records: Sequence[Mapping[str, Any]],
                    thresholds: Sequence[float] = THRESHOLDS) -> dict[float, dict[str, Any]]:
    """五阈值汇总表：主指标是 638 张的逐图 Dice 算术平均（等权）。"""
    rows = list(records)
    if not rows:
        raise AssemblyError("没有开发集逐图记录，无法汇总主指标")
    table: dict[float, dict[str, Any]] = {}
    for index, threshold in enumerate(thresholds):
        dice = [float(r["dice"][index]) for r in rows]
        iou = [float(r["iou"][index]) for r in rows]
        sources: dict[str, float] = {}
        for name in sorted({str(r["source"]) for r in rows}):
            subset = [float(r["dice"][index]) for r in rows if str(r["source"]) == name]
            sources[name] = float(np.mean(np.asarray(subset, dtype=np.float64)))
        table[float(threshold)] = {
            "n": len(rows),
            "dice": _mean_or_none(dice),
            "iou": _mean_or_none(iou),
            "source_dice": sources,
            "source_dice_balanced": _mean_or_none(list(sources.values())),
        }
    return table


def _rank_key(entry: Mapping[str, Any]) -> tuple[float, float]:
    return (-round(float(entry["dice"] or 0.0), METRIC_TIE_TOLERANCE_DIGITS),
            -round(float(entry["source_dice_balanced"] or 0.0), METRIC_TIE_TOLERANCE_DIGITS))


def select_threshold(table: Mapping[float, Mapping[str, Any]],
                     thresholds: Sequence[float] = THRESHOLDS) -> float:
    """在五个全局统一阈值中取主指标最高者；并列时先看三来源均值，再取更小阈值。"""
    if not table:
        raise AssemblyError("阈值汇总表为空，无法选择阈值")
    return float(min(thresholds, key=lambda t: (_rank_key(table[float(t)]), float(t))))


def _better_than(candidate: Mapping[str, Any], best: Mapping[str, Any] | None) -> bool:
    if best is None:
        return True
    candidate_key = (-round(float(candidate["primary_dice"]), METRIC_TIE_TOLERANCE_DIGITS),
                     -round(float(candidate["source_dice_balanced"]), METRIC_TIE_TOLERANCE_DIGITS))
    best_key = (-round(float(best["primary_dice"]), METRIC_TIE_TOLERANCE_DIGITS),
                -round(float(best["source_dice_balanced"]), METRIC_TIE_TOLERANCE_DIGITS))
    return candidate_key < best_key


# ------------------------------------------------------------------- 数据规格
@dataclass(frozen=True)
class SourceSpec:
    """一份来源的磁盘位置、来源清单与冻结选择规则。

    属性:
        name: 稳定名称，写入协议与分组报告。
        role: ``train`` 或 ``dev``。
        image_dir / mask_dir: 官方命名配对的目录。
        provenance: 来源清单 JSON；``None`` 表示没有可核对的清单。
        provenance_count: 清单里应记录的原始图像数；``None`` 表示清单不含逐项记录。
        expected_raw: 磁盘上应有的配对数（排除前）。
        exclusions_key: ``overlap_exclusions.json`` 中的键；与 ``id_list_key`` 二选一。
        expected_kept: 排除/冻结选择之后应有的样本数。
        id_list_key: 冻结 ``split.json`` 中的键（``train_ids``/``validation_ids``）。
    """

    name: str
    role: str
    image_dir: Path
    mask_dir: Path
    provenance: Path | None
    provenance_count: int | None
    expected_raw: int
    exclusions_key: str | None
    expected_kept: int
    id_list_key: str | None = None


@dataclass(frozen=True)
class DataSpec:
    """本试验允许使用的全部来源与冻结产物（可注入小样本路径用于自测）。"""

    train_sources: tuple[SourceSpec, ...]
    dev_sources: tuple[SourceSpec, ...]
    overlap_exclusions: Path
    exposed_split: Path
    exposed_exclusions: Path
    development_baseline: Path | None
    baseline_checkpoint: Path | None
    expected_train_total: int
    expected_dev_total: int
    expected_exposed_clean: int


@dataclass(frozen=True)
class DevSet:
    """一套开发集：样本、原始掩膜面积与自身的 q10/q50 分位。"""

    name: str
    items: tuple[IsicPaths, ...]
    areas: Mapping[str, float]
    cuts: tuple[float, float]


@dataclass(frozen=True)
class Assembly:
    """装配并校验后的数据集快照（不含任何图像缓存）。"""

    spec: DataSpec
    train_items: tuple[IsicPaths, ...]
    dev_sets: Mapping[str, DevSet]
    exclusions: Mapping[str, tuple[str, ...]]
    source_records: tuple[Mapping[str, Any], ...]
    counts: Mapping[str, int]
    split: Mapping[str, Any]
    split_digest: Mapping[str, str]
    hashes: Mapping[str, str | None]
    image_hash_checks: int
    train_ids: tuple[str, ...]
    dev_ids: Mapping[str, tuple[str, ...]]


def build_default_spec(project: Path | str = PROJECT) -> DataSpec:
    """真实项目的数据规格：frozen 的 6 份来源 + 2 份排除清单 + 冻结 split 与基线。

    这里只有常量，不读磁盘；所有数量核对在 :func:`assemble` 里执行。
    """
    root = Path(project)
    raw = root / "data" / "raw"
    isic2016 = raw / "isic2016"
    exclusions = root / "runs" / "role_a_full2017" / "overlap_exclusions.json"
    exposed_exclusions = root / "runs" / "role_a_full2017" / "test2017_overlap_exclusions.json"
    prepared = root / "runs" / "role_a_augmented_v4"
    return DataSpec(
        train_sources=(
            SourceSpec(name="isic2016_train", role="train",
                       image_dir=isic2016 / "train" / "images",
                       mask_dir=isic2016 / "train" / "masks",
                       provenance=isic2016 / "provenance.json", provenance_count=None,
                       expected_raw=900, exclusions_key="isic2016_training", expected_kept=897),
            SourceSpec(name="isic2017_new_train", role="train",
                       image_dir=raw / "isic2017_training" / "images",
                       mask_dir=raw / "isic2017_training" / "masks",
                       provenance=raw / "isic2017_training" / "provenance.json",
                       provenance_count=949, expected_raw=949,
                       exclusions_key="supplemental_training", expected_kept=946),
            SourceSpec(name="isic2017_exposed_train", role="train",
                       image_dir=raw / "isic2017_test" / "images",
                       mask_dir=raw / "isic2017_test" / "masks",
                       provenance=raw / "isic2017_test" / "provenance.json",
                       provenance_count=600, expected_raw=600, exclusions_key=None,
                       expected_kept=466, id_list_key="train_ids"),
        ),
        dev_sources=(
            SourceSpec(name="isic2016_test", role="dev",
                       image_dir=isic2016 / "test" / "images",
                       mask_dir=isic2016 / "test" / "masks",
                       provenance=isic2016 / "provenance.json", provenance_count=None,
                       expected_raw=379,
                       exclusions_key="official_test_clean_subset_exclusions", expected_kept=376),
            SourceSpec(name="isic2017_validation", role="dev",
                       image_dir=raw / "isic2017_validation" / "images",
                       mask_dir=raw / "isic2017_validation" / "masks",
                       provenance=raw / "isic2017_validation" / "provenance.json",
                       provenance_count=147, expected_raw=147,
                       exclusions_key="external_validation", expected_kept=145),
            SourceSpec(name="isic2017_exposed_validation", role="dev",
                       image_dir=raw / "isic2017_test" / "images",
                       mask_dir=raw / "isic2017_test" / "masks",
                       provenance=raw / "isic2017_test" / "provenance.json",
                       provenance_count=600, expected_raw=600, exclusions_key=None,
                       expected_kept=117, id_list_key="validation_ids"),
        ),
        overlap_exclusions=exclusions,
        exposed_split=prepared / "split.json",
        exposed_exclusions=exposed_exclusions,
        development_baseline=prepared / "validation_baseline.json",
        baseline_checkpoint=root / "runs" / "role_a_tinyunet" / "best.pt",
        expected_train_total=TRAIN_TOTAL,
        expected_dev_total=DEVELOPMENT_TOTAL,
        expected_exposed_clean=EXPOSED_CLEAN_TOTAL,
    )


def spec_payload(spec: DataSpec) -> dict[str, Any]:
    """把规格转成可哈希的纯 JSON 结构（所有路径为绝对 posix）。"""
    def _source(source: SourceSpec) -> dict[str, Any]:
        return {
            "name": source.name,
            "role": source.role,
            "image_dir": _posix(source.image_dir),
            "mask_dir": _posix(source.mask_dir),
            "provenance": None if source.provenance is None else _posix(source.provenance),
            "provenance_count": source.provenance_count,
            "expected_raw": source.expected_raw,
            "exclusions_key": source.exclusions_key,
            "expected_kept": source.expected_kept,
            "id_list_key": source.id_list_key,
        }

    return {
        "train_sources": [_source(s) for s in spec.train_sources],
        "dev_sources": [_source(s) for s in spec.dev_sources],
        "overlap_exclusions": _posix(spec.overlap_exclusions),
        "exposed_split": _posix(spec.exposed_split),
        "exposed_exclusions": _posix(spec.exposed_exclusions),
        "development_baseline": (None if spec.development_baseline is None
                                 else _posix(spec.development_baseline)),
        "baseline_checkpoint": (None if spec.baseline_checkpoint is None
                                else _posix(spec.baseline_checkpoint)),
        "expected_train_total": spec.expected_train_total,
        "expected_dev_total": spec.expected_dev_total,
        "expected_exposed_clean": spec.expected_exposed_clean,
    }


def frozen_split_provenance(spec: DataSpec) -> Path:
    """冻结划分（``train_ids``/``validation_ids``）所指来源的来源清单路径。"""
    provenances = {_posix(s.provenance) for s in (*spec.train_sources, *spec.dev_sources)
                   if s.id_list_key is not None and s.provenance is not None}
    if len(provenances) != 1:
        raise AssemblyError(
            "冻结划分的来源必须且只能有一份 provenance.json；收到 "
            f"{sorted(provenances) or '空'}")
    return Path(next(iter(provenances)))


# --------------------------------------------------------------------- 装配
def assemble(spec: DataSpec, *, verify_frozen: bool = True,
             verify_images: bool = False) -> Assembly:
    """按冻结协议装配训练/开发集，任何不符立即抛 :class:`AssemblyError`。

    校验内容：来源文件数、来源清单（``complete``/``image_count``/逐项 ID）、
    排除清单（键完整、ID 必须真实存在）、冻结 split（与干净集合的划分关系）、
    每份来源文件的 SHA256、训练/开发 ID 全局互斥、以及（可选的）逐图 SHA256。

    ``verify_frozen=True`` 时还会核对 ``development_baseline.json`` 与
    ``split.json`` 的哈希链、面积分位与分组标签；这些是冻结产物，不允许被改写。
    """
    exclusions_path = _require_file(spec.overlap_exclusions, "overlap_exclusions")
    exclusions = _read_json(exclusions_path)
    if not isinstance(exclusions, Mapping):
        raise AssemblyError(f"{exclusions_path} 不是 JSON 对象")
    for key in EXCLUSION_KEYS:
        value = exclusions.get(key)
        if not isinstance(value, Mapping):
            raise AssemblyError(f"排除清单 {exclusions_path} 缺少对象字段 {key!r}")

    split_path = _require_file(spec.exposed_split, "exposed_split")
    split = _read_json(split_path)
    if not isinstance(split, Mapping):
        raise AssemblyError(f"{split_path} 不是 JSON 对象")
    for key in ("train_ids", "validation_ids"):
        value = split.get(key)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise AssemblyError(f"冻结 split {split_path} 的 {key!r} 必须是字符串列表")

    exposed_exclusions_path = _require_file(spec.exposed_exclusions, "exposed_exclusions")
    exposed_exclusions = _read_json(exposed_exclusions_path)
    if not isinstance(exposed_exclusions, Mapping) or \
            not isinstance(exposed_exclusions.get("clean_exclusions"), list):
        raise AssemblyError(f"{exposed_exclusions_path} 缺少 clean_exclusions 列表")
    clean_excluded = {str(v) for v in exposed_exclusions["clean_exclusions"]}

    baseline_path: Path | None = None
    baseline: Mapping[str, Any] | None = None
    if spec.development_baseline is not None:
        baseline_path = Path(spec.development_baseline)
        if verify_frozen and not baseline_path.is_file():
            raise AssemblyError(f"冻结开发基线不存在: {baseline_path}")
        if baseline_path.is_file():
            loaded = _read_json(baseline_path)
            if not isinstance(loaded, Mapping):
                raise AssemblyError(f"{baseline_path} 不是 JSON 对象")
            baseline = loaded

    cache: dict[tuple[str, str, str], tuple[tuple[IsicPaths, ...], Mapping[str, Any]]] = {}

    def _load(source: SourceSpec) -> tuple[tuple[IsicPaths, ...], Mapping[str, Any]]:
        key = (_posix(source.image_dir), _posix(source.mask_dir),
               "" if source.provenance is None else _posix(source.provenance))
        if key in cache:
            return cache[key]
        items = tuple(enumerate_isic_pairs(source.image_dir, source.mask_dir, require_mask=True))
        if len(items) != source.expected_raw:
            raise AssemblyError(
                f"{source.name}: 磁盘配对 {len(items)} != 冻结期望 {source.expected_raw}")
        observed = {item.stem for item in items}
        if len(observed) != len(items):
            raise AssemblyError(f"{source.name}: 存在重复 stem")
        rows: Mapping[str, Any] = {}
        if source.provenance is not None:
            provenance_path = _require_file(source.provenance, f"{source.name} 来源清单")
            payload = _read_json(provenance_path)
            if not isinstance(payload, Mapping):
                raise AssemblyError(f"{provenance_path} 不是 JSON 对象")
            if "complete" in payload and payload["complete"] is not True:
                raise AssemblyError(f"{source.name}: 来源清单 complete 不为 True")
            candidate = payload.get("items")
            if isinstance(candidate, Mapping) and candidate:
                rows = candidate
            if rows:
                if source.provenance_count is not None:
                    recorded = int(payload.get("image_count", len(rows)))
                    if recorded != source.provenance_count:
                        raise AssemblyError(
                            f"{source.name}: 来源清单记录 {recorded} != 期望 "
                            f"{source.provenance_count}")
                if set(rows) != observed:
                    extra = sorted(observed - set(rows))[:5]
                    missing = sorted(set(rows) - observed)[:5]
                    raise AssemblyError(
                        f"{source.name}: 磁盘文件与来源清单逐项不一致"
                        f"（磁盘多出 {extra}，清单多出 {missing}）")
            elif source.provenance_count is not None:
                raise AssemblyError(
                    f"{source.name}: 来源清单缺少 items，无法核对 {source.provenance_count} 项")
        cache[key] = (items, rows)
        return cache[key]

    sources = (*spec.train_sources, *spec.dev_sources)
    per_source_ids: dict[str, tuple[str, ...]] = {}
    source_records: list[dict[str, Any]] = []
    dev_sets: dict[str, DevSet] = {}

    for source in sources:
        if source.role not in ("train", "dev"):
            raise AssemblyError(f"{source.name}: role 必须是 train/dev，收到 {source.role!r}")
        if (source.exclusions_key is None) == (source.id_list_key is None):
            raise AssemblyError(f"{source.name}: 必须且只能给出 exclusions_key 或 id_list_key 之一")
        items, rows = _load(source)
        observed = {item.stem for item in items}
        excluded_ids: list[str] = []
        if source.id_list_key is not None:
            listed = [str(v) for v in split[source.id_list_key]]
            chosen = set(listed)
            if len(chosen) != len(listed):
                raise AssemblyError(f"{source.name}: 冻结 {source.id_list_key} 有重复 ID")
            unknown = sorted(chosen - observed)
            if unknown:
                raise AssemblyError(
                    f"{source.name}: 冻结 {source.id_list_key} 有 {len(unknown)} 个 ID 不在来源目录"
                    f"（前几个 {unknown[:5]}）")
        else:
            exclusion_key = str(source.exclusions_key)
            listed = sorted(str(v) for v in exclusions[exclusion_key])
            chosen = observed - set(listed)
            excluded_ids = listed
            unknown = sorted(set(listed) - observed)
            if unknown:
                raise AssemblyError(
                    f"{source.name}: 排除清单 {exclusion_key} 含有来源目录中不存在的 ID"
                    f"（前几个 {unknown[:5]}）")
        if len(chosen) != source.expected_kept:
            raise AssemblyError(
                f"{source.name}: 选择后 {len(chosen)} != 冻结期望 {source.expected_kept}")
        ordered = tuple(item for item in items if item.stem in chosen)
        if len(ordered) != source.expected_kept:
            raise AssemblyError(f"{source.name}: 排序后数量 {len(ordered)} != {source.expected_kept}")
        per_source_ids[source.name] = tuple(sorted(chosen))
        record: dict[str, Any] = {
            "name": source.name,
            "role": source.role,
            "image_dir": _posix(source.image_dir),
            "mask_dir": _posix(source.mask_dir),
            "raw_count": source.expected_raw,
            "kept_count": source.expected_kept,
            "provenance": None if source.provenance is None else _posix(source.provenance),
            "provenance_items": len(rows),
            "exclusions_key": source.exclusions_key,
            "excluded_ids": excluded_ids,
            "id_list_key": source.id_list_key,
        }
        source_records.append(record)
        if source.role == "dev":
            areas: dict[str, float] = {}
            for item in ordered:
                row = rows.get(item.stem) if rows else None
                if isinstance(row, Mapping) and "area_ratio" in row:
                    value = float(row["area_ratio"])
                    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                        raise AssemblyError(
                            f"{source.name}: {item.stem} 的来源面积 {value!r} 不合法")
                    areas[item.stem] = value
                else:
                    areas[item.stem] = area_ratio(item.mask)
            dev_sets[source.name] = DevSet(name=source.name, items=ordered, areas=areas,
                                           cuts=area_cuts(areas))

    # 冻结划分必须正好把"干净暴露集合"分成训练与开发两半。
    split_sources = [s for s in sources if s.id_list_key is not None]
    if len(split_sources) != 2:
        raise AssemblyError(f"预期恰好两个冻结划分来源，收到 {len(split_sources)}")
    split_by_role = {s.role: s for s in split_sources}
    if set(split_by_role) != {"train", "dev"}:
        raise AssemblyError("冻结划分来源必须一个是 train、一个是 dev")
    exposed_dev_source = split_by_role["dev"]
    exposed_train = set(per_source_ids[split_by_role["train"].name])
    exposed_dev = set(per_source_ids[exposed_dev_source.name])
    if exposed_train & exposed_dev:
        raise AssemblyError("冻结 split 的训练与开发 ID 重叠")
    exposed_items, _ = _load(split_by_role["train"])
    exposed_clean = {item.stem for item in exposed_items} - clean_excluded
    if exposed_train | exposed_dev != exposed_clean:
        missing = sorted(exposed_clean - (exposed_train | exposed_dev))[:5]
        extra = sorted((exposed_train | exposed_dev) - exposed_clean)[:5]
        raise AssemblyError(
            "冻结 split 的 train+validation != 干净暴露集合"
            f"（缺少 {missing}，多出 {extra}）")
    if len(exposed_clean) != spec.expected_exposed_clean:
        raise AssemblyError(
            f"干净暴露集合 {len(exposed_clean)} != 冻结期望 {spec.expected_exposed_clean}")

    train_ids: set[str] = set()
    dev_ids: set[str] = set()
    for source in spec.train_sources:
        ids = set(per_source_ids[source.name])
        if train_ids & ids:
            raise AssemblyError(f"{source.name} 与其它训练来源 ID 重叠")
        train_ids |= ids
    for source in spec.dev_sources:
        ids = set(per_source_ids[source.name])
        if dev_ids & ids:
            raise AssemblyError(f"{source.name} 与其它开发来源 ID 重叠")
        dev_ids |= ids
    overlap = sorted(train_ids & dev_ids)
    if overlap:
        raise AssemblyError(
            f"训练/开发 ID 重叠 {len(overlap)} 个（前几个 {overlap[:5]}）；开发样本绝不能进入训练")
    if len(train_ids) != spec.expected_train_total:
        raise AssemblyError(f"训练集总数 {len(train_ids)} != 冻结期望 {spec.expected_train_total}")
    if len(dev_ids) != spec.expected_dev_total:
        raise AssemblyError(f"开发集总数 {len(dev_ids)} != 冻结期望 {spec.expected_dev_total}")

    # ---- 冻结产物哈希链 ----
    if verify_frozen:
        if baseline is None or baseline_path is None:
            raise AssemblyError("verify_frozen=True 需要 development_baseline.json")
        recorded_split = baseline.get("split_sha256")
        if recorded_split != sha256_file(split_path):
            raise AssemblyError(
                f"冻结 split 的 SHA256 与 {baseline_path.name} 记录不符；split 可能被改写")
        recorded_count = baseline.get("count")
        if int(recorded_count if recorded_count is not None else -1) != \
                len(dev_sets[exposed_dev_source.name].items):
            raise AssemblyError(
                f"{baseline_path.name} 的 count 与冻结开发集样本数不符")
        provenance_path = frozen_split_provenance(spec)
        if split.get("source_provenance_sha256") != sha256_file(provenance_path):
            raise AssemblyError("split.json 的 source_provenance_sha256 与来源清单不符")
        if split.get("source_overlap_exclusions_sha256") != \
                sha256_file(exposed_exclusions_path):
            raise AssemblyError("split.json 的 source_overlap_exclusions_sha256 与排除清单不符")
        if spec.baseline_checkpoint is not None and Path(spec.baseline_checkpoint).is_file():
            if baseline.get("baseline_sha256") != sha256_file(spec.baseline_checkpoint):
                raise AssemblyError("development_baseline 记录的基线权重 SHA256 与磁盘不符")

    if baseline is not None and "records" in baseline and \
            dev_sets.get(exposed_dev_source.name) is not None:
        dev = dev_sets[exposed_dev_source.name]
        recorded_rows = baseline["records"]
        if not isinstance(recorded_rows, list) or not recorded_rows:
            raise AssemblyError(f"{baseline_path} 的 records 必须是非空列表")
        seen: set[str] = set()
        for row in recorded_rows:
            if not isinstance(row, Mapping) or "stem" not in row:
                raise AssemblyError(f"{baseline_path} 的 records 条目缺少 stem")
            stem = str(row["stem"])
            if stem in seen:
                raise AssemblyError(f"{baseline_path} 的 records 有重复 stem {stem}")
            seen.add(stem)
            if stem not in dev.areas:
                raise AssemblyError(f"{baseline_path} 记录了不属于冻结开发集的样本 {stem}")
            if "area_ratio" not in row:
                raise AssemblyError(f"{baseline_path} 的 records 条目 {stem} 缺少 area_ratio")
            recorded_area = float(row["area_ratio"])
            if not np.isfinite(recorded_area) or \
                    abs(recorded_area - dev.areas[stem]) > AREA_TOLERANCE:
                raise AssemblyError(f"样本 {stem} 的面积与冻结开发基线不符")
            if row.get("band") != band_of(dev.areas[stem], dev.cuts):
                raise AssemblyError(f"样本 {stem} 的面积分组与冻结开发基线不符")
        if seen != set(dev.areas):
            raise AssemblyError(f"{baseline_path} 未覆盖全部冻结开发样本")

    hashes: dict[str, str | None] = {}
    for source in sources:
        if source.provenance is not None:
            path = Path(source.provenance)
            hashes[f"provenance:{source.name}"] = sha256_file(path) if path.is_file() else None
    for name, path in (("overlap_exclusions", exclusions_path),
                       ("exposed_split", split_path),
                       ("exposed_exclusions", exposed_exclusions_path),
                       ("development_baseline", baseline_path),
                       ("baseline_checkpoint",
                        None if spec.baseline_checkpoint is None
                        else Path(spec.baseline_checkpoint))):
        hashes[name] = (sha256_file(path) if path is not None and Path(path).is_file() else None)

    image_hash_checks = 0
    if verify_images:
        for source in sources:
            items, rows = _load(source)
            if not rows:
                continue
            for item in items:
                row = rows.get(item.stem)
                if not isinstance(row, Mapping):
                    continue
                if "image_sha256" in row and sha256_file(item.image) != row["image_sha256"]:
                    raise AssemblyError(f"{source.name}: {item.stem} 图像 SHA256 与来源清单不符")
                if "mask_sha256" in row and row["mask_sha256"] and item.mask is not None and \
                        sha256_file(item.mask) != row["mask_sha256"]:
                    raise AssemblyError(f"{source.name}: {item.stem} 掩膜 SHA256 与来源清单不符")
                image_hash_checks += 1
        print(f"[overall-v1] 逐图 SHA256 复核通过: {image_hash_checks} 对", flush=True)

    counts: dict[str, int] = {}
    for source in spec.train_sources:
        counts[f"train:{source.name}"] = len(per_source_ids[source.name])
    for source in spec.dev_sources:
        counts[f"development:{source.name}"] = len(per_source_ids[source.name])
    counts["train_total"] = len(train_ids)
    counts["development_total"] = len(dev_ids)
    counts["exposed_clean_total"] = len(exposed_clean)

    train_items: list[IsicPaths] = []
    for source in spec.train_sources:
        chosen = set(per_source_ids[source.name])
        items, _ = _load(source)
        train_items.extend(item for item in items if item.stem in chosen)

    split_digest: dict[str, str] = {"train_ids": id_digest(train_ids)}
    for source in spec.dev_sources:
        split_digest[f"development:{source.name}"] = id_digest(per_source_ids[source.name])
    split_digest["development_all"] = id_digest(dev_ids)
    split_digest["reference_split_file"] = sha256_file(split_path)

    return Assembly(spec=spec, train_items=tuple(train_items), dev_sets=dev_sets,
                    exclusions={key: tuple(sorted(str(v) for v in exclusions[key]))
                                for key in EXCLUSION_KEYS},
                    source_records=tuple(source_records), counts=counts, split=split,
                    split_digest=split_digest, hashes=hashes,
                    image_hash_checks=image_hash_checks,
                    train_ids=tuple(sorted(train_ids)),
                    dev_ids={source.name: per_source_ids[source.name]
                             for source in spec.dev_sources})


# ----------------------------------------------------------------- 静态估算
def _conv_parameters(in_channels: int, out_channels: int, kernel: int) -> int:
    return int(out_channels) * (int(in_channels) * kernel * kernel + 1)


def _conv_macs(in_channels: int, out_channels: int, kernel: int, resolution: int) -> int:
    return int(out_channels) * resolution * resolution * int(in_channels) * kernel * kernel


def estimate_model(base_channels: int, num_pools: int = 3, size: int = 256,
                   in_channels: int = 3) -> dict[str, Any]:
    """Tiny U-Net 的静态估算：参数量、卷积 MAC/图、单个峰值 INT8 特征字节数。

    只数 Conv3x3/Conv1x1 的乘加；ReLU/MaxPool/nearest Upsample/Concat 不产生 MAC。
    "峰值特征" 定义为最浅一级解码器同时驻留的两块全分辨率张量
    （``concat`` 的 ``2*base`` 通道 + 输出 ``base`` 通道，共 ``3*base`` 个通道），
    即 ``3 * base_channels * size * size`` 字节；这是算术估计，**不是**实测资源占用。
    """
    base = int(base_channels)
    if base < 1:
        raise ConfigError(f"base_channels 必须 >= 1，收到 {base_channels}")
    if not 1 <= int(num_pools) <= 3:
        raise ConfigError(f"num_pools 必须在 1..3 之间，收到 {num_pools}")
    size = int(size)
    channels = [base * (2 ** level) for level in range(int(num_pools) + 1)]
    parameters = 0
    macs = 0
    for level, out_channels in enumerate(channels):
        level_in = int(in_channels) if level == 0 else channels[level - 1]
        resolution = size >> level
        parameters += _conv_parameters(level_in, out_channels, 3)
        parameters += _conv_parameters(out_channels, out_channels, 3)
        macs += _conv_macs(level_in, out_channels, 3, resolution)
        macs += _conv_macs(out_channels, out_channels, 3, resolution)
    for level in range(int(num_pools) - 1, -1, -1):
        level_in = channels[level + 1] + channels[level]
        out_channels = channels[level]
        resolution = size >> level
        parameters += _conv_parameters(level_in, out_channels, 3)
        parameters += _conv_parameters(out_channels, out_channels, 3)
        macs += _conv_macs(level_in, out_channels, 3, resolution)
        macs += _conv_macs(out_channels, out_channels, 3, resolution)
    parameters += channels[0] + 1
    macs += 1 * size * size * channels[0]
    peak_bytes = 3 * base * size * size
    return {
        "base_channels": base,
        "num_pools": int(num_pools),
        "size": size,
        "in_channels": int(in_channels),
        "channels": channels,
        "parameters": int(parameters),
        "conv_macs": int(macs),
        "conv_macs_giga": macs / 1e9,
        "peak_feature_bytes_int8": int(peak_bytes),
        "peak_feature_megabytes_int8": peak_bytes / 1e6,
        "operators": ["Conv2d 3x3", "Conv2d 1x1", "ReLU", "MaxPool2d 2x2",
                      "Upsample nearest 2x", "Concat"],
        "note": ("纯算术估计，不含 ReLU/MaxPool/Upsample/Concat 的搬运开销；"
                 "不是实测资源、时序或功耗，也不能作为上板证明"),
    }


# ------------------------------------------------------- 数据加载与增强
def epoch_seed(seed: int, epoch: int) -> int:
    """某轮的确定性种子（只依赖 seed 与轮数，恢复后不变）。"""
    return (int(seed) * 1_000_003 + int(epoch) * 7919 + 17) % (2**31 - 1)


def item_seed(seed: int, epoch: int, index: int) -> int:
    """某轮某张图的确定性增强种子（不依赖 worker 调度）。"""
    return (epoch_seed(seed, epoch) * 1_000_003 + int(index)) % (2**31 - 1)


def augment_geometry(image: np.ndarray, mask: np.ndarray, valid: np.ndarray,
                     rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """同步几何增强：水平/垂直翻转同时作用于图像、掩膜与 valid。

    ``valid`` 是非补边矩形，翻转后仍是非补边矩形；掩膜在补边区恒为 0，
    因此 ``mask > 0`` 永远不会跑到 ``valid == 0`` 的区域。
    """
    if rng.random() < 0.5:
        image, mask, valid = image[:, ::-1], mask[:, ::-1], valid[:, ::-1]
    if rng.random() < 0.5:
        image, mask, valid = image[::-1], mask[::-1], valid[::-1]
    return (np.ascontiguousarray(image), np.ascontiguousarray(mask),
            np.ascontiguousarray(valid))


def augment_photometry(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """轻量颜色增强：只改图像（亮度/对比度线性变换），不触碰掩膜与 valid。"""
    if rng.random() >= COLOR_APPLY_PROBABILITY:
        return np.ascontiguousarray(image)
    gain = rng.uniform(*COLOR_GAIN_RANGE)
    offset = rng.uniform(*COLOR_OFFSET_RANGE)
    scaled = np.clip(image.astype(np.float32) * gain + offset, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(scaled)


def augment_arrays(image: np.ndarray, mask: np.ndarray, valid: np.ndarray,
                   rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """完整的同步增强：几何 -> 颜色。"""
    image, mask, valid = augment_geometry(image, mask, valid, rng)
    return augment_photometry(image, rng), mask, valid


class TrainDataset(Dataset):
    """训练集：letterbox + 可复现的同步几何/颜色增强（无磁盘缓存）。

    增强随机数只由 ``(seed, epoch, 样本下标)`` 决定，不读全局 RNG，
    因此与 ``num_workers`` 及 worker 调度顺序无关，``--resume`` 后可逐位复现。
    ``augment=False`` 时的输出与 ``ISICDataset(augment=False)`` 逐位一致。
    """

    def __init__(self, items: Sequence[IsicPaths], size: int = 256, seed: int = 0,
                 augment: bool = True, epoch: int = 0) -> None:
        if not items:
            raise AssemblyError("训练集为空：没有任何已配对的样本")
        if int(size) <= 0:
            raise AssemblyError(f"size 必须为正，收到 {size}")
        self.items: tuple[IsicPaths, ...] = tuple(items)
        self.size = int(size)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def set_epoch(self, epoch: int) -> None:
        """切换轮次（增强参数随之确定性地改变）。"""
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> DatasetItem:
        item = self.items[index]
        image, mask, valid, info, mask_was_binary = letterbox(item.image, item.mask,
                                                              size=self.size)
        if self.augment:
            rng = random.Random(item_seed(self.seed, self.epoch, index))
            image, mask, valid = augment_arrays(image, mask, valid, rng)
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
        if tensor.shape != (3, self.size, self.size):
            raise AssemblyError(f"图像张量 shape 异常: {tuple(tensor.shape)}")
        mask_tensor = None
        if mask is not None:
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        valid_tensor = torch.from_numpy(np.ascontiguousarray(valid)).float().unsqueeze(0)
        return DatasetItem(image=tensor, mask=mask_tensor, valid=valid_tensor,
                           filename=item.image.name, original_size=info.original_size,
                           letterbox=info, mask_was_binary=mask_was_binary)


def make_train_loader(dataset: TrainDataset, batch_size: int, num_workers: int,
                      seed: int, epoch: int) -> DataLoader:
    """训练 DataLoader：洗牌顺序由 ``(seed, epoch)`` 派生的 generator 决定。"""
    if int(batch_size) < 1:
        raise ConfigError(f"batch_size 必须 >= 1，收到 {batch_size}")
    if int(num_workers) < 0:
        raise ConfigError(f"num_workers 必须 >= 0，收到 {num_workers}")
    generator = torch.Generator()
    generator.manual_seed(epoch_seed(seed, epoch))
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=True,
                      num_workers=int(num_workers), collate_fn=isic_collate,
                      generator=generator, drop_last=False, persistent_workers=False)


def run_train_epoch(model: torch.nn.Module, loader: Sequence[Mapping[str, Any]],
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    *, bce_weight: float = 1.0, dice_weight: float = 1.0,
                    grad_clip: float = 5.0) -> dict[str, float]:
    """跑一轮训练：有效像素 BCE + soft Dice，返回各项 batch 均值。"""
    model.train(True)
    totals = {"loss": 0.0, "bce": 0.0, "soft_dice": 0.0}
    batches = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"]
        if masks is None:
            raise AssemblyError("训练 batch 缺少掩膜")
        masks = masks.to(device, non_blocking=True)
        valids = batch["valid"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss, parts = segmentation_loss(logits, masks, valids,
                                        bce_weight=bce_weight, dice_weight=dice_weight)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["bce"] += float(parts["bce"])
        totals["soft_dice"] += float(parts["soft_dice"])
        batches += 1
    if batches == 0:
        raise AssemblyError("训练 loader 为空，没有产生任何 batch")
    return {key: value / batches for key, value in totals.items()}


def _binary_dice_iou(pred_sum: int, true_sum: int, intersection: int) -> tuple[float, float]:
    """与 ``metrics.py`` 一致的边界约定：都空 => 1.0；只有一方空 => 0.0。"""
    pred_sum, true_sum, intersection = int(pred_sum), int(true_sum), int(intersection)
    denominator = pred_sum + true_sum
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    union = denominator - intersection
    iou = 1.0 if union == 0 else intersection / union
    return dice, iou


@torch.inference_mode()
def evaluate_development(model: torch.nn.Module, dev_sets: Mapping[str, DevSet], *,
                         size: int = 256, batch_size: int = 4, num_workers: int = 0,
                         thresholds: Sequence[float] = THRESHOLDS,
                         device: torch.device | str | None = None
                         ) -> list[dict[str, Any]]:
    """在三套开发集上**每张图只前向一次**，同时收集五个阈值的逐图 Dice/IoU。

    验证预处理固定为 ``ISICDataset(augment=False)``（letterbox + valid），
    与训练增强完全分离；开发集样本从不参与训练。
    """
    target = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    records: list[dict[str, Any]] = []
    for name, dev in dev_sets.items():
        dataset = ISICDataset(dev.items, size=int(size), augment=False)
        loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False,
                            num_workers=int(num_workers), collate_fn=isic_collate,
                            drop_last=False)
        for batch in loader:
            images = batch["image"].to(target, non_blocking=True)
            masks = batch["mask"]
            if masks is None:
                raise AssemblyError("开发集 batch 缺少掩膜，无法计算 Dice/IoU")
            masks = masks.to(target, non_blocking=True)
            valids = batch["valid"].to(target, non_blocking=True)
            probability = torch.sigmoid(model(images))
            keep = valids > 0
            truth = (masks > 0) & keep
            true_sums = [int(v) for v in truth.flatten(1).sum(dim=1).tolist()]
            collected: list[tuple[list[int], list[int]]] = []
            for threshold in thresholds:
                predicted = (probability >= float(threshold)) & keep
                pred_sums = [int(v) for v in predicted.flatten(1).sum(dim=1).tolist()]
                intersections = [int(v) for v in (predicted & truth).flatten(1)
                                 .sum(dim=1).tolist()]
                collected.append((pred_sums, intersections))
            for row, filename in enumerate(batch["filename"]):
                stem = Path(str(filename)).stem
                area = float(dev.areas[stem])
                dice: list[float] = []
                iou: list[float] = []
                empty: list[bool] = []
                for pred_sums, intersections in collected:
                    value_dice, value_iou = _binary_dice_iou(pred_sums[row], true_sums[row],
                                                             intersections[row])
                    dice.append(value_dice)
                    iou.append(value_iou)
                    empty.append(pred_sums[row] == 0)
                records.append({"source": name, "stem": stem, "area_ratio": area,
                                "band": band_of(area, dev.cuts), "dice": dice, "iou": iou,
                                "empty_prediction": empty})
    expected = sum(len(dev.items) for dev in dev_sets.values())
    if len(records) != expected:
        raise AssemblyError(f"开发集逐图记录 {len(records)} != 期望 {expected}")
    return records


def build_development_report(records: Sequence[Mapping[str, Any]],
                             dev_sets: Mapping[str, DevSet],
                             thresholds: Sequence[float] = THRESHOLDS) -> dict[str, Any]:
    """由逐图记录生成主指标、五阈值表、逐来源/逐面积分组报告。"""
    rows = list(records)
    table = threshold_table(rows, thresholds)
    selected = select_threshold(table, thresholds)
    index = list(thresholds).index(selected)
    sources: dict[str, Any] = {}
    for name, dev in dev_sets.items():
        subset = [r for r in rows if str(r["source"]) == name]
        if not subset:
            raise AssemblyError(f"开发集 {name} 没有任何逐图记录")
        sources[name] = {
            "n": len(subset),
            "area_cuts": {"q10": float(dev.cuts[0]), "q50": float(dev.cuts[1])},
            **summarize_records(subset, index),
        }
    return {
        "primary_metric": PRIMARY_METRIC,
        "selected_threshold": float(selected),
        "primary": {"n": len(rows), "dice": table[selected]["dice"],
                    "iou": table[selected]["iou"]},
        "threshold_table": {float(t): dict(table[float(t)]) for t in thresholds},
        "sources": sources,
        "area_groups_pooled": summarize_records(rows, index),
        "selection_rule": SELECTION_RULE,
    }


def best_entry(epoch: int, report: Mapping[str, Any],
               thresholds: Sequence[float] = THRESHOLDS) -> dict[str, Any]:
    """把某一轮的开发集报告压成"当前最佳候选"条目（可写入检查点）。"""
    selected = float(report["selected_threshold"])
    index = list(thresholds).index(selected)
    source_dice = {name: float(info["all"]["dice"])
                   for name, info in report["sources"].items()}
    return {
        "epoch": int(epoch),
        "primary_metric": PRIMARY_METRIC,
        "selected_threshold": selected,
        "primary_dice": float(report["primary"]["dice"]),
        "primary_iou": float(report["primary"]["iou"]),
        "development_images": int(report["primary"]["n"]),
        "source_dice": source_dice,
        "source_dice_balanced": _mean_or_none(list(source_dice.values())),
        "sources": report["sources"],
        "area_groups_pooled": report["area_groups_pooled"],
        "threshold_dice": {str(float(t)): float(report["threshold_table"][float(t)]["dice"])
                           for t in thresholds},
        "threshold_index": int(index),
        "selection_rule": SELECTION_RULE,
    }


def dev_evidence(records: Sequence[Mapping[str, Any]], selected: float,
                 thresholds: Sequence[float] = THRESHOLDS) -> list[dict[str, Any]]:
    """最佳轮次在选中阈值下的逐图证据（只保存必要的标量，不写原图缓存）。"""
    index = list(thresholds).index(float(selected))
    return [{
        "stem": str(r["stem"]),
        "source": str(r["source"]),
        "area_ratio": float(r["area_ratio"]),
        "band": str(r["band"]),
        "threshold": float(selected),
        "dice": float(r["dice"][index]),
        "iou": float(r["iou"][index]),
        "empty_prediction": bool(r["empty_prediction"][index]),
    } for r in records]


# ------------------------------------------------------------------- 运行配置
@dataclass(frozen=True)
class RunConfig:
    """一次训练（或恢复）的完整配置；:meth:`payload` 决定配置哈希。"""

    output_dir: Path
    init: str
    base_channels: int
    num_pools: int
    size: int
    epochs: int
    batch_size: int
    lr: float
    seed: int
    device: str
    num_workers: int
    data_spec: DataSpec
    init_checkpoint: Path | None = None
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    augment: bool = True
    resume: bool = False
    verify_images: bool = False
    thresholds: tuple[float, ...] = THRESHOLDS

    def payload(self) -> dict[str, Any]:
        """参与配置哈希的部分。``device`` 不参与（同一数学配置换设备仍应可恢复），
        但设备名会记录在检查点里。"""
        return {
            "schema": CONFIG_SCHEMA,
            "trial": TRIAL_NAME,
            "init": self.init,
            "init_checkpoint": (None if self.init_checkpoint is None
                                else _posix(self.init_checkpoint)),
            "base_channels": int(self.base_channels),
            "num_pools": int(self.num_pools),
            "size": int(self.size),
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "lr": float(self.lr),
            "betas": [float(v) for v in self.betas],
            "eps": float(self.eps),
            "weight_decay": float(self.weight_decay),
            "grad_clip": float(self.grad_clip),
            "bce_weight": float(self.bce_weight),
            "dice_weight": float(self.dice_weight),
            "seed": int(self.seed),
            "num_workers": int(self.num_workers),
            "augment": bool(self.augment),
            "verify_images": bool(self.verify_images),
            "thresholds": [float(t) for t in self.thresholds],
            "primary_metric": PRIMARY_METRIC,
            "loss": "valid-pixel BCE + soft Dice (bce_weight/dice_weight)",
            "optimizer": "torch.optim.Adam",
            "output_dir": _posix(self.output_dir),
            "data": spec_payload(self.data_spec),
        }


def _validate_config(config: RunConfig) -> None:
    if config.init not in ("baseline", "scratch", "checkpoint"):
        raise ConfigError(f"--init 必须是 baseline、scratch 或 checkpoint，收到 {config.init!r}")
    if (config.init == "checkpoint") != (config.init_checkpoint is not None):
        raise ConfigError("--init checkpoint 必须提供 --init-checkpoint；其它起点不能同时指定它")
    if int(config.base_channels) not in ALLOWED_BASE_CHANNELS:
        raise ConfigError(
            f"--base-channels 只允许 {list(ALLOWED_BASE_CHANNELS)}，收到 {config.base_channels}")
    if not 1 <= int(config.num_pools) <= 3:
        raise ConfigError(f"--num-pools 必须在 1..3，收到 {config.num_pools}")
    if int(config.epochs) < 1:
        raise ConfigError(f"--epochs 必须 >= 1，收到 {config.epochs}")
    if int(config.batch_size) < 1:
        raise ConfigError(f"--batch-size 必须 >= 1，收到 {config.batch_size}")
    if int(config.num_workers) < 0:
        raise ConfigError(f"--num-workers 必须 >= 0，收到 {config.num_workers}")
    if float(config.lr) <= 0:
        raise ConfigError(f"--lr 必须 > 0，收到 {config.lr}")
    if int(config.size) < 8 or int(config.size) % (2 ** int(config.num_pools)):
        raise ConfigError(
            f"--size 必须 >= 8 且能被 2**num_pools={2 ** int(config.num_pools)} 整除，"
            f"收到 {config.size}")
    if tuple(float(t) for t in config.thresholds) != THRESHOLDS:
        raise ConfigError(
            f"阈值网格冻结为 {list(THRESHOLDS)}，收到 {list(config.thresholds)}；"
            "不允许按图或按来源挑阈值")
    for name in ("bce_weight", "dice_weight"):
        if float(getattr(config, name)) < 0:
            raise ConfigError(f"{name} 必须非负，收到 {getattr(config, name)}")
    if float(config.bce_weight) == 0 and float(config.dice_weight) == 0:
        raise ConfigError("bce_weight 与 dice_weight 不能同时为 0")


# --------------------------------------------------------------- 检查点状态
def plan_mode(out_dir: Path | str, resume: bool) -> str:
    """返回 ``fresh`` / ``resume`` / ``complete`` / ``missing-checkpoint`` /
    ``partial`` / ``unknown``。"""
    out_dir = Path(out_dir)
    if (out_dir / COMPLETION_NAME).exists():
        return "complete"
    if (out_dir / CHECKPOINT_NAME).exists():
        return "resume" if resume else "partial"
    if out_dir.is_dir() and any(out_dir.iterdir()):
        return "unknown"
    return "missing-checkpoint" if resume else "fresh"


REQUIRED_CHECKPOINT_FIELDS: tuple[str, ...] = (
    "schema", "epoch", "phase", "config", "config_hash", "input_hashes", "code_hashes",
    "model_state", "optimizer_state", "rng", "history",
)


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    """``weights_only=True`` 加载检查点（只允许张量与基本类型）。"""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ResumeError(f"检查点结构异常: {path}")
    return checkpoint


def validate_resume(checkpoint: Mapping[str, Any], *, config_hash: str,
                    input_hashes: Mapping[str, str | None],
                    code_hashes: Mapping[str, str], epochs: int) -> tuple[int, str]:
    """校验检查点与当前配置/输入/代码一致，返回 ``(epoch, phase)``。

    任何不一致都抛 :class:`ResumeError`，绝不"尽力而为"地继续训练。
    """
    if not isinstance(checkpoint, Mapping):
        raise ResumeError("检查点不是字典结构")
    missing = [name for name in REQUIRED_CHECKPOINT_FIELDS if name not in checkpoint]
    if missing:
        raise ResumeError(f"检查点缺少字段: {missing}")
    if checkpoint["schema"] != CHECKPOINT_SCHEMA:
        raise ResumeError(f"检查点 schema 不符: {checkpoint['schema']!r}")
    if checkpoint["config_hash"] != config_hash:
        raise ResumeError("检查点配置哈希与当前配置不符（超参数/路径/来源已变化）")
    stored_inputs = checkpoint["input_hashes"]
    if not isinstance(stored_inputs, Mapping):
        raise ResumeError("检查点 input_hashes 不是字典")
    for name, digest in input_hashes.items():
        if stored_inputs.get(name) != digest:
            raise ResumeError(
                f"输入 {name} 的 SHA256 与当前文件不符（权重/排除清单/冻结划分已变化）")
    stored_code = checkpoint["code_hashes"]
    if not isinstance(stored_code, Mapping):
        raise ResumeError("检查点 code_hashes 不是字典")
    for name, digest in code_hashes.items():
        if stored_code.get(name) != digest:
            raise ResumeError(
                f"代码 {name} 的 SHA256 与检查点不符；本脚本修改后拒绝恢复已有试验，"
                "请另建输出目录")
    epoch = checkpoint["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= int(epochs):
        raise ResumeError(f"检查点轮数 {epoch!r} 不在 1..{epochs} 内")
    phase = checkpoint["phase"]
    if phase not in ("trained", "evaluated"):
        raise ResumeError(f"检查点阶段 {phase!r} 无法识别")
    return int(epoch), str(phase)


def restore_model_state(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """严格恢复权重；结构不符时抛 :class:`ResumeError`。"""
    try:
        model.load_state_dict(dict(state), strict=True)
    except RuntimeError as error:
        raise ResumeError(f"检查点权重与当前模型结构不匹配: {error}") from error


def load_baseline_weights(path: Path | str, model: torch.nn.Module, *, base_channels: int,
                          num_pools: int, size: int) -> dict[str, Any]:
    """从旧基线权重起步：结构不符时抛 :class:`IncompatibleWeightsError`。

    绝不静默部分加载：先比对 ``config`` 里的结构，再 ``strict=True`` 载入。
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise ResumeError(f"起点权重不存在: {resolved}")
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise IncompatibleWeightsError(f"起点权重缺少 model_state: {resolved}")
    stored = checkpoint.get("config")
    if not isinstance(stored, Mapping):
        raise IncompatibleWeightsError(f"起点权重缺少 config，无法核对结构: {resolved}")
    found = (int(stored.get("base_channels", -1)), int(stored.get("num_pools", -1)),
             int(stored.get("size", -1)))
    wanted = (int(base_channels), int(num_pools), int(size))
    if found != wanted:
        raise IncompatibleWeightsError(
            f"起点权重结构 (base_channels, num_pools, size)={found} 与当前模型 {wanted} "
            f"不兼容；--init baseline 只接受匹配的结构（原基线为 8 通道 256 分辨率），"
            "禁止静默部分加载，请改用 --init scratch")
    try:
        model.load_state_dict(dict(checkpoint["model_state"]), strict=True)
    except RuntimeError as error:
        raise IncompatibleWeightsError(f"起点权重与当前模型不匹配: {error}") from error
    return {
        "source_checkpoint": resolved.as_posix(),
        "source_checkpoint_sha256": sha256_file(resolved),
        "source_epoch": int(checkpoint.get("epoch", -1)),
        "source_base_channels": found[0],
        "source_num_pools": found[1],
        "source_size": found[2],
        "source_selector": str(checkpoint.get("best_metric", checkpoint.get("selection", ""))),
    }


def code_hashes_snapshot() -> dict[str, str]:
    """本试验依赖的代码文件 SHA256；任何一处变化都会让 ``--resume`` 停止。"""
    role_a = Path(__file__).resolve().parent
    names = ("train_overall_v1.py", "model.py", "data.py", "metrics.py")
    return {name: sha256_file(role_a / name) for name in names}


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


# --------------------------------------------------------------------- 训练
def _history_row(epoch: int, train_metrics: Mapping[str, float], report: Mapping[str, Any],
                 thresholds: Sequence[float], improved: bool) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "train_loss": float(train_metrics["loss"]),
        "train_bce": float(train_metrics["bce"]),
        "train_soft_dice": float(train_metrics["soft_dice"]),
        "selected_threshold": float(report["selected_threshold"]),
        "primary_dice": float(report["primary"]["dice"]),
        "primary_iou": float(report["primary"]["iou"]),
        "threshold_dice": {str(float(t)): float(report["threshold_table"][float(t)]["dice"])
                           for t in thresholds},
        "source_dice": {name: float(info["all"]["dice"])
                        for name, info in report["sources"].items()},
        "improved": bool(improved),
    }


def train(config: RunConfig) -> dict[str, Any]:
    """执行（或继续）一次多来源训练。返回简短状态字典。"""
    _validate_config(config)
    output_dir = Path(config.output_dir)
    mode = plan_mode(output_dir, config.resume)

    if mode == "complete":
        if config.resume:
            print(f"[overall-v1] {output_dir} 已有完成标记 {COMPLETION_NAME}；"
                  "--resume 不重复训练，也不覆盖既有产物。")
            return {"status": "already-complete", "output_dir": output_dir.as_posix()}
        raise TrialStateError(
            f"{output_dir} 已存在完成标记 {COMPLETION_NAME}；拒绝静默覆盖已完成试验。"
            "如需重跑，请另建输出目录。")
    if mode == "partial":
        raise TrialStateError(
            f"{output_dir} 存在未完成检查点 {CHECKPOINT_NAME}；请显式使用 --resume 继续，"
            "脚本不会静默覆盖。")
    if mode == "unknown":
        raise TrialStateError(
            f"{output_dir} 非空但没有 {CHECKPOINT_NAME} / {COMPLETION_NAME}；"
            "无法判断试验状态，拒绝写入。")
    if mode == "missing-checkpoint":
        raise ResumeError(f"--resume 需要 {output_dir / CHECKPOINT_NAME}，但该文件不存在。")

    if config.init in ("baseline", "checkpoint"):
        starting = (config.data_spec.baseline_checkpoint if config.init == "baseline"
                    else config.init_checkpoint)
        if starting is None or not Path(starting).is_file():
            raise ResumeError(f"--init {config.init} 需要已存在的起点权重: {starting}")

    if mode == "fresh":
        seed_everything(config.seed)

    device = resolve_device(config.device)
    assembly = assemble(config.data_spec, verify_frozen=True, verify_images=config.verify_images)
    input_hashes = dict(assembly.hashes)
    if config.init == "checkpoint":
        assert config.init_checkpoint is not None
        input_hashes["init_checkpoint"] = sha256_file(config.init_checkpoint)
    code_hashes = code_hashes_snapshot()
    payload = config.payload()
    config_hash = config_digest(payload)

    model = build_model(base_channels=config.base_channels, num_pools=config.num_pools,
                        input_shape=(3, config.size, config.size)).to(device)
    graph = describe_graph(model)
    if not bool(graph["allowed_only"]):
        raise ConfigError(f"模型含有不允许的算子 {graph['forbidden']}，会破坏 INT8 可映射性")
    estimates = estimate_model(config.base_channels, config.num_pools, config.size)
    if int(estimates["parameters"]) != int(model.num_parameters()):
        raise ConfigError(
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

    if mode == "fresh":
        if config.init in ("baseline", "checkpoint"):
            starting_path = (config.data_spec.baseline_checkpoint if config.init == "baseline"
                             else config.init_checkpoint)
            if starting_path is None:
                raise ResumeError("--init baseline 缺少起点权重路径")
            source_checkpoint = load_baseline_weights(
                starting_path, model,
                base_channels=config.base_channels, num_pools=config.num_pools, size=config.size)
            print(f"[overall-v1] 从已审查权重起步: {source_checkpoint['source_checkpoint']} "
                  f"(epoch={source_checkpoint['source_epoch']}, "
                  f"sha256={source_checkpoint['source_checkpoint_sha256'][:12]}…)", flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_dir / PROTOCOL_NAME, {
            "schema": PROTOCOL_SCHEMA,
            "trial": TRIAL_NAME,
            "created_at": _timestamp(),
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
                "betas": [float(v) for v in config.betas],
                "eps": float(config.eps),
                "weight_decay": float(config.weight_decay),
                "grad_clip": float(config.grad_clip),
                "bce_weight": float(config.bce_weight),
                "dice_weight": float(config.dice_weight),
                "batch_size": int(config.batch_size),
                "epochs": int(config.epochs),
                "seed": int(config.seed),
                "augment": bool(config.augment),
                "num_workers": int(config.num_workers),
                "device": str(device),
            },
            "thresholds": [float(t) for t in config.thresholds],
            "primary_metric": PRIMARY_METRIC,
            "selection_rule": SELECTION_RULE,
            "resume_scope": RESUME_SCOPE,
            "image_hash_checks": int(assembly.image_hash_checks),
            "final_holdout": FINAL_HOLDOUT,
            "known_limits": [
                "验证集/开发集指标空间是 letterbox 后的 NxN 有效区域，不是原始分辨率",
                "GPU 算子未强制确定性；换设备不承诺逐位一致",
                "params/MAC/峰值特征为算术估算，不是实测资源、时序或功耗",
            ],
        })
        atomic_write_json(output_dir / SPLIT_NAME, {
            "train_ids": list(assembly.train_ids),
            "development_ids": {name: list(ids) for name, ids in assembly.dev_ids.items()},
            "digests": dict(assembly.split_digest),
            "note": "训练与开发 ID 全局互斥；开发样本用于每轮评价、统一阈值及最优轮次选择，不参与梯度训练",
        })
        print(f"[overall-v1] 装配完成: 训练 {len(assembly.train_items)}，"
              f"开发 {sum(len(d.items) for d in assembly.dev_sets.values())}，"
              f"config_hash={config_hash[:12]}…", flush=True)
    else:
        checkpoint = load_checkpoint(output_dir / CHECKPOINT_NAME)
        epoch, phase = validate_resume(checkpoint, config_hash=config_hash,
                                       input_hashes=input_hashes, code_hashes=code_hashes,
                                       epochs=config.epochs)
        restore_model_state(model, checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = list(checkpoint["history"])
        stored_best = checkpoint.get("best")
        best = dict(stored_best) if isinstance(stored_best, Mapping) else None
        # A "trained" checkpoint still needs its evaluation, but its gradient
        # updates are complete. Never run that epoch's training loop twice.
        start_epoch = epoch + 1
        pending_eval_epoch = epoch if phase == "trained" else None
        stored_metrics = checkpoint.get("train_metrics")
        if pending_eval_epoch is not None:
            if not isinstance(stored_metrics, Mapping):
                raise ResumeError(
                    "检查点停在评估前但缺少 train_metrics，无法补做该轮历史；拒绝继续")
            pending_train_metrics = {key: float(stored_metrics[key])
                                     for key in ("loss", "bce", "soft_dice")}
        restore_rng_state(checkpoint["rng"])
        protocol_path = output_dir / PROTOCOL_NAME
        if not protocol_path.is_file():
            raise ResumeError(f"缺少 {protocol_path}，无法核对协议")
        if _read_json(protocol_path).get("config_hash") != config_hash:
            raise ResumeError("protocol.json 的配置哈希与当前配置不符")
        _append_json_log(output_dir / RESUME_LOG_NAME, {
            "resumed_at": _timestamp(),
            "checkpoint_epoch": int(epoch),
            "checkpoint_phase": str(phase),
            "next_epoch": int(start_epoch),
            "pending_eval_epoch": pending_eval_epoch,
            "config_hash": config_hash,
            "input_hashes": dict(input_hashes),
            "code_hashes": dict(code_hashes),
            "resume_scope": RESUME_SCOPE,
        })
        print(f"[overall-v1] --resume: 检查点 epoch={epoch} phase={phase}；"
              f"下一轮={start_epoch}", flush=True)

    def persist(epoch: int, phase: str,
                train_metrics: Mapping[str, float] | None = None) -> None:
        atomic_torch_save({
            "schema": CHECKPOINT_SCHEMA,
            "created_at": _timestamp(),
            "trial": TRIAL_NAME,
            "epoch": int(epoch),
            "phase": str(phase),
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
            "rng": capture_rng_state(),
            "history": list(history),
            "best": best,
            "train_metrics": (None if train_metrics is None
                              else {key: float(train_metrics[key])
                                    for key in ("loss", "bce", "soft_dice")}),
            "resume_scope": RESUME_SCOPE,
        }, output_dir / CHECKPOINT_NAME)
        print(f"[overall-v1] 原子检查点已保存: epoch={epoch} phase={phase}", flush=True)

    def evaluate_and_maybe_save_best(epoch: int
                                     ) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
        nonlocal best
        records = evaluate_development(model, assembly.dev_sets, size=config.size,
                                      batch_size=config.batch_size, num_workers=0,
                                      thresholds=config.thresholds, device=device)
        report = build_development_report(records, assembly.dev_sets, config.thresholds)
        entry = best_entry(epoch, report, config.thresholds)
        improved = _better_than(entry, best)
        if improved:
            best = entry
            atomic_torch_save({
                "schema": BEST_SCHEMA,
                "status": "candidate",
                "completed": False,
                "created_at": _timestamp(),
                "trial": TRIAL_NAME,
                "epoch": int(epoch),
                "config": payload,
                "config_hash": config_hash,
                "input_hashes": dict(input_hashes),
                "code_hashes": dict(code_hashes),
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
            atomic_write_json(output_dir / DEV_EVIDENCE_NAME, {
                "epoch": int(epoch),
                "selected_threshold": float(entry["selected_threshold"]),
                "primary_dice": float(entry["primary_dice"]),
                "development_images": int(entry["development_images"]),
                "records": dev_evidence(records, entry["selected_threshold"], config.thresholds),
            })
        return report, records, improved

    if pending_eval_epoch is not None:
        print(f"[overall-v1] 检查点停在 epoch={pending_eval_epoch} 的评估前，补做该轮评估",
              flush=True)
        report, _, improved = evaluate_and_maybe_save_best(pending_eval_epoch)
        if pending_train_metrics is None:  # pragma: no cover - 上面已强制校验
            raise ResumeError("检查点停在评估前但缺少 train_metrics，无法补写历史")
        history[:] = [row for row in history if int(row["epoch"]) != int(pending_eval_epoch)]
        history.append(_history_row(pending_eval_epoch, pending_train_metrics, report,
                                    config.thresholds, improved=bool(improved)))
        atomic_write_json(output_dir / HISTORY_NAME, history)
        persist(pending_eval_epoch, "evaluated", pending_train_metrics)

    for epoch in range(start_epoch, int(config.epochs) + 1):
        dataset = TrainDataset(assembly.train_items, size=config.size, seed=config.seed,
                              augment=config.augment, epoch=epoch)
        loader = make_train_loader(dataset, config.batch_size, config.num_workers,
                                   config.seed, epoch)
        train_metrics = run_train_epoch(model, loader, optimizer, device,
                                       bce_weight=config.bce_weight,
                                       dice_weight=config.dice_weight,
                                       grad_clip=config.grad_clip)
        persist(epoch, "trained", train_metrics)
        report, _, improved = evaluate_and_maybe_save_best(epoch)
        history[:] = [row for row in history if int(row["epoch"]) != int(epoch)]
        row = _history_row(epoch, train_metrics, report, config.thresholds,
                           improved=bool(improved))
        history.append(row)
        atomic_write_json(output_dir / HISTORY_NAME, history)
        persist(epoch, "evaluated")
        print(f"[overall-v1] epoch {epoch}/{config.epochs} "
              f"train_loss={row['train_loss']:.6f} primary_dice={row['primary_dice']:.6f} "
              f"threshold={row['selected_threshold']:.2f} improved={row['improved']}", flush=True)

    completion = {
        "schema": COMPLETION_SCHEMA,
        "trial": TRIAL_NAME,
        "completed_at": _timestamp(),
        "completed_epochs": int(config.epochs),
        "config_hash": config_hash,
        "primary_metric": PRIMARY_METRIC,
        "selection_rule": SELECTION_RULE,
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
    atomic_write_json(output_dir / COMPLETION_NAME, completion)
    print(f"[overall-v1] 训练完成: best epoch="
          f"{None if best is None else best['epoch']} -> {output_dir / COMPLETION_NAME}",
          flush=True)
    return {"status": "completed", "output_dir": output_dir.as_posix(),
            "best": completion["best"], "completed_epochs": int(config.epochs)}


# --------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_overall_v1.py",
        description="总体精度优先的多来源 Tiny U-Net 训练器（可恢复、638 开发集主指标）",
    )
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="试验输出目录（已完成的试验会被拒绝覆盖）")
    parser.add_argument("--init", choices=("baseline", "scratch", "checkpoint"), default="baseline",
                        help="baseline=从 runs/role_a_tinyunet/best.pt 起步（仅 8 通道）；"
                             "checkpoint=从指定同结构权重起步；scratch=从头训练")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="仅配合 --init checkpoint；必须与当前结构完全相同")
    parser.add_argument("--base-channels", type=int, default=8,
                        help=f"基通道，只允许 {list(ALLOWED_BASE_CHANNELS)}")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--resume", action="store_true",
                        help="从 latest.pt 的最后一个完整轮次继续；配置/输入/代码不符即停")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto",
                        help="auto/cpu/cuda/cuda:0/mps（不可用时直接报错，不静默降级）")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--size", type=int, default=256, help="letterbox 输出边长，默认 256")
    parser.add_argument("--num-pools", type=int, default=3, help="2x2 MaxPool 次数，1..3")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-augment", action="store_true",
                        help="关闭训练集几何/颜色增强（验证预处理永远不变）")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="<=0 表示关闭梯度裁剪")
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--verify-image-hashes", action="store_true",
                        help="额外逐图比对来源清单里的 image/mask SHA256（慢，但最严格）")
    parser.add_argument("--project", type=Path, default=PROJECT,
                        help="项目根目录（默认本仓库根；用于注入其它数据根做自测）")
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """把 argparse 结果转成 :class:`RunConfig`（此时只构造路径，不读数据）。"""
    return RunConfig(
        output_dir=Path(args.output_dir).resolve(),
        init=str(args.init),
        init_checkpoint=(None if args.init_checkpoint is None
                         else Path(args.init_checkpoint).resolve()),
        base_channels=int(args.base_channels),
        num_pools=int(args.num_pools),
        size=int(args.size),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        device=str(args.device),
        num_workers=int(args.num_workers),
        data_spec=build_default_spec(Path(args.project)),
        grad_clip=float(args.grad_clip),
        bce_weight=float(args.bce_weight),
        dice_weight=float(args.dice_weight),
        augment=not bool(args.no_augment),
        resume=bool(args.resume),
        verify_images=bool(args.verify_image_hashes),
        thresholds=THRESHOLDS,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        config = config_from_args(args)
        train(config)
        return 0
    except (AssemblyError, ConfigError, ResumeError, TrialStateError,
            FileNotFoundError, ValueError) as error:
        print(f"[错误] {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
