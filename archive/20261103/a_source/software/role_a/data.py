"""ISIC 2016 Task 1 的数据配对、letterbox 预处理与可复现划分。

本模块只做本地文件读写，**不下载数据**、不修改 ``data/raw/`` 中的原始文件。

约定
----
官方训练集 900 张：图像 ``ISIC_*.jpg``，掩膜 ``ISIC_*_Segmentation.png``。
掩膜文件名 = 图像文件名主干的 ``_Segmentation`` 后缀版本，例如::

    ISIC_0000001.jpg  <->  ISIC_0000001_Segmentation.png

配对规则严格按此命名，不做模糊匹配：缺失、重复、无法配对的文件都会抛出
:class:`DataLayoutError`，而不是被静默跳过。

预处理：保持宽高比缩放（图像双线性、掩膜最近邻）后居中补零到 ``size x size``，
同时产生 ``valid`` 掩膜标记非补边像素。图像的补边像素值为 0，因此**不是**由
像素值推断有效区域，而是由 letterbox 的几何信息生成。
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

__all__ = [
    "DataLayoutError",
    "LetterboxInfo",
    "DatasetItem",
    "IsicPaths",
    "letterbox",
    "image_stem",
    "enumerate_isic_pairs",
    "render_split",
    "load_split",
    "export_split",
    "export_split_csv",
    "ISICDataset",
    "isic_collate",
    "IMAGE_SUFFIXES",
    "MASK_SUFFIXES",
    "MASK_SUFFIX",
    "MASK_BINARY_THRESHOLD",
    "DEFAULT_SEED",
    "SPLIT_ROLES",
]

IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg")
MASK_SUFFIXES: tuple[str, ...] = (".png",)
MASK_SUFFIX: str = "_Segmentation"
SPLIT_ROLES: tuple[str, ...] = ("train", "val", "test")
DEFAULT_SEED: int = 42
# 掩膜二值化阈值：官方掩膜是 0/255，这里不假设严格二值，
# 而是按 >= 127 归一化为 0/1，并把非严格二值的文件数记录到划分报告里。
MASK_BINARY_THRESHOLD: int = 127


class DataLayoutError(ValueError):
    """数据目录布局与官方 ISIC 命名不一致（缺失、重复、无法配对、尺寸不符等）。"""


@dataclass(frozen=True)
class LetterboxInfo:
    """letterbox 的几何信息。

    属性:
        scale: 等比缩放系数。
        resized: 缩放后、补边前的 ``(width, height)``。
        offset: 左上角补边偏移 ``(left, top)``。
        canvas: 画布 ``(width, height)``，两个方向都为 ``size``。
        original_size: 原始 ``(width, height)``（已应用 EXIF 旋转）。
    """

    scale: float
    resized: tuple[int, int]
    offset: tuple[int, int]
    canvas: tuple[int, int]
    original_size: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "resized": list(self.resized),
            "offset": list(self.offset),
            "canvas": list(self.canvas),
            "original_size": list(self.original_size),
        }


@dataclass(frozen=True)
class IsicPaths:
    """一对已配对的图像/掩膜路径。``mask`` 在仅推理时为 ``None``。"""

    stem: str
    image: Path
    mask: Path | None

    @property
    def has_mask(self) -> bool:
        return self.mask is not None

    def to_dict(self, root: Path | None = None) -> dict[str, Any]:
        def _rel(path: Path | None) -> str | None:
            if path is None:
                return None
            if root is not None:
                try:
                    return path.relative_to(root).as_posix()
                except ValueError:
                    return path.as_posix()
            return path.as_posix()

        return {"stem": self.stem, "image": _rel(self.image), "mask": _rel(self.mask)}


@dataclass(frozen=True)
class DatasetItem:
    """``ISICDataset`` 的一条样本。

    属性:
        image: ``float32`` 张量 ``[3, S, S]``，取值 ``[0, 1]``，补边为 0。
        mask: ``float32`` 张量 ``[1, S, S]``，取值 ``{0.0, 1.0}``；无真值时为 ``None``。
        valid: ``float32`` 张量 ``[1, S, S]``，非补边区域为 1。
        filename: 图像文件名（不含目录）。
        original_size: 原始 ``(width, height)``。
        letterbox: :class:`LetterboxInfo`；仅便于调试，不参与训练。
        mask_was_binary: 掩膜原始像素是否严格只含 0/255。
    """

    image: torch.Tensor
    mask: torch.Tensor | None
    valid: torch.Tensor
    filename: str
    original_size: tuple[int, int]
    letterbox: LetterboxInfo
    mask_was_binary: bool


# --------------------------------------------------------------------- 配对
def image_stem(path: Path) -> str:
    """返回 ISIC 图像标识（去掉扩展名），例如 ``ISIC_0000001``。"""
    return Path(path).stem


def _scan(directory: Path, suffixes: Sequence[str], what: str) -> dict[str, list[Path]]:
    if not directory.is_dir():
        raise DataLayoutError(f"{what}目录不存在或不是目录: {directory}")
    found: dict[str, list[Path]] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        found.setdefault(image_stem(path), []).append(path)
    return found


def _detect_duplicates(mapping: Mapping[str, Sequence[Path]], what: str) -> None:
    duplicates = {stem: [p.as_posix() for p in paths]
                  for stem, paths in mapping.items() if len(paths) > 1}
    if duplicates:
        detail = json.dumps(duplicates, indent=2, ensure_ascii=False)
        raise DataLayoutError(f"{what}存在重复标识（同一 ISIC 标识对应多个文件）:\n{detail}")


def enumerate_isic_pairs(
    image_dir: Path | str,
    mask_dir: Path | str | None = None,
    require_mask: bool = True,
) -> list[IsicPaths]:
    """扫描目录并按官方命名配对，返回按标识排序的列表。

    参数:
        image_dir: 图像目录（递归扫描 ``ISIC_*.jpg`` / ``.jpeg``）。
        mask_dir: 掩膜目录；``None`` 时表示掩膜与图像同目录。
        require_mask: ``True`` 时任何缺失配对都视为错误（训练/验证）；
            ``False`` 时允许图像没有掩膜（例如官方测试集若未提供标注）。

    异常:
        DataLayoutError: 目录不存在、重复标识、单边缺失、无任何配对、
            或掩膜文件名不含 ``_Segmentation`` 约定。
    """
    images = _scan(Path(image_dir), IMAGE_SUFFIXES, "图像")
    if not images:
        raise DataLayoutError(
            f"在 {Path(image_dir)} 下没有找到任何 ISIC 图像（{'/'.join(IMAGE_SUFFIXES)}）"
        )
    _detect_duplicates(images, "图像")

    masks: dict[str, list[Path]] = {}
    if mask_dir is not None:
        masks = _scan(Path(mask_dir), MASK_SUFFIXES, "掩膜")
        _detect_duplicates(masks, "掩膜")
    elif require_mask:
        # 同目录扫描：只收 *_Segmentation.png，避免把其他 PNG 当掩膜。
        same_dir = _scan(Path(image_dir), MASK_SUFFIXES, "掩膜")
        masks = {stem: paths for stem, paths in same_dir.items() if stem.endswith(MASK_SUFFIX)}
        _detect_duplicates(masks, "掩膜")

    expected: dict[str, str] = {}
    for stem in images:
        expected[stem] = stem + MASK_SUFFIX

    missing: list[str] = []
    if require_mask:
        missing = sorted(stem for stem, mask_stem in expected.items() if mask_stem not in masks)
    orphan = sorted(mask_stem for mask_stem in masks
                    if mask_stem[: -len(MASK_SUFFIX)] not in images) if masks else []

    problems: list[str] = []
    if missing:
        problems.append(f"缺少掩膜的图像 {len(missing)} 个: {missing[:10]}")
    if orphan:
        problems.append(f"没有对应图像的掩膜 {len(orphan)} 个: {orphan[:10]}")
    if problems:
        raise DataLayoutError("；".join(problems))

    pairs: list[IsicPaths] = []
    for stem in sorted(images):
        image_path = images[stem][0]
        candidates = masks.get(expected[stem], [])
        mask_path = candidates[0] if candidates else None
        pairs.append(IsicPaths(stem=stem, image=image_path, mask=mask_path))
    return pairs


# ---------------------------------------------------------------- letterbox
def _resized_shape(width: int, height: int, size: int) -> tuple[int, int, float]:
    if width <= 0 or height <= 0:
        raise DataLayoutError(f"图像尺寸必须为正，收到 {(width, height)}")
    scale = size / max(width, height)
    resized = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    if max(resized) > size:
        # 浮点四舍五入可能越界 1 像素，收紧到画布内。
        excess = max(resized) - size
        resized = (resized[0] - (excess if resized[0] == max(resized) else 0),
                   resized[1] - (excess if resized[1] == max(resized) else 0))
    return resized[0], resized[1], scale


def letterbox(
    image_path: Path | str,
    mask_path: Path | str | None = None,
    size: int = 256,
    pad_value: int = 0,
    mask_threshold: int = MASK_BINARY_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, LetterboxInfo, bool]:
    """等比缩放 + 居中补零，图像双线性、掩膜最近邻。

    返回 ``(image, mask, valid, info, mask_was_binary)``：

    * ``image``: ``uint8`` ``[size, size, 3]``，补边为 ``pad_value``；
    * ``mask``: ``uint8`` ``[size, size]``，取值 ``{0, 1}``；未给掩膜时为 ``None``；
    * ``valid``: ``uint8`` ``[size, size]``，取值 ``{0, 1}``，非补边为 1；
    * ``info``: :class:`LetterboxInfo`；
    * ``mask_was_binary``: 原掩膜像素是否严格只含 0/255。

    异常:
        DataLayoutError: 图像/掩膜尺寸不一致、文件不存在或掩膜为空。
    """
    if size <= 0:
        raise DataLayoutError(f"size 必须为正，收到 {size}")
    image_path, mask_path = Path(image_path), (Path(mask_path) if mask_path else None)
    if not image_path.is_file():
        raise DataLayoutError(f"图像文件不存在: {image_path}")
    if mask_path is not None and not mask_path.is_file():
        raise DataLayoutError(f"掩膜文件不存在: {mask_path}")

    with Image.open(image_path) as handle:
        rgb = ImageOps.exif_transpose(handle).convert("RGB")
        original_size = rgb.size
        resized_w, resized_h, scale = _resized_shape(original_size[0], original_size[1], size)
        offset = ((size - resized_w) // 2, (size - resized_h) // 2)
        canvas = Image.new("RGB", (size, size), (pad_value, pad_value, pad_value))
        canvas.paste(rgb.resize((resized_w, resized_h), Image.Resampling.BILINEAR), offset)
        image = np.asarray(canvas, dtype=np.uint8)

    mask: np.ndarray | None = None
    mask_was_binary = True
    if mask_path is not None:
        with Image.open(mask_path) as handle:
            gray = ImageOps.exif_transpose(handle).convert("L")
            if gray.size != original_size:
                raise DataLayoutError(
                    f"图像与掩膜尺寸不一致: {image_path.name} {original_size} "
                    f"vs {mask_path.name} {gray.size}"
                )
            values = np.asarray(gray, dtype=np.uint8)
            mask_was_binary = bool(np.isin(values, (0, 255)).all())
            resized = np.asarray(
                gray.resize((resized_w, resized_h), Image.Resampling.NEAREST), dtype=np.uint8
            )
        canvas_mask = np.full((size, size), pad_value, dtype=np.uint8)
        canvas_mask[offset[1]:offset[1] + resized_h, offset[0]:offset[0] + resized_w] = resized
        mask = (canvas_mask >= mask_threshold).astype(np.uint8)

    valid = np.zeros((size, size), dtype=np.uint8)
    valid[offset[1]:offset[1] + resized_h, offset[0]:offset[0] + resized_w] = 1

    if image.shape != (size, size, 3):
        raise DataLayoutError(f"letterbox 结果 shape 异常: {image.shape}")
    if mask is not None and mask.shape != (size, size):
        raise DataLayoutError(f"掩膜 letterbox 结果 shape 异常: {mask.shape}")
    if int(valid.sum()) == 0:
        raise DataLayoutError(f"valid 区域为空: {image_path.name}")
    return image, mask, valid, LetterboxInfo(scale=scale, resized=(resized_w, resized_h),
                                             offset=offset, canvas=(size, size),
                                             original_size=(original_size[0], original_size[1])), mask_was_binary


# -------------------------------------------------------------------- 划分
def render_split(
    pairs: Sequence[IsicPaths],
    seed: int = DEFAULT_SEED,
    val_fraction: float = 0.2,
    stratify_by_area: bool = True,
    report_dir: Path | str | None = None,
) -> dict[str, list[str]]:
    """把配对列表划分成 ``train`` / ``val``，返回 ``{"train": [...], "val": [...]}``。

    同一 ``seed`` + 同一 ``pairs`` 顺序 ⇒ 完全相同的划分（确定性）。

    参数:
        pairs: :func:`enumerate_isic_pairs` 的输出，必须按 stem 排序。
        seed: 随机种子。
        val_fraction: 验证集比例，必须在 ``(0, 1)`` 内。
        stratify_by_area: 为 ``True`` 时按掩膜前景占比分成 5 档后再分配，
            让训练/验证的前景比例分布接近（不是患者级划分，见 README 限制）。
            需要掩膜存在，否则自动退化为随机划分。
        report_dir: 若给出，则把非二值掩膜数量等元信息写到
            ``<report_dir>/split_report.json``。

    异常:
        DataLayoutError: 样本数不足 2、``val_fraction`` 越界、重复 stem 或 mask 缺失。
    """
    if not 0.0 < val_fraction < 1.0:
        raise DataLayoutError(f"val_fraction 必须在 (0, 1) 内，收到 {val_fraction}")
    if len(pairs) < 2:
        raise DataLayoutError(f"至少需要 2 个样本才能划分，收到 {len(pairs)}")
    stems = [item.stem for item in pairs]
    if len(set(stems)) != len(stems):
        raise DataLayoutError("配对列表中存在重复 stem，无法划分")

    target_val = max(1, int(round(len(pairs) * val_fraction)))
    target_val = min(target_val, len(pairs) - 1)
    rng = random.Random(seed)

    groups: list[list[IsicPaths]] = []
    if stratify_by_area and all(item.has_mask for item in pairs):
        areas: list[tuple[float, IsicPaths]] = []
        non_binary: list[str] = []
        for item in pairs:
            with Image.open(item.mask) as handle:  # type: ignore[arg-type]
                values = np.asarray(handle.convert("L"), dtype=np.uint8)
            if values.size == 0:
                raise DataLayoutError(f"掩膜为空: {item.mask}")
            if not bool(np.isin(values, (0, 255)).all()):
                non_binary.append(item.stem)
            areas.append((float((values >= MASK_BINARY_THRESHOLD).mean()), item))
        for _, bucket in _area_buckets(areas):
            rng.shuffle(bucket)
            groups.append(bucket)
        if report_dir is not None:
            report = {
                "seed": seed,
                "val_fraction": val_fraction,
                "stratify_by_area": True,
                "area_bucket_count": 5,
                "non_binary_mask_count": len(non_binary),
                "non_binary_mask_stems": sorted(non_binary),
                "note": "非二值掩膜将按 >= 127 阈值二值化；数值来自原始掩膜统计",
            }
            path = Path(report_dir) / "split_report.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        groups = [list(pairs)]
        rng.shuffle(groups[0])

    val: list[str] = []
    train: list[str] = []
    # 在排序后的条目里轮流取，保证 train/val 的前景分布形状接近。
    for group in groups:
        for index, item in enumerate(group):
            (val if index % 2 == 0 else train).append(item.stem)
    if len(val) > target_val:  # 把多余的最近条目移回 train，保持目标比例。
        for stem in val[target_val:]:
            val.remove(stem)
            train.append(stem)
    elif len(val) < target_val:
        for stem in list(train):
            if len(val) >= target_val:
                break
            train.remove(stem)
            val.append(stem)

    if set(train) & set(val):
        raise DataLayoutError("划分出现重叠，逻辑错误")
    if sorted(train + val) != sorted(stems):
        raise DataLayoutError("划分未覆盖全部样本，逻辑错误")
    for name, part in (("train", train), ("val", val)):
        if not part:
            raise DataLayoutError(f"{name} 划分为空（样本数 {len(pairs)}，val_fraction {val_fraction}）")
    train.sort()
    val.sort()
    return {"train": train, "val": val}


def _area_buckets(areas: Sequence[tuple[float, IsicPaths]], buckets: int = 5) -> list[tuple[float, list[IsicPaths]]]:
    """按前景占比排序后等分成若干档，返回 ``[(该档平均占比, 成员), ...]``。"""
    ordered = sorted(areas, key=lambda pair: (pair[0], pair[1].stem))
    chunk = max(1, len(ordered) // buckets)
    result: list[tuple[float, list[IsicPaths]]] = []
    for start in range(0, len(ordered), chunk):
        piece = ordered[start:start + chunk]
        if not piece:
            continue
        mean_area = sum(area for area, _ in piece) / len(piece)
        result.append((mean_area, [item for _, item in piece]))
    return result


def load_split(path: Path | str) -> dict[str, list[str]]:
    """读取 :func:`export_split` 写出的 JSON 划分文件。"""
    split_path = Path(path)
    if not split_path.is_file():
        raise DataLayoutError(f"划分文件不存在: {split_path}")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "splits" not in payload:
        raise DataLayoutError(f"划分文件格式不符合预期（缺少 splits 字段）: {split_path}")
    splits = payload["splits"]
    if not isinstance(splits, dict):
        raise DataLayoutError(f"划分文件 splits 字段必须是对象: {split_path}")
    result: dict[str, list[str]] = {}
    for name in SPLIT_ROLES:
        if name not in splits:
            continue
        value = splits[name]
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise DataLayoutError(f"划分 {name} 必须是字符串列表: {split_path}")
        result[name] = sorted(value)
    if "train" not in result or "val" not in result:
        raise DataLayoutError(f"划分文件必须同时包含 train 与 val: {split_path}")
    if set(result["train"]) & set(result["val"]):
        raise DataLayoutError(f"划分文件 train 与 val 重叠: {split_path}")
    if "test" in result and (set(result["test"]) &
                             (set(result["train"]) | set(result["val"]))):
        raise DataLayoutError(f"划分文件 test 与 train/val 重叠: {split_path}")
    return result


def export_split(
    split: Mapping[str, Sequence[str]],
    output_path: Path | str,
    seed: int,
    pairs: Sequence[IsicPaths] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """把划分与排序后的文件名写成 JSON，供复现与审计。

    ``test`` 只有在 ``split`` 显式包含时才写入，且不会与 ``train``/``val`` 混用。
    """
    output = Path(output_path)
    for name in ("train", "val"):
        if name not in split:
            raise DataLayoutError(f"划分缺少必需的 {name} 角色")
    overlap = set(split.get("train", ())) & set(split.get("val", ()))
    if overlap:
        raise DataLayoutError(f"train 与 val 重叠: {sorted(overlap)[:10]}")
    if "test" in split:
        leak = set(split["test"]) & (set(split["train"]) | set(split["val"]))
        if leak:
            raise DataLayoutError(f"test 与 train/val 重叠: {sorted(leak)[:10]}")
    index = {item.stem: item for item in (pairs or [])}
    payload: dict[str, Any] = {
        "dataset": "ISIC 2016 Task 1 (official train partition)",
        "seed": seed,
        "roles": {name: len(split[name]) for name in SPLIT_ROLES if name in split},
        "splits": {name: sorted(split[name]) for name in SPLIT_ROLES if name in split},
        "pairs": {stem: item.to_dict() for stem, item in sorted(index.items())},
        "note": ("test 角色仅用于最终评估；最佳 checkpoint 只由 val 选择。"
                 "划分按文件名记录，不含患者级信息。"),
    }
    if extra:
        payload["extra"] = dict(extra)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def export_split_csv(
    split: Mapping[str, Sequence[str]],
    output_path: Path | str,
    pairs: Sequence[IsicPaths] | None = None,
    root: Path | str | None = None,
) -> Path:
    """把划分写成逐行 CSV（``role,stem,image,mask``）。"""
    output = Path(output_path)
    index = {item.stem: item for item in (pairs or [])}
    root_path = Path(root) if root is not None else None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["role", "stem", "image", "mask"])
        for name in SPLIT_ROLES:
            for stem in sorted(split.get(name, ())):
                item = index.get(stem)
                if item is None:
                    writer.writerow([name, stem, "", ""])
                else:
                    record = item.to_dict(root_path)
                    writer.writerow([name, stem, record["image"], record["mask"] or ""])
    return output


# ------------------------------------------------------------------ Dataset
class ISICDataset(Dataset):
    """在内存中按需读取的 ISIC 数据集（图像 + 掩膜 + valid），不做磁盘缓存。

    参数:
        items: 已配对的 :class:`IsicPaths` 列表（顺序即索引顺序）。
        root: 用于错误信息与相对路径记录的根目录，可为 ``None``。
        size: letterbox 输出边长，默认 256。
        pad_value: 补边像素值，默认 0（任务要求的补零）。
        augment: 默认 ``False``。为 ``True`` 时只做水平/垂直翻转，且**同时**
            作用于图像、掩膜与 valid 掩膜。
    """

    def __init__(
        self,
        items: Sequence[IsicPaths],
        root: Path | str | None = None,
        size: int = 256,
        pad_value: int = 0,
        augment: bool = False,
    ) -> None:
        if not items:
            raise DataLayoutError("数据集为空：没有任何已配对的样本")
        if size <= 0:
            raise DataLayoutError(f"size 必须为正，收到 {size}")
        self.items: tuple[IsicPaths, ...] = tuple(items)
        self.root = Path(root) if root is not None else None
        self.size = int(size)
        self.pad_value = int(pad_value)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> DatasetItem:
        item = self.items[index]
        image, mask, valid, info, mask_was_binary = letterbox(
            item.image, item.mask, size=self.size, pad_value=self.pad_value
        )
        if self.augment:
            # 翻转同时作用于图像与掩膜，因此不会破坏二者的几何对齐；
            # valid 是非补边矩形，任何翻转/转置后仍然是非补边矩形，故同步翻转即可。
            if torch.rand(1).item() < 0.5:
                image, mask, valid = image[:, ::-1], None if mask is None else mask[:, ::-1], valid[:, ::-1]
            if torch.rand(1).item() < 0.5:
                image, mask, valid = image[::-1], None if mask is None else mask[::-1], valid[::-1]
            image, mask, valid = (np.ascontiguousarray(image),
                                  None if mask is None else np.ascontiguousarray(mask),
                                  np.ascontiguousarray(valid))
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
        if tensor.shape != (3, self.size, self.size):
            raise DataLayoutError(f"图像张量 shape 异常: {tuple(tensor.shape)}")
        mask_tensor = None
        if mask is not None:
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        valid_tensor = torch.from_numpy(np.ascontiguousarray(valid)).float().unsqueeze(0)
        return DatasetItem(
            image=tensor,
            mask=mask_tensor,
            valid=valid_tensor,
            filename=item.image.name,
            original_size=info.original_size,
            letterbox=info,
            mask_was_binary=mask_was_binary,
        )


def isic_collate(batch: Iterable[DatasetItem]) -> dict[str, Any]:
    """把 :class:`DatasetItem` 列表拼成 batch。

    有掩膜时 ``mask`` 是 ``[N, 1, S, S]`` 张量；全部无掩膜时为 ``None``。
    ``filename`` 等元信息保留为 Python 列表（不能用 ``weights_only`` 反序列化的
    自定义对象一概不进入 checkpoint）。
    """
    rows = list(batch)
    if not rows:
        raise DataLayoutError("collate 收到空 batch")
    images = torch.stack([row.image for row in rows], dim=0)
    valids = torch.stack([row.valid for row in rows], dim=0)
    masks = None
    if all(row.mask is not None for row in rows):
        masks = torch.stack([row.mask for row in rows], dim=0)  # type: ignore[arg-type]
    elif any(row.mask is not None for row in rows):
        raise DataLayoutError("同一 batch 中部分样本有掩膜、部分没有；请分开加载")
    return {
        "image": images,
        "mask": masks,
        "valid": valids,
        "filename": [row.filename for row in rows],
        "original_size": [tuple(row.original_size) for row in rows],
        "mask_was_binary": [row.mask_was_binary for row in rows],
    }
