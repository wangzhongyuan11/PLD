"""A11: v4 INT8 掩膜后处理试验（identity / fill holes / largest component / combined）。

设计约束：

* 只读既有 v4 权重与 INT8 spec，不训练、不改模型/量化器/旧 run。
* 只在 letterbox 的 ``valid`` 区域内处理；``valid`` 之外永远保持背景。
* 指标定义、面积分组、阈值口径全部复用 ``train_overall_v1``，避免重抄。
* 只用 NumPy / PyTorch / 标准库，不引入新依赖。

连通性约定（与任务书一致）：

* 填洞：背景取 **4 连通**。背景分量若能从 ``valid`` 边缘（含图像边界）连通到达即视为
  外部，不填；否则为内部空洞，填成前景。
* 最大分量：前景取 **8 连通**。面积并列时取扫描顺序最靠上、再最靠左的分量
  （即行优先最早出现的像素最小的那个分量）。

实现与复杂度：

* 背景洪泛与前景标记都是**标准队列/栈式单次遍历**（显式 DFS 栈，等价于 BFS）：
  每个像素最多入栈一次，时间 O(H*W)，临时空间 O(H*W)。
* 画布四周补一圈哨兵，邻居访问用平坦标量下标算术，避免逐像素边界判断。
* 临时空间：每个 bool/uint8 画布 (H+2)*(W+2) 字节；256x256 时约 66 KB/画布，
  同时最多约 5 个画布（free、visited、标签 int32 等），峰值 < 1.5 MB/图。
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from software.role_a import quant
from software.role_a import train_overall_v1 as overall
from software.role_a.data import ISICDataset, isic_collate

THRESHOLD: float = 0.45
CANDIDATES: tuple[str, ...] = ("identity", "fill_holes", "largest_component", "combined")
# 并列时的固定优先顺序（任务书规定）
TIE_BREAK_ORDER: tuple[str, ...] = CANDIDATES
EXPECTED_IDENTITY_DICE: float = 0.840632936148671
IDENTITY_TOLERANCE: float = 1e-9

_FOUR_OFFSETS = ("up", "down", "left", "right")


def _padded_bytes(mask: np.ndarray) -> tuple[bytearray, int, int]:
    """把 bool 画布放进四周补 0 的平坦 bytearray，返回 (数据, 行宽, 行数)。"""
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    padded = np.zeros((height + 2, width + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask
    return bytearray(padded.ravel().tobytes()), width + 2, height + 2


def _flood_from_edge(free: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """4 连通背景洪泛：从 ``valid`` 边缘可达的背景像素（单次遍历 O(H*W)）。"""
    free = np.asarray(free, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    height, width = free.shape
    data, stride, _rows = _padded_bytes(free)
    outside = np.ones((height + 2, width + 2), dtype=bool)
    outside[1:-1, 1:-1] = ~valid
    neighbour_outside = np.zeros((height + 2, width + 2), dtype=bool)
    neighbour_outside[1:-1, 1:-1] = (outside[:-2, 1:-1] | outside[2:, 1:-1]
                                     | outside[1:-1, :-2] | outside[1:-1, 2:])
    seeds = np.flatnonzero((np.frombuffer(bytes(data), dtype=np.uint8)
                            .reshape(height + 2, width + 2) > 0) & neighbour_outside)
    reachable = bytearray(len(data))
    stack = deque(int(index) for index in seeds)
    while stack:
        index = stack.pop()
        if reachable[index]:
            continue
        reachable[index] = 1
        up = index - stride
        if data[up] and not reachable[up]:
            stack.append(up)
        down = index + stride
        if data[down] and not reachable[down]:
            stack.append(down)
        left = index - 1
        if data[left] and not reachable[left]:
            stack.append(left)
        right = index + 1
        if data[right] and not reachable[right]:
            stack.append(right)
    shaped = np.frombuffer(bytes(reachable), dtype=np.uint8).reshape(height + 2, width + 2)
    return shaped[1:-1, 1:-1].astype(bool)


def fill_holes(pred: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """填充前景内部的封闭空洞（背景 4 连通，以 ``valid`` 边缘为外部）。"""
    pred = np.asarray(pred, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if pred.shape != valid.shape:
        raise ValueError("pred 与 valid 形状不一致")
    inside = pred & valid                      # valid 之外永远是背景
    free = valid & ~inside
    if not free.any():
        return inside
    reachable = _flood_from_edge(free, valid)
    return inside | (free & ~reachable)


def _label_components(pred: np.ndarray, *, connectivity: int) -> tuple[np.ndarray, int, list, list]:
    """前景连通分量标记；返回 (labels, count, sizes, first_positions)。

    ``sizes[i-1]`` 与 ``first_positions[i-1]`` 对应标签 ``i``；位置为行优先最早出现的
    像素坐标 ``(row, col)``，标签按该顺序从 1 开始分配。单次遍历 O(H*W)。
    """
    mask = np.asarray(pred, dtype=bool)
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity 只支持 4 或 8，收到 {connectivity}")
    height, width = mask.shape
    data, stride, _rows = _padded_bytes(mask)
    visited = bytearray(len(data))
    labels = np.zeros((height, width), dtype=np.int32)
    order = np.flatnonzero(
        np.frombuffer(bytes(data), dtype=np.uint8) > 0)  # 行优先
    sizes: list[int] = []
    positions: list[tuple[int, int]] = []
    label = 0
    for start in order.tolist():
        if visited[start]:
            continue
        label += 1
        members: list[int] = []
        stack = deque([int(start)])
        while stack:
            index = stack.pop()
            if visited[index]:
                continue
            visited[index] = 1
            members.append(index)
            up = index - stride
            down = index + stride
            if data[up] and not visited[up]:
                stack.append(up)
            if data[down] and not visited[down]:
                stack.append(down)
            if data[index - 1] and not visited[index - 1]:
                stack.append(index - 1)
            if data[index + 1] and not visited[index + 1]:
                stack.append(index + 1)
            if connectivity == 8:
                if data[up - 1] and not visited[up - 1]:
                    stack.append(up - 1)
                if data[up + 1] and not visited[up + 1]:
                    stack.append(up + 1)
                if data[down - 1] and not visited[down - 1]:
                    stack.append(down - 1)
                if data[down + 1] and not visited[down + 1]:
                    stack.append(down + 1)
        flat = np.asarray(members, dtype=np.int64)
        rows, cols = np.divmod(flat, stride)
        rows -= 1
        cols -= 1
        labels[rows, cols] = label
        top = int(rows.min())
        sizes.append(int(flat.size))
        positions.append((top, int(cols[rows == top].min())))
    return labels, label, sizes, positions


def label_components(pred: np.ndarray, *, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """前景连通分量标记（默认 8 连通），返回 (labels, 分量数)。"""
    labels, count, _sizes, _positions = _label_components(pred, connectivity=connectivity)
    return labels, count


def largest_component(pred: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """只保留最大 8 连通前景分量；面积并列时取最靠上、再最靠左者。"""
    pred = np.asarray(pred, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if pred.shape != valid.shape:
        raise ValueError("pred 与 valid 形状不一致")
    inside = pred & valid
    if not inside.any():
        return np.zeros_like(inside)
    labels, count, sizes, positions = _label_components(inside, connectivity=8)
    if count <= 1:
        return inside.copy()
    best_size = max(sizes)
    tied = [index for index, size in enumerate(sizes) if size == best_size]
    best = min(tied, key=lambda index: positions[index])
    return labels == (best + 1)


def apply_candidate(name: str, pred: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """按名称应用候选；四种之外一律报错（不按图自适应）。"""
    pred = np.asarray(pred, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if name == "identity":
        return pred & valid
    if name == "fill_holes":
        return fill_holes(pred & valid, valid)
    if name == "largest_component":
        return largest_component(pred & valid, valid)
    if name == "combined":
        return fill_holes(largest_component(pred & valid, valid), valid)
    raise ValueError(f"未知候选: {name}")


# ------------------------------------------------------------------- 汇总
def _per_image_metrics(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, bool, int]:
    pred_sum = int(pred.sum())
    true_sum = int(truth.sum())
    intersection = int((pred & truth).sum())
    dice, iou = overall._binary_dice_iou(pred_sum, true_sum, intersection)
    return float(dice), float(iou), pred_sum == 0, pred_sum


def _summarise(rows: Sequence[dict[str, Any]], dev_sets, threshold: float) -> dict[str, Any]:
    return overall.build_development_report(rows, dev_sets, thresholds=(float(threshold),))


def _candidate_rows(name: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"source": row["source"], "stem": row["stem"], "area_ratio": row["area_ratio"],
             "band": row["band"], "dice": [row["candidates"][name]["dice"]],
             "iou": [row["candidates"][name]["iou"]],
             "empty_prediction": [row["candidates"][name]["empty_prediction"]]}
            for row in records]


def _compare(identity: list[float], other: list[float]) -> dict[str, Any]:
    better = worse = same = 0
    deltas: list[tuple[float, int]] = []
    for index, (left, right) in enumerate(zip(identity, other)):
        delta = float(right) - float(left)
        deltas.append((delta, index))
        if delta > 1e-12:
            better += 1
        elif delta < -1e-12:
            worse += 1
        else:
            same += 1
    deltas.sort()
    return {"better": better, "worse": worse, "same": same,
            "worst_delta": deltas[0][0], "worst_index": deltas[0][1],
            "best_delta": deltas[-1][0], "best_index": deltas[-1][1]}


# --------------------------------------------------------------------- 主流程
def evaluate(checkpoint: Path, spec_path: Path, output_dir: Path, *,
             threshold: float = THRESHOLD, batch_size: int = 1,
             max_images: int | None = None) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    spec_path = Path(spec_path)
    output_dir = Path(output_dir)
    checkpoint_hash = overall.sha256_file(checkpoint)
    spec_hash = overall.sha256_file(spec_path)
    fp32, config = quant._load_fp32(checkpoint)
    int8, spec = quant._load_quantized(spec_path)
    if spec["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("INT8 spec 与 FP32 checkpoint 不一致")
    if int(config["size"]) != int(spec["size"]):
        raise ValueError("FP32 / INT8 输入尺寸不一致")
    assembly = overall.assemble(overall.build_default_spec())
    if len(assembly.train_items) != 2309:
        raise ValueError("冻结训练集数量变化")
    dev_total = sum(len(x.items) for x in assembly.dev_sets.values())
    if dev_total != 638:
        raise ValueError("冻结开发集数量变化")

    device = torch.device("cpu")
    int8 = int8.to(device).eval()

    records: list[dict[str, Any]] = []
    timings: dict[str, list[float]] = {name: [] for name in CANDIDATES}
    total_start = time.perf_counter()
    stop = False
    for name, dev in assembly.dev_sets.items():
        dataset = ISICDataset(dev.items, size=int(config["size"]), augment=False)
        loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False,
                            num_workers=0, collate_fn=isic_collate, drop_last=False)
        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].to(device)
                masks = batch["mask"]
                if masks is None:
                    raise overall.AssemblyError("开发集缺少掩膜")
                masks = masks.to(device)
                valids = batch["valid"].to(device)
                probability = torch.sigmoid(int8(images))
                keep = (valids > 0).to(torch.bool)
                truth_all = ((masks > 0) & keep).numpy()
                pred_all = ((probability >= float(threshold)) & keep).numpy()
                keep_np = keep.numpy()
                for row, filename in enumerate(batch["filename"]):
                    stem = Path(str(filename)).stem
                    truth = truth_all[row, 0]
                    keep_row = keep_np[row, 0]
                    base = pred_all[row, 0] & keep_row
                    entry = {"source": name, "stem": stem,
                             "area_ratio": float(dev.areas[stem]),
                             "band": overall.band_of(float(dev.areas[stem]), dev.cuts),
                             "candidates": {}}
                    for candidate in CANDIDATES:
                        started = time.perf_counter()
                        processed = apply_candidate(candidate, base, keep_row)
                        timings[candidate].append(time.perf_counter() - started)
                        if processed.shape != keep_row.shape:
                            raise ValueError("后处理输出形状异常")
                        if bool((processed & ~keep_row).any()):
                            raise ValueError("后处理越过 valid 区域")
                        dice, iou, empty, pred_sum = _per_image_metrics(processed, truth)
                        entry["candidates"][candidate] = {
                            "dice": dice, "iou": iou, "empty_prediction": empty,
                            "pred_pixels": pred_sum}
                    entry["gt_pixels"] = int(truth.sum())
                    records.append(entry)
                    if max_images is not None and len(records) >= int(max_images):
                        stop = True
                        break
                if stop:
                    break
        if stop:
            break
    total_seconds = time.perf_counter() - total_start
    if max_images is None and len(records) != dev_total:
        raise overall.AssemblyError(f"逐图记录 {len(records)} != 期望 {dev_total}")

    if max_images is not None:
        # 冒烟模式：只验证推理/后处理链路，不做跨来源汇总与 identity 复现校验
        smoke = {"schema": "v4-int8-postprocess-a11-smoke-v1",
                 "development_images": len(records), "threshold": float(threshold),
                 "records": records,
                 "timings_seconds": {name: float(sum(values))
                                     for name, values in timings.items()},
                 "note": "只跑前 max_images 张；不用于选模，也不做 identity 复现校验"}
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "smoke.json", smoke)
        print(json.dumps({"smoke": True, "images": len(records),
                          "timings_seconds": smoke["timings_seconds"]}, ensure_ascii=False),
              flush=True)
        return smoke

    reports: dict[str, Any] = {}
    for candidate in CANDIDATES:
        reports[candidate] = _summarise(_candidate_rows(candidate, records), assembly.dev_sets,
                                        threshold)
    identity_dice = float(reports["identity"]["primary"]["dice"])
    match = abs(identity_dice - EXPECTED_IDENTITY_DICE) <= IDENTITY_TOLERANCE
    if max_images is None and not match:
        raise overall.AssemblyError(
            f"identity 未复现既有 INT8 分数: {identity_dice!r} != {EXPECTED_IDENTITY_DICE!r}")

    identity_values = [row["candidates"]["identity"]["dice"] for row in records]
    comparisons = {}
    for candidate in CANDIDATES:
        values = [row["candidates"][candidate]["dice"] for row in records]
        comparison = _compare(identity_values, values)
        comparison["worst_stem"] = (records[comparison["worst_index"]]["stem"]
                                    if comparison["worse"] else None)
        comparison["worst_source"] = (records[comparison["worst_index"]]["source"]
                                      if comparison["worse"] else None)
        comparison["best_stem"] = records[comparison["best_index"]]["stem"]
        comparisons[candidate] = comparison

    ordered = sorted(CANDIDATES, key=lambda name: (-round(
        float(reports[name]["primary"]["dice"] or 0.0), 12), TIE_BREAK_ORDER.index(name)))
    summary = {
        "schema": "v4-int8-postprocess-a11-v1",
        "identity": {"checkpoint_sha256": checkpoint_hash, "spec_sha256": spec_hash,
                     "development_ids_sha256": assembly.split_digest["development_all"],
                     "developer_sha256": overall.sha256_file(Path(__file__)),
                     "threshold": float(threshold)},
        "development_images": len(records),
        "identity_reproduction": {"observed": identity_dice,
                                  "expected": EXPECTED_IDENTITY_DICE,
                                  "tolerance": IDENTITY_TOLERANCE, "match": bool(match)},
        "candidates": {name: {
            "primary_dice": float(reports[name]["primary"]["dice"]),
            "primary_iou": float(reports[name]["primary"]["iou"]),
            "source_dice": {k: float(v["all"]["dice"])
                            for k, v in reports[name]["sources"].items()},
            "area_groups": {k: {"n": int(v["n"]), "dice": v["dice"], "iou": v["iou"]}
                            for k, v in reports[name]["area_groups_pooled"].items()},
            "delta_vs_identity": float(reports[name]["primary"]["dice"]) - identity_dice,
            "per_image": comparisons[name],
            "timing_seconds": {
                "total": float(sum(timings[name])),
                "mean_ms": float(1000.0 * sum(timings[name]) / max(1, len(timings[name]))),
                "max_ms": float(1000.0 * max(timings[name]) if timings[name] else 0.0),
            },
        } for name in CANDIDATES},
        "ranking": ordered,
        "selected": ordered[0],
        "selection_rule": ("同规则应用于全部 638 图；以逐图平均 Dice 最大为主，"
                           "完全相同时按 identity / fill_holes / largest_component / combined 优先"),
        "complexity": {
            "fill_holes": "4 连通背景显式栈 DFS 单次遍历；时间 O(H*W)，空间 O(H*W)",
            "largest_component": "8 连通前景显式栈 DFS 单次遍历 + 标签分配；时间 O(H*W)，空间 O(H*W)",
            "temporary_memory_bytes_256x256": (258 * 258) * (1 + 1) + 256 * 256 * 4,
            "note": "仅增加 ARM/PS 端掩膜后处理，未改变 CNN 网络算子；未做 PYNQ-Z2 时序测量",
        },
        "holdout": "ISIC 2018 Task 1 masks have not been evaluated",
        "total_wall_seconds": float(total_seconds),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "per_image.json", {"schema": "v4-int8-postprocess-a11-per-image-v1",
                                                "threshold": float(threshold),
                                                "records": records})
    return summary


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("runs/role_a_overall_v4_dice2_base8/best.pt"))
    parser.add_argument("--spec", type=Path,
                        default=Path("runs/role_a_overall_v4_dice2_base8/int8_overall96/model_spec.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/role_a_v4_postprocess_a11"))
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    summary = evaluate(args.checkpoint, args.spec, args.output_dir,
                       threshold=args.threshold, batch_size=args.batch_size,
                       max_images=args.max_images)
    if args.max_images is not None:
        return 0
    for name in CANDIDATES:
        block = summary["candidates"][name]
        print(f"{name:<18} dice={block['primary_dice']:.10f} "
              f"iou={block['primary_iou']:.10f} delta={block['delta_vs_identity']:+.10f} "
              f"mean_ms={block['timing_seconds']['mean_ms']:.3f}", flush=True)
    print(f"selected={summary['selected']} identity_match="
          f"{summary['identity_reproduction']['match']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())