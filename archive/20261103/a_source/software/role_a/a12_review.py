"""A12 阶段一：A11 combined 后处理严重退步样本的证据整理与伪影核算。

只读 638 张冻结开发集与 v4 INT8 模型；不读 2018 最终留出集，不训练、不下载。
输出：runs/role_a_a12_review/（对照页 + 诊断 JSON + 报告）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from software.role_a import postprocess_int8 as pp
from software.role_a import quant
from software.role_a import train_overall_v1 as overall
from software.role_a.data import ISICDataset, isic_collate

A11 = Path("runs/role_a_v4_postprocess_a11")
OUT = Path("runs/role_a_a12_review")
CHECKPOINT = Path("runs/role_a_overall_v4_dice2_base8/best.pt")
SPEC = Path("runs/role_a_overall_v4_dice2_base8/int8_overall96/model_spec.json")
THRESHOLD = 0.45
PANEL = 256
PER_SHEET = 3


def _error_map(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """TP=白 FP=红 FN=蓝 TN=黑 无效区=深灰。"""
    canvas = np.zeros((*pred.shape, 3), dtype=np.uint8)
    canvas[~valid] = (48, 48, 48)
    canvas[valid & ~pred & ~truth] = (0, 0, 0)
    canvas[valid & pred & truth] = (255, 255, 255)
    canvas[valid & pred & ~truth] = (220, 40, 40)
    canvas[valid & ~pred & truth] = (60, 120, 255)
    return canvas


def _binary(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[~valid] = (48, 48, 48)
    canvas[valid & mask] = (255, 255, 255)
    return canvas


def _fit(canvas: np.ndarray, size: int = PANEL) -> Image.Image:
    image = Image.fromarray(canvas)
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    return image


def _components(mask: np.ndarray) -> tuple[int, list[int]]:
    _labels, count, sizes, _pos = pp._label_components(mask, connectivity=8)
    return count, sorted(sizes, reverse=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a11 = json.loads((A11 / "per_image.json").read_text(encoding="utf-8"))["records"]
    deltas = {row["stem"]: row["candidates"]["combined"]["dice"]
              - row["candidates"]["identity"]["dice"] for row in a11}
    worst = sorted(deltas.items(), key=lambda kv: kv[1])[:9]
    worst_stems = {stem for stem, _ in worst}
    if len(worst) != 9 or any(delta >= -0.1 for _stem, delta in worst):
        raise SystemExit(f"未找到预期的 9 张退步样本: {worst}")

    _fp32, config = quant._load_fp32(CHECKPOINT)
    int8, spec = quant._load_quantized(SPEC)
    if spec["checkpoint_sha256"] != overall.sha256_file(CHECKPOINT):
        raise SystemExit("INT8 spec 与 checkpoint 不一致")
    int8 = int8.to(torch.device("cpu")).eval()
    assembly = overall.assemble(overall.build_default_spec())
    if sum(len(x.items) for x in assembly.dev_sets.values()) != 638:
        raise SystemExit("冻结开发集数量变化")

    accounting = {"images": 0,
                  "fill_added_px": 0, "fill_added_tp": 0,
                  "largest_removed_px": 0, "largest_removed_tp": 0,
                  "combined_added_px": 0, "combined_added_tp": 0,
                  "combined_removed_px": 0, "combined_removed_tp": 0,
                  "gt_multi_component_images": 0,
                  "dice_delta_sum": 0.0,
                  "kept_component_gt_main_coverage_sum": 0.0,
                  "all_components_gt_main_coverage_sum": 0.0,
                  "kept_component_covers_under_half_of_gt_main": 0,
                  "invalidated_fields": [
                      "base_first_component_misses_gt_largest: 判据无效（以最大预测分量面积等于真值面积为前置，"
                      "会把一般不完美预测大量计为漏掉最大真值），已删除且不用于任何裁定",
                  ]}
    diagnostics: list[dict[str, Any]] = []
    panels: list[dict[str, Any]] = []
    with torch.inference_mode():
        for name, dev in assembly.dev_sets.items():
            dataset = ISICDataset(dev.items, size=int(config["size"]), augment=False)
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                collate_fn=isic_collate, drop_last=False)
            for batch in loader:
                stem = Path(str(batch["filename"][0])).stem
                images = batch["image"]
                valid = (batch["valid"][0, 0] > 0).numpy()
                truth = ((batch["mask"][0, 0] > 0).numpy()) & valid
                probability = torch.sigmoid(int8(images))
                base = ((probability[0, 0] >= THRESHOLD).numpy()) & valid
                filled = pp.fill_holes(base, valid)
                largest = pp.largest_component(base, valid)
                combined = pp.fill_holes(largest, valid)

                accounting["images"] += 1
                add_f = filled & ~base
                rem_l = base & ~largest
                add_c = combined & ~base
                rem_c = base & ~combined
                accounting["fill_added_px"] += int(add_f.sum())
                accounting["fill_added_tp"] += int((add_f & truth).sum())
                accounting["largest_removed_px"] += int(rem_l.sum())
                accounting["largest_removed_tp"] += int((rem_l & truth).sum())
                accounting["combined_added_px"] += int(add_c.sum())
                accounting["combined_added_tp"] += int((add_c & truth).sum())
                accounting["combined_removed_px"] += int(rem_c.sum())
                accounting["combined_removed_tp"] += int((rem_c & truth).sum())
                gt_count, gt_sizes = _components(truth)
                base_count, base_sizes = _components(base)
                comb_count, comb_sizes = _components(combined)
                if gt_count > 1:
                    accounting["gt_multi_component_images"] += 1
                # 修正后的良定义统计：保留分量对【最大真值连通块】的覆盖率
                dice_base, _iou_base, _e_base, _p_base = pp._per_image_metrics(base, truth)
                dice_comb, _iou_comb, _e_comb, _p_comb = pp._per_image_metrics(combined, truth)
                accounting["dice_delta_sum"] += dice_comb - dice_base
                labels_gt, _n_gt, sizes_gt, _p_gt = pp._label_components(truth, connectivity=8)
                if sizes_gt:
                    gt_main = labels_gt == (int(np.argmax(sizes_gt)) + 1)
                    gt_main_px = max(1, int(gt_main.sum()))
                    accounting["all_components_gt_main_coverage_sum"] += (
                        int((base & gt_main).sum()) / gt_main_px)
                    labels_pred, _n_pred, sizes_pred, _p_pred = pp._label_components(
                        base, connectivity=8)
                    if sizes_pred:
                        kept = labels_pred == (int(np.argmax(sizes_pred)) + 1)
                        coverage = int((kept & gt_main).sum()) / gt_main_px
                        accounting["kept_component_gt_main_coverage_sum"] += coverage
                        if coverage < 0.5:
                            accounting["kept_component_covers_under_half_of_gt_main"] += 1
                if stem in worst_stems:
                    image_rgb = (images[0].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                    panels.append({
                        "stem": stem, "source": name,
                        "image": image_rgb, "truth": truth, "valid": valid,
                        "base": base, "combined": combined,
                        "identity_dice": next(r["candidates"]["identity"]["dice"]
                                              for r in a11 if r["stem"] == stem),
                        "combined_dice": next(r["candidates"]["combined"]["dice"]
                                              for r in a11 if r["stem"] == stem),
                    })
                if stem in worst_stems or True:
                    diagnostics.append({
                        "stem": stem, "source": name, "band": overall.band_of(
                            float(dev.areas[stem]), dev.cuts),
                        "gt_pixels": int(truth.sum()), "gt_components": gt_count,
                        "gt_component_sizes": gt_sizes[:5],
                        "base_pixels": int(base.sum()), "base_components": base_count,
                        "base_component_sizes": base_sizes[:5],
                        "combined_components": comb_count,
                        "combined_component_sizes": comb_sizes[:5],
                    })
    (OUT / "component_diagnostics.json").write_text(
        json.dumps({"schema": "a12-component-diagnostics-v1",
                    "records": [d for d in diagnostics if d["stem"] in worst_stems]},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "artifact_accounting.json").write_text(
        json.dumps({"schema": "a12-artifact-accounting-v1", "threshold": THRESHOLD,
                    "checkpoint": str(CHECKPOINT), "spec": str(SPEC),
                    "counts": accounting}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    panels.sort(key=lambda row: row["combined_dice"] - row["identity_dice"])
    sheets = []
    for offset in range(0, len(panels), PER_SHEET):
        chunk = panels[offset:offset + PER_SHEET]
        canvas = Image.new("RGB", (PANEL * 4 + 50, len(chunk) * (PANEL + 40) + 10), "white")
        draw = ImageDraw.Draw(canvas)
        for row_index, row in enumerate(chunk):
            top = 10 + row_index * (PANEL + 40)
            titles = ("原图", "真值(GT)", "原样 INT8 (TP白/FP红/FN蓝)", "combined (TP白/FP红/FN蓝)")
            images = (_fit(row["image"]), _fit(_binary(row["truth"], row["valid"])),
                      _fit(_error_map(row["base"], row["truth"], row["valid"])),
                      _fit(_error_map(row["combined"], row["truth"], row["valid"])))
            draw.text((6, top),
                      f"{row['stem']}  {row['source']}  identity={row['identity_dice']:.4f} "
                      f"combined={row['combined_dice']:.4f} "
                      f"delta={row['combined_dice'] - row['identity_dice']:+.4f}",
                      fill="black")
            for column, image in enumerate(images):
                x = 6 + column * (PANEL + 11)
                canvas.paste(image, (x, top + 20))
                draw.text((x, top + 4 + PANEL + 22), titles[column], fill="black")
        path = OUT / f"a12_worst_sheet_{offset // PER_SHEET + 1}.jpg"
        canvas.save(path, quality=92)
        sheets.append(path.name)
        print(path, flush=True)

    summary = {
        "schema": "a12-review-summary-v1",
        "visual_review_by_executor": False,
        "visual_review_note": ("本执行助手没有图像查看能力（工具集只有文本通道），"
                               "未做视觉初审；对照页供 Codex 人工复核"),
        "worst_nine": [{"stem": stem, "delta": delta} for stem, delta in worst],
        "sheets": sheets,
        "artifact_accounting": accounting,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(accounting, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()