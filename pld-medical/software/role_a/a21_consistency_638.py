"""638 张逐图一致性：假量化 FP32 代理 vs 已验证的真实 INT8（soup_v5_a050）。

只读现有权重与导出产物，不算主指标（主指标来自既有 evaluate_quant_overall 证据）。
输出逐图统计：逐像素平均绝对差、逐图最大值、阈值化翻转像素比例。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from software.role_a import qat, quant
from software.role_a.train_overall_v1 import _binary_dice_iou


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    scales = {layer["name"]: {"input_scale": float(layer["input_scale"]),
                              "output_scale": float(layer["output_scale"])}
              for layer in spec["layers"]}
    fake, config = quant._load_fp32(args.candidate)
    qat.enable_weight_qat(fake)
    qat.set_scales_from_calibration(
        fake, {name: {"input_max_abs": pair["input_scale"] * qat.QMAX,
                      "output_max_abs": pair["output_scale"] * qat.QMAX}
               for name, pair in scales.items()})
    fake.eval()
    real, _ = quant._load_quantized(args.spec)
    real.eval()

    assembly = quant.load_overall_assembly()
    rows = []
    thresholds = (0.45, 0.50, 0.53, 0.56, 0.60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake = fake.to(device)
    dice_delta: list[float] = []
    signed_dice_delta: list[float] = []
    with torch.no_grad():
        for source, dev in sorted(assembly.dev_sets.items()):
            dataset = quant.ISICDataset(list(dev.items), root=None, size=int(config["size"]))
            for index in range(len(dev.items)):
                item = dataset[index]
                image = item.image.unsqueeze(0)
                mask = item.mask.squeeze(0).numpy()
                valid = item.valid.squeeze(0).numpy()
                fake_probability = torch.sigmoid(fake(image.to(device))).squeeze().cpu().numpy()
                real_probability = torch.sigmoid(real(image)).squeeze().numpy()
                difference = np.abs(fake_probability - real_probability)
                keep = valid > 0
                truth = (mask > 0) & keep
                fake_pred = (fake_probability >= 0.45) & keep
                real_pred = (real_probability >= 0.45) & keep
                fake_dice, _ = _binary_dice_iou(int(fake_pred.sum()), int(truth.sum()),
                                                int((fake_pred & truth).sum()))
                real_dice, _ = _binary_dice_iou(int(real_pred.sum()), int(truth.sum()),
                                                int((real_pred & truth).sum()))
                dice_delta.append(abs(fake_dice - real_dice))
                signed_dice_delta.append(fake_dice - real_dice)
                rows.append({
                    "source": source, "stem": dev.items[index].stem,
                    "image_max_abs_probability_diff": float(difference.max()),
                    # 注意口径：image_pixel_mean_* 是**整张模型输入图（256×256，含 letterbox 补边）**
                    # 的逐像素平均；valid 区域内的逐像素平均另记 *_valid_only。
                    "image_pixel_mean_abs_probability_diff": float(difference.mean()),
                    "image_pixel_mean_abs_probability_diff_valid_only":
                        float(difference[keep].mean()) if keep.any() else 0.0,
                    "flipped_pixels_at_0.45": int((fake_pred != real_pred).sum()),
                    "total_valid_pixels": int(keep.sum()),
                    "dice_delta_abs_at_0.45": abs(fake_dice - real_dice),
                    "dice_delta_signed_at_0.45": fake_dice - real_dice,
                })
            print(f"done {source}: {len(dev.items)} images", flush=True)

    image_maxima = np.array([row["image_max_abs_probability_diff"] for row in rows])
    pixel_means = np.array([row["image_pixel_mean_abs_probability_diff"] for row in rows])
    valid_pixel_means = np.array([row["image_pixel_mean_abs_probability_diff_valid_only"]
                                  for row in rows])
    dice_deltas = np.array(dice_delta)
    signed_deltas = np.array(signed_dice_delta)
    result = {
        "schema": "a21-consistency-638-v1",
        "candidate": args.candidate.as_posix(), "spec": args.spec.as_posix(),
        "images": len(rows),
        "sources": sorted({row["source"] for row in rows}),
        "per_image": rows,
        "metric_scopes": {
            "pixel_mean_abs_probability_diff":
                "整张模型输入 256×256（含 letterbox 补边）的逐像素平均绝对概率差，再对 638 张取平均",
            "pixel_mean_abs_probability_diff_valid_only":
                "同上但只在 valid（非补边）区域内统计",
            "dice_delta_signed_at_0.45":
                "逐图 signed Dice 差（fake − real），mean 即代理与真 INT8 的整体 Dice 偏向",
            "dice_delta_abs_at_0.45": "逐图 Dice 差的绝对值",
        },
        "pixel_mean_abs_probability_diff": float(pixel_means.mean()),
        "pixel_mean_abs_probability_diff_valid_only": float(valid_pixel_means.mean()),
        "mean_of_image_max_abs_probability_diff": float(image_maxima.mean()),
        "max_of_image_max_abs_probability_diff": float(image_maxima.max()),
        "p95_of_image_max_abs_probability_diff": float(np.quantile(image_maxima, 0.95)),
        "flipped_pixel_fraction_max": max(row["flipped_pixels_at_0.45"] /
                                          max(row["total_valid_pixels"], 1) for row in rows),
        "dice_delta_signed_at_0.45_mean": float(signed_deltas.mean()),
        "dice_delta_abs_mean_at_0.45": float(dice_deltas.mean()),
        "dice_delta_abs_max_at_0.45": float(dice_deltas.max()),
        "dice_delta_gt_0.001_count": int((dice_deltas > 0.001).sum()),
        "note": "638 张逐图代理一致性；主指标仍以真实 INT8 的 evaluate_quant_overall 为准",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    # 控制台可能是 GBK（Windows 默认），这里只打印 ASCII，避免 U+2212 等字符触发编码错误
    summary = {k: v for k, v in result.items()
               if k not in ("per_image", "sources", "metric_scopes")}
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
