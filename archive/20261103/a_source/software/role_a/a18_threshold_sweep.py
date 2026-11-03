"""A18：soup_v5_a050 INT8 输出阈值的预设网格扩展（固定模型/导出/校准，无训练无下载）。

只做：一次 INT8 前向 -> 11 个预设阈值并行统计；先精确复现旧 5 阈值，再给候选部署阈值。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from software.role_a import quant
from software.role_a import train_overall_v1 as overall
from software.role_a.data import ISICDataset, isic_collate

OUT = Path("runs/role_a_a18_threshold_sweep")
CHECKPOINT = Path("runs/role_a_a14_soup/candidates/soup_v5_a050.pt")
SPEC = Path("runs/role_a_a14_soup/int8_soup_v5_a050/model_spec.json")
GRID: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.53, 0.56, 0.60)
OLD_GRID: tuple[float, ...] = (0.45, 0.50, 0.53, 0.56, 0.60)
REFERENCE_045 = 0.8420508068333175
TOL = 1e-9
JACCARD_CUT = 0.65


def binary_metrics(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """与 metrics.py 一致的边界约定：都空 => 1.0；只有一方空 => 0.0。"""
    p = int(pred.sum()); t = int(truth.sum()); inter = int((pred & truth).sum())
    dice, iou = overall._binary_dice_iou(p, t, inter)
    return float(dice), float(iou)


def per_image_for_grid(probability: np.ndarray, truth: np.ndarray, valid: np.ndarray,
                       grid: Sequence[float]) -> dict[float, tuple[float, float]]:
    """同一概率图对多个阈值并行统计（统一阈值，逐图）。"""
    if probability.shape != truth.shape or probability.shape != valid.shape:
        raise ValueError(f"shape 不一致: {probability.shape} {truth.shape} {valid.shape}")
    out: dict[float, tuple[float, float]] = {}
    for t in grid:
        pred = (probability >= float(t)) & valid          # 无效区恒为背景
        out[float(t)] = binary_metrics(pred, truth & valid)
    return out


def band_of(area: float, cuts: tuple[float, float]) -> str:
    return overall.band_of(area, cuts)


def evaluate(grid: Sequence[float] = GRID, *, max_images: int | None = None) -> dict[str, Any]:
    ck_sha = overall.sha256_file(CHECKPOINT)
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["checkpoint_sha256"] != ck_sha:
        raise SystemExit("spec 与候选 checkpoint 不一致，停止")
    if tuple(spec["layers"]) is None or len(spec["layers"]) != 15:
        raise SystemExit("spec 层数不等于 15，停止")
    model, _spec = quant._load_quantized(SPEC)
    model = model.to(torch.device("cpu")).eval()
    assembly = overall.assemble(overall.build_default_spec())
    dev_total = sum(len(d.items) for d in assembly.dev_sets.values())
    if max_images is None and dev_total != 638:
        raise SystemExit(f"冻结开发集数量异常: {dev_total}")

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for name, dev in assembly.dev_sets.items():
            ds = ISICDataset(dev.items, size=int(spec["size"]), augment=False)
            loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                                collate_fn=isic_collate, drop_last=False)
            for batch in loader:
                stem = Path(str(batch["filename"][0])).stem
                valid = (batch["valid"][0, 0] > 0).numpy()
                truth = ((batch["mask"][0, 0] > 0).numpy()) & valid
                prob = torch.sigmoid(model(batch["image"]))[0, 0].numpy()
                res = per_image_for_grid(prob, truth, valid, grid)
                records.append({"source": name, "stem": stem,
                                "area_ratio": float(dev.areas[stem]),
                                "band": band_of(float(dev.areas[stem]), dev.cuts),
                                "dice": {str(t): res[float(t)][0] for t in grid},
                                "iou": {str(t): res[float(t)][1] for t in grid},
                                "gt_pixels": int(truth.sum())})
                if max_images is not None and len(records) >= max_images:
                    break
            if max_images is not None and len(records) >= max_images:
                break
    wall = time.perf_counter() - started
    if max_images is not None:
        return {"smoke": True, "images": len(records), "sample": records[:2], "wall": wall}

    if len(records) != dev_total:
        raise SystemExit(f"逐图记录 {len(records)} != {dev_total}")

    def agg(subset: list[dict[str, Any]], t: float) -> dict[str, Any]:
        d = [r["dice"][str(t)] for r in subset]
        i = [r["iou"][str(t)] for r in subset]
        jac = [0.0 if r["iou"][str(t)] < JACCARD_CUT else r["iou"][str(t)] for r in subset]
        return {"n": len(subset), "dice": float(np.mean(d)) if d else None,
                "iou": float(np.mean(i)) if i else None,
                "jaccard_official": float(np.mean(jac)) if jac else None}

    base = [r["dice"]["0.45"] for r in records]
    table = {}
    for t in grid:
        vals = [r["dice"][str(t)] for r in records]
        deltas = [v - b for v, b in zip(vals, base)]
        better = sum(1 for d in deltas if d > 1e-12)
        worse = sum(1 for d in deltas if d < -1e-12)
        sources = {}
        for src in sorted({r["source"] for r in records}):
            sources[src] = agg([r for r in records if r["source"] == src], t)
        areas = {}
        for band in ("tiny", "medium", "large"):
            areas[band] = agg([r for r in records if r["band"] == band], t)
        table[str(t)] = {"all": agg(records, t), "sources": sources, "area_groups": areas,
                         "vs_0.45": {"better": better, "worse": worse, "same": len(records)-better-worse,
                                     "mean_delta": float(np.mean(deltas)),
                                     "worst_delta": float(min(deltas)),
                                     "worst_stems": [records[i]["stem"] for i in
                                                     sorted(range(len(deltas)), key=lambda k: deltas[k])[:5]]}}
    best = max(grid, key=lambda t: table[str(t)]["all"]["dice"])
    summary = {"schema": "a18-threshold-sweep-v1",
               "checkpoint": CHECKPOINT.as_posix(), "checkpoint_sha256": ck_sha,
               "spec_sha256": overall.sha256_file(SPEC),
               "calibration_images": spec["calibration_images"], "layers": len(spec["layers"]),
               "grid": list(grid), "old_grid": list(OLD_GRID),
               "development_images": len(records),
               "developer_sha256": overall.sha256_file(Path(__file__)),
               "reference_0.45": REFERENCE_045,
               "observed_0.45": table["0.45"]["all"]["dice"],
               "reproduction_ok": abs(table["0.45"]["all"]["dice"] - REFERENCE_045) <= TOL,
               "table": table, "best_threshold": float(best),
               "best_on_lower_boundary": bool(abs(best - min(grid)) < 1e-12),
               "wall_seconds": wall,
               "holdout": "ISIC 2018 Task 1 masks have not been evaluated"}
    if not summary["reproduction_ok"]:
        raise SystemExit(f"0.45 未复现 {REFERENCE_045}: {table['0.45']['all']['dice']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (OUT/"per_image.json").write_text(json.dumps({"schema": "a18-per-image-v1", "grid": list(grid),
                                                  "records": records}, ensure_ascii=False, indent=2)+"\n",
                                      encoding="utf-8")
    for t in grid:
        blk = table[str(t)]["all"]
        print(f"thr={t:.2f} dice={blk['dice']:.10f} iou={blk['iou']:.10f} "
              f"jac={blk['jaccard_official']:.10f} vs0.45 better/worse="
              f"{table[str(t)]['vs_0.45']['better']}/{table[str(t)]['vs_0.45']['worse']}", flush=True)
    print(f"best={best} on_lower_boundary={summary['best_on_lower_boundary']} "
          f"reproduction_ok={summary['reproduction_ok']}", flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args(list(argv) if argv is not None else None)
    if args.smoke:
        out = evaluate(max_images=args.smoke)
        print(json.dumps({"smoke": True, "images": out["images"], "wall": out["wall"]}, ensure_ascii=False))
        return 0
    evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())