"""A15-P: 对 soup_v5_a050 的 INT8 预测复用 A11 四规则（只读 a11 原语，不改 a11）。

先复现原样 INT8 Dice，再报四规则的同口径指标、逐图增减与最差样本对照页。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader

from software.role_a import postprocess_int8 as pp
from software.role_a import quant
from software.role_a import train_overall_v1 as overall
from software.role_a.data import ISICDataset, isic_collate

CHECKPOINT = Path("runs/role_a_a14_soup/candidates/soup_v5_a050.pt")
SPEC = Path("runs/role_a_a14_soup/int8_soup_v5_a050/model_spec.json")
OUT = Path("runs/role_a_a15_postprocess_soup")
THRESHOLD = 0.45
EXPECTED_IDENTITY = 0.8420508068
TOLERANCE = 1e-9
PANEL = 256
PER_SHEET = 3


def _error_map(pred, truth, valid):
    canvas = np.zeros((*pred.shape, 3), dtype=np.uint8)
    canvas[~valid] = (48, 48, 48)
    canvas[valid & pred & truth] = (255, 255, 255)
    canvas[valid & pred & ~truth] = (220, 40, 40)
    canvas[valid & ~pred & truth] = (60, 120, 255)
    return canvas


def _fit(canvas):
    image = Image.fromarray(canvas)
    return image if image.size == (PANEL, PANEL) else image.resize((PANEL, PANEL),
                                                                   Image.Resampling.BILINEAR)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--spec", type=Path, default=SPEC)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    checkpoint_hash = overall.sha256_file(args.checkpoint)
    int8, spec = quant._load_quantized(args.spec)
    if spec["checkpoint_sha256"] != checkpoint_hash:
        raise SystemExit("INT8 spec 与候选 checkpoint 不一致")
    int8 = int8.to(torch.device("cpu")).eval()
    assembly = overall.assemble(overall.build_default_spec())
    if sum(len(x.items) for x in assembly.dev_sets.values()) != 638:
        raise SystemExit("冻结开发集数量变化")

    records: list[dict[str, Any]] = []
    timings = {name: [] for name in pp.CANDIDATES}
    started_all = time.perf_counter()
    with torch.inference_mode():
        for name, dev in assembly.dev_sets.items():
            dataset = ISICDataset(dev.items, size=int(spec["size"]), augment=False)
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                collate_fn=isic_collate, drop_last=False)
            for batch in loader:
                stem = Path(str(batch["filename"][0])).stem
                valid = (batch["valid"][0, 0] > 0).numpy()
                truth = ((batch["mask"][0, 0] > 0).numpy()) & valid
                probability = torch.sigmoid(int8(batch["image"]))
                base = ((probability[0, 0] >= THRESHOLD).numpy()) & valid
                entry = {"source": name, "stem": stem,
                         "area_ratio": float(dev.areas[stem]),
                         "band": overall.band_of(float(dev.areas[stem]), dev.cuts),
                         "candidates": {}}
                for candidate in pp.CANDIDATES:
                    t0 = time.perf_counter()
                    processed = pp.apply_candidate(candidate, base, valid)
                    timings[candidate].append(time.perf_counter() - t0)
                    if bool((processed & ~valid).any()):
                        raise SystemExit("后处理越过 valid 区域")
                    dice, iou, empty, pred_px = pp._per_image_metrics(processed, truth)
                    entry["candidates"][candidate] = {"dice": dice, "iou": iou,
                                                      "empty_prediction": empty,
                                                      "pred_pixels": pred_px}
                entry["gt_pixels"] = int(truth.sum())
                image_rgb = (batch["image"][0].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                entry["_image"] = image_rgb
                entry["_truth"] = truth
                entry["_valid"] = valid
                entry["_base"] = base
                records.append(entry)
    wall = time.perf_counter() - started_all
    if len(records) != 638:
        raise SystemExit(f"逐图记录 {len(records)} != 638")

    def rows_for(candidate: str) -> list[dict[str, Any]]:
        return [{"source": r["source"], "stem": r["stem"], "area_ratio": r["area_ratio"],
                 "band": r["band"], "dice": [r["candidates"][candidate]["dice"]],
                 "iou": [r["candidates"][candidate]["iou"]],
                 "empty_prediction": [r["candidates"][candidate]["empty_prediction"]]}
                for r in records]

    reports = {c: overall.build_development_report(rows_for(c), assembly.dev_sets,
                                                   (THRESHOLD,)) for c in pp.CANDIDATES}
    identity_dice = float(reports["identity"]["primary"]["dice"])
    match = abs(identity_dice - EXPECTED_IDENTITY) <= TOLERANCE
    if not match:
        raise SystemExit(f"identity 未复现 {EXPECTED_IDENTITY}: {identity_dice!r}")

    identity_values = [r["candidates"]["identity"]["dice"] for r in records]
    summary: dict[str, Any] = {
        "schema": "a15-postprocess-soup-v1",
        "identity": {"checkpoint": args.checkpoint.as_posix(),
                     "checkpoint_sha256": checkpoint_hash,
                     "spec_sha256": overall.sha256_file(args.spec),
                     "development_ids_sha256": assembly.split_digest["development_all"],
                     "threshold": THRESHOLD,
                     "developer_sha256": overall.sha256_file(Path(__file__))},
        "identity_reproduction": {"observed": identity_dice, "expected": EXPECTED_IDENTITY,
                                  "tolerance": TOLERANCE, "match": True},
        "candidates": {},
        "holdout": "ISIC 2018 Task 1 masks have not been evaluated",
        "total_wall_seconds": wall,
    }
    severe: list[dict[str, Any]] = []
    for candidate in pp.CANDIDATES:
        report = reports[candidate]
        values = [r["candidates"][candidate]["dice"] for r in records]
        deltas = sorted((float(v) - float(b), i) for i, (v, b) in enumerate(zip(values,
                                                                               identity_values)))
        better = sum(1 for d, _ in deltas if d > 1e-12)
        worse = sum(1 for d, _ in deltas if d < -1e-12)
        summary["candidates"][candidate] = {
            "primary_dice": float(report["primary"]["dice"]),
            "primary_iou": float(report["primary"]["iou"]),
            "delta_vs_identity": float(report["primary"]["dice"]) - identity_dice,
            "source_dice": {k: float(v["all"]["dice"]) for k, v in report["sources"].items()},
            "area_groups": {k: {"n": int(v["n"]), "dice": v["dice"], "iou": v["iou"]}
                            for k, v in report["area_groups_pooled"].items()},
            "better": better, "worse": worse, "same": 638 - better - worse,
            "worst": [{"stem": records[i]["stem"], "source": records[i]["source"],
                       "delta": d} for d, i in deltas[:10]],
            "timing_seconds": {"mean_ms": 1000.0 * float(np.mean(timings[candidate])),
                               "max_ms": 1000.0 * float(np.max(timings[candidate]))},
        }
        for d, i in deltas:
            if d < -0.1:
                severe.append({"candidate": candidate, "stem": records[i]["stem"], "delta": d})
    ranked = sorted(pp.CANDIDATES, key=lambda c: -summary["candidates"][c]["primary_dice"])
    summary["ranking"] = ranked
    summary["best_by_overall_dice"] = ranked[0]
    summary["severe_regressions_over_0.1"] = severe

    sheets = []
    if severe:
        seen = []
        for item in sorted(severe, key=lambda x: x["delta"]):
            if item["stem"] in seen:
                continue
            seen.append(item["stem"])
        panels = [r for r in records if r["stem"] in seen][:9]
        for offset in range(0, len(panels), PER_SHEET):
            chunk = panels[offset:offset + PER_SHEET]
            canvas = Image.new("RGB", (PANEL * 4 + 50, len(chunk) * (PANEL + 40) + 10), "white")
            draw = ImageDraw.Draw(canvas)
            for row_index, r in enumerate(chunk):
                top = 10 + row_index * (PANEL + 40)
                combined = pp.apply_candidate("combined", r["_base"], r["_valid"])
                largest = pp.apply_candidate("largest_component", r["_base"], r["_valid"])
                titles = ("原图", "GT", "原预测(TP白/FP红/FN蓝)", "combined(TP白/FP红/FN蓝)")
                images = (_fit(r["_image"]),
                          _fit(np.dstack([(r["_truth"] & r["_valid"]).astype(np.uint8) * 255] * 3)),
                          _fit(_error_map(r["_base"], r["_truth"], r["_valid"])),
                          _fit(_error_map(combined, r["_truth"], r["_valid"])))
                draw.text((6, top),
                          f"{r['stem']} {r['source']} identity="
                          f"{r['candidates']['identity']['dice']:.4f} "
                          f"largest={r['candidates']['largest_component']['dice']:.4f} "
                          f"combined={r['candidates']['combined']['dice']:.4f}",
                          fill="black")
                for column, image in enumerate(images):
                    x = 6 + column * (PANEL + 11)
                    canvas.paste(image, (x, top + 20))
                    draw.text((x, top + 4 + PANEL + 22), titles[column], fill="black")
            path = out / f"a15_severe_sheet_{offset // PER_SHEET + 1}.jpg"
            canvas.save(path, quality=92)
            sheets.append(path.name)
    summary["severe_sheets"] = sheets

    for r in records:
        for key in ("_image", "_truth", "_valid", "_base"):
            r.pop(key, None)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    (out / "per_image.json").write_text(
        json.dumps({"schema": "a15-postprocess-soup-per-image-v1", "threshold": THRESHOLD,
                    "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for candidate in ranked:
        block = summary["candidates"][candidate]
        print(f"{candidate:<18} dice={block['primary_dice']:.10f} "
              f"iou={block['primary_iou']:.10f} delta={block['delta_vs_identity']:+.10f} "
              f"better={block['better']} worse={block['worse']}", flush=True)
    print(f"best={ranked[0]} severe={len(severe)} sheets={len(sheets)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())