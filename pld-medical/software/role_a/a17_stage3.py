"""A17 阶段 3：单次 INT8 前向计算 identity / A15 combined / 5-10-20% 多分量保留。"""
from __future__ import annotations
import json, os, time
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import torch
from torch.utils.data import DataLoader

from software.role_a import a17_component_ratio as a17
from software.role_a import postprocess_int8 as pp
from software.role_a import quant
from software.role_a import train_overall_v1 as overall
from software.role_a.data import ISICDataset, isic_collate

OUT = Path("runs/role_a_a17_component_ratio")
CHECKPOINT = Path("runs/role_a_a14_soup/candidates/soup_v5_a050.pt")
SPEC = Path("runs/role_a_a14_soup/int8_soup_v5_a050/model_spec.json")
A15 = Path("runs/role_a_a15_postprocess_soup/summary.json")
THRESHOLD = 0.45
REF_IDENTITY = 0.8420508068333175
JACCARD_CUT = 0.65
RULES = ("identity", "combined", "keep_ratio_5", "keep_ratio_10", "keep_ratio_20")


def atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def block(rule: str, subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
    dice = [r[f"{rule}_dice"] for r in subset]
    iou = [r[f"{rule}_iou"] for r in subset]
    jac = [0.0 if r[f"{rule}_iou"] < JACCARD_CUT else r[f"{rule}_iou"] for r in subset]
    return {"n": len(subset), "dice": float(np.mean(dice)) if dice else None,
            "iou": float(np.mean(iou)) if iou else None,
            "jaccard_official": float(np.mean(jac)) if jac else None}


def aggregate(records: Sequence[dict[str, Any]], old13_stems: Sequence[str]) -> dict[str, Any]:
    """纯聚合：给定逐图记录与旧严重退步样本清单，产出全部报告字段（可单测）。"""
    old13 = set(old13_stems)
    base = [r["identity_dice"] for r in records]
    table: dict[str, Any] = {}
    for rule in RULES:
        deltas = [r[f"{rule}_dice"] - b for r, b in zip(records, base)]
        better = sum(1 for d in deltas if d > 1e-12)
        worse = sum(1 for d in deltas if d < -1e-12)
        severe = [d for d in deltas if d < -0.1]
        worst_idx = int(np.argmin(deltas)) if deltas else 0
        subset13 = [r for r in records if r["stem"] in old13]
        table[rule] = {
            "all": block(rule, records),
            "area_groups": {b: block(rule, [r for r in records if r["band"] == b])
                            for b in ("tiny", "medium", "large")},
            "sources": {s: block(rule, [r for r in records if r["source"] == s])
                        for s in sorted({r["source"] for r in records})},
            "vs_identity": {"better": better, "worse": worse, "same": len(deltas) - better - worse,
                            "severe_gt_0.1": len(severe),
                            "worst_delta": float(min(deltas)) if deltas else None,
                            "worst_stem": records[worst_idx]["stem"],
                            "severe_stems": [records[i]["stem"] for i, d in enumerate(deltas) if d < -0.1]},
            "old13": {"n": len(subset13),
                      "mean_dice": float(np.mean([r[f"{rule}_dice"] for r in subset13])) if subset13 else None,
                      "mean_identity_dice": float(np.mean([r["identity_dice"] for r in subset13])) if subset13 else None,
                      "min_delta": float(min(r[f"{rule}_dice"] - r["identity_dice"] for r in subset13)) if subset13 else None},
        }
    improvement = {rule: {"mean_delta_vs_identity": (table[rule]["old13"]["mean_dice"]
                                                   - table[rule]["old13"]["mean_identity_dice"])
                          if table[rule]["old13"]["n"] else None,
                          "min_delta": table[rule]["old13"]["min_delta"],
                          "severe_remaining": sum(1 for r in records if r["stem"] in old13
                                                  and (r[f"{rule}_dice"] - r["identity_dice"]) < -0.1)}
                   for rule in RULES}
    return {"table": table, "old13_improvement": improvement}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ck = overall.sha256_file(CHECKPOINT)
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["checkpoint_sha256"] != ck or len(spec["layers"]) != 15:
        raise SystemExit("checkpoint/spec 校验失败，停止")
    a15 = json.loads(A15.read_text(encoding="utf-8"))
    ref_combined = float(a15["candidates"]["combined"]["primary_dice"])
    old13 = sorted({x["stem"] for x in a15["severe_regressions_over_0.1"]})

    model, _ = quant._load_quantized(SPEC)
    model = model.to(torch.device("cpu")).eval()
    assembly = overall.assemble(overall.build_default_spec())
    expected = sum(len(d.items) for d in assembly.dev_sets.values())
    if expected != 638:
        raise SystemExit(f"开发集数量异常 {expected}")

    records: list[dict[str, Any]] = []
    timings = {r: [] for r in RULES}
    t_all = time.perf_counter()
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
                base = (prob >= THRESHOLD) & valid
                t0 = time.perf_counter(); ident = base; timings["identity"].append(time.perf_counter()-t0)
                t0 = time.perf_counter(); comb = pp.apply_candidate("combined", base, valid)
                timings["combined"].append(time.perf_counter()-t0)
                stats: dict = {}
                t0 = time.perf_counter()
                ratios = a17.apply_all_ratios(base, valid, stats=stats)   # 共享一次连通域标记
                dt = time.perf_counter()-t0
                for k in ("keep_ratio_5", "keep_ratio_10", "keep_ratio_20"):
                    timings[k].append(dt / 3.0)
                outs = {"identity": ident, "combined": comb, "keep_ratio_5": ratios["keep_ratio_5"],
                        "keep_ratio_10": ratios["keep_ratio_10"], "keep_ratio_20": ratios["keep_ratio_20"]}
                rec = {"source": name, "stem": stem, "area_ratio": float(dev.areas[stem]),
                       "band": overall.band_of(float(dev.areas[stem]), dev.cuts), "gt_pixels": int(truth.sum()),
                       "components_total": int(stats["components_total"]),
                       "largest_component_px": int(stats["largest_component_px"]),
                       "kept_5": int(stats["kept"][5]), "kept_10": int(stats["kept"][10]),
                       "kept_20": int(stats["kept"][20])}
                for rule, pred in outs.items():
                    d, i = pp._per_image_metrics(pred, truth)[0:2]
                    rec[f"{rule}_dice"] = float(d); rec[f"{rule}_iou"] = float(i)
                records.append(rec)
    wall = time.perf_counter() - t_all
    if len(records) != expected:
        raise SystemExit(f"逐图记录 {len(records)} != {expected}")

    agg = aggregate(records, old13)
    table = agg["table"]
    obs_id = table["identity"]["all"]["dice"]
    obs_comb = table["combined"]["all"]["dice"]
    ok = abs(obs_id - REF_IDENTITY) <= 1e-9 and abs(obs_comb - ref_combined) <= 1e-9
    summary = {"schema": "a17-stage3-v1", "checkpoint_sha256": ck,
               "spec_sha256": overall.sha256_file(SPEC),
               "code_sha256": {"a17_component_ratio.py": overall.sha256_file(Path(a17.__file__)),
                               "a17_stage3.py": overall.sha256_file(Path(__file__)),
                               "postprocess_int8.py": overall.sha256_file(Path(pp.__file__))},
               "threshold": THRESHOLD, "development_images": len(records),
               "reproduction": {"identity_observed": obs_id, "identity_reference": REF_IDENTITY,
                                "combined_observed": obs_comb, "combined_reference": ref_combined,
                                "tolerance": 1e-9, "ok": bool(ok)},
               "table": table, "old13_improvement": agg["old13_improvement"],
               "old13_stems": old13,
               "timings_seconds": {r: {"mean_ms": 1000*float(np.mean(timings[r])),
                                       "max_ms": 1000*float(np.max(timings[r]))} for r in RULES},
               "wall_seconds": wall,
               "board_note": "桌面 x86 参考；PYNQ-Z2 PS 端耗时未实测",
               "holdout": "ISIC 2018 Task 1 masks have not been evaluated"}
    if not ok:
        atomic_write_json(OUT/"reproduction_failure.json", summary)
        raise SystemExit(f"复现失败 identity={obs_id} combined={obs_comb}")
    prev = OUT/"per_image_v3.json"
    consistency = None
    if prev.is_file():
        old_records = {r["stem"]: r for r in json.loads(prev.read_text(encoding="utf-8"))["records"]}
        worst = 0.0
        for r in records:
            o = old_records.get(r["stem"])
            if not o:
                continue
            for rule in RULES:
                worst = max(worst, abs(r[f"{rule}_dice"] - o[f"{rule}_dice"]),
                            abs(r[f"{rule}_iou"] - o[f"{rule}_iou"]))
        consistency = {"compared_stems": len(old_records), "max_abs_diff_vs_prev": worst,
                       "identical_within_1e-12": bool(worst <= 1e-12)}
    summary["consistency_vs_prev_run"] = consistency
    summary["counts"] = {"components_total_sum": int(sum(r["components_total"] for r in records)),
                         "kept_5_sum": int(sum(r["kept_5"] for r in records)),
                         "kept_10_sum": int(sum(r["kept_10"] for r in records)),
                         "kept_20_sum": int(sum(r["kept_20"] for r in records)),
                         "images_with_multi_components": int(sum(1 for r in records if r["components_total"] > 1))}
    atomic_write_json(OUT/"summary_v3_counts.json", summary)
    atomic_write_json(OUT/"per_image_v3_counts.json", {"schema": "a17-per-image-v3-counts",
                                                       "rules": list(RULES), "records": records})
    for rule in RULES:
        b = table[rule]
        print(f"{rule:<15} dice={b['all']['dice']:.10f} iou={b['all']['iou']:.10f} "
              f"jac={b['all']['jaccard_official']:.10f} severe={b['vs_identity']['severe_gt_0.1']} "
              f"worst={b['vs_identity']['worst_delta']:+.6f} "
              f"old13_delta={agg['old13_improvement'][rule]['mean_delta_vs_identity']:+.6f}", flush=True)
    print("counts sum: total=%d kept5=%d kept10=%d kept20=%d multi=%d" % (
        summary["counts"]["components_total_sum"], summary["counts"]["kept_5_sum"],
        summary["counts"]["kept_10_sum"], summary["counts"]["kept_20_sum"],
        summary["counts"]["images_with_multi_components"]), flush=True)
    print(f"consistency_vs_prev={consistency}", flush=True)
    print(f"reproduction ok={ok} identity={obs_id:.16f} combined={obs_comb:.16f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())