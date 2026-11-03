"""Lock one INT8 model and threshold, then evaluate the untouched ISIC 2018 set.

The ``lock`` command reads only selection reports and holdout provenance
metadata. The ``evaluate`` command is the sole entry point that opens holdout
images and masks. It never searches thresholds or changes the selected model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from software.role_a import quant, refine_small as base, train_overall_v1 as overall
from software.role_a.data import enumerate_isic_pairs
from software.role_a.refine_conservative import evaluate_grid


HOLDOUT = base.PROJECT / "data" / "raw" / "isic2018_test"
PROVENANCE = HOLDOUT / "provenance.json"
EXCLUSIONS = (base.PROJECT / "runs" / "role_a_from_baseline_v3" /
              "test2018_overlap_exclusions.json")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as stream:
        tmp = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(tmp, path)


def lock(checkpoint: Path, spec: Path, development: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Selection already locked: {output}")
    ck_hash = overall.sha256_file(checkpoint)
    sp_hash = overall.sha256_file(spec)
    spec_payload = json.loads(spec.read_text(encoding="utf-8"))
    if spec_payload["checkpoint_sha256"] != ck_hash:
        raise ValueError("INT8 export is not from the selected FP32 checkpoint")
    report = json.loads(development.read_text(encoding="utf-8"))
    if report.get("schema") != "overall-int8-development-v1":
        raise ValueError("Development comparison has an unexpected schema")
    if report["identity"]["checkpoint_sha256"] != ck_hash or \
            report["identity"]["spec_sha256"] != sp_hash:
        raise ValueError("Development comparison does not match selected model")
    assembly = overall.assemble(overall.build_default_spec())
    if (report["identity"].get("development_ids_sha256") !=
            assembly.split_digest["development_all"] or
            report["int8"]["primary"].get("n") != 638):
        raise ValueError("Development comparison is not from the frozen 638-image set")
    threshold = float(report["int8"]["selected_threshold"])
    if threshold not in overall.THRESHOLDS:
        raise ValueError("Threshold was not in the frozen development grid")
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    exclusions = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))
    if provenance.get("complete") is not True or len(provenance.get("items", {})) != 1000:
        raise ValueError("Official 2018 holdout provenance is incomplete")
    if exclusions.get("frozen_before_segmentation_evaluation") is not True or \
            len(set(exclusions.get("excluded_test_ids", []))) != 41 or \
            exclusions.get("clean_count_expected") != 959:
        raise ValueError("2018 image-only overlap exclusions differ from frozen protocol")
    atomic_json(output, {
        "schema": "role-a-2018-holdout-selection-lock-v1",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "frozen_before_holdout_score": True,
        "checkpoint": checkpoint.resolve().as_posix(), "checkpoint_sha256": ck_hash,
        "spec": spec.resolve().as_posix(), "spec_sha256": sp_hash,
        "development_report": development.resolve().as_posix(),
        "development_report_sha256": overall.sha256_file(development),
        "threshold": threshold,
        "holdout_provenance_sha256": overall.sha256_file(PROVENANCE),
        "holdout_exclusions_sha256": overall.sha256_file(EXCLUSIONS),
        "expected_full": 1000, "expected_clean": 959,
        "rule": "Fixed model and one global INT8 development threshold; no holdout tuning",
    })


def validate_cached_rows(rows: object, areas: dict[str, float],
                         area_cuts: tuple[float, float]) -> list[dict]:
    """Reject stale or damaged cached holdout records before summarizing them."""
    if not isinstance(rows, list) or len(rows) != len(areas):
        raise ValueError("Cached holdout records have wrong count")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("stem"), str):
            raise ValueError("Cached holdout record has no image ID")
        stem = row["stem"]
        if stem not in areas or stem in seen:
            raise ValueError(f"Cached holdout record has missing/duplicate ID: {stem}")
        seen.add(stem)
        area_ratio = float(row["area_ratio"])
        if not math.isfinite(area_ratio) or abs(area_ratio - areas[stem]) > 1e-12 or \
                row["band"] != base.band(areas[stem], area_cuts):
            raise ValueError(f"Cached holdout record has wrong area group: {stem}")
        for key in ("dice", "iou", "recall", "precision"):
            value = float(row[key])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Cached holdout record has invalid {key}: {stem}")
        if row["empty_prediction"] not in (False, True, 0, 1):
            raise ValueError(f"Cached holdout record has invalid empty_prediction: {stem}")
    if seen != set(areas):
        raise ValueError("Cached holdout records omit official image IDs")
    return rows


def evaluate(lock_path: Path, output_dir: Path) -> None:
    selection = json.loads(lock_path.read_text(encoding="utf-8"))
    if selection.get("schema") != "role-a-2018-holdout-selection-lock-v1" or \
            selection.get("frozen_before_holdout_score") is not True:
        raise ValueError("Selection lock is missing or invalid")
    checkpoint = Path(selection["checkpoint"])
    spec_path = Path(selection["spec"])
    for path, key in ((checkpoint, "checkpoint_sha256"),
                      (spec_path, "spec_sha256"),
                      (Path(selection["development_report"]), "development_report_sha256"),
                      (PROVENANCE, "holdout_provenance_sha256"),
                      (EXCLUSIONS, "holdout_exclusions_sha256")):
        if overall.sha256_file(path) != selection[key]:
            raise ValueError(f"Locked input changed: {path}")
    threshold = float(selection["threshold"])
    if threshold not in overall.THRESHOLDS:
        raise ValueError("Locked threshold is outside the frozen grid")
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    excluded = set(json.loads(EXCLUSIONS.read_text(encoding="utf-8"))["excluded_test_ids"])
    items = enumerate_isic_pairs(HOLDOUT / "images", HOLDOUT / "masks", require_mask=True)
    if len(items) != 1000 or {item.stem for item in items} != set(provenance["items"]):
        raise ValueError("2018 image/mask pairs differ from verified provenance")
    if len(excluded) != 41 or not excluded <= {item.stem for item in items}:
        raise ValueError("Frozen exclusions differ from official 2018 pairs")
    areas = base.area_map(items)
    cuts_full = base.cuts(areas)
    clean_areas = {stem: area for stem, area in areas.items() if stem not in excluded}
    cuts_clean = base.cuts(clean_areas)
    fp32, _ = quant._load_fp32(checkpoint)
    int8, _ = quant._load_quantized(spec_path)
    models = (("fp32", fp32, torch.device("cuda" if torch.cuda.is_available() else "cpu")),
              ("int8", int8, torch.device("cpu")))
    reports = {}
    for name, model, device in models:
        cache = output_dir / f"{name}_fixed_records.json"
        if cache.is_file():
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if payload.get("selection_lock_sha256") != overall.sha256_file(lock_path):
                raise ValueError(f"Cached holdout records use a different lock: {cache}")
            if float(payload.get("threshold", -1)) != threshold:
                raise ValueError(f"Cached holdout records use a different threshold: {cache}")
            rows = validate_cached_rows(payload["records"], areas, cuts_full)
        else:
            model = model.to(device).eval()
            grid = evaluate_grid(model, items, areas, cuts_full, device, (threshold,))
            rows = grid[threshold][1]
            atomic_json(cache, {"selection_lock_sha256": overall.sha256_file(lock_path),
                                "threshold": threshold, "records": rows})
        full = base.summarize(rows, cuts_full)
        clean = [{**row, "band": base.band(float(row["area_ratio"]), cuts_clean)}
                 for row in rows if row["stem"] not in excluded]
        if len(clean) != 959:
            raise ValueError("Clean holdout count is not 959")
        reports[name] = {"full": full, "clean": base.summarize(clean, cuts_clean)}
        print(f"{name}: full={full['all']['dice']:.6f} clean={reports[name]['clean']['all']['dice']:.6f}",
              flush=True)
    atomic_json(output_dir / "final_holdout_report.json", {
        "schema": "role-a-2018-final-holdout-fixed-v1",
        "selection_lock_sha256": overall.sha256_file(lock_path),
        "threshold": threshold, "models": reports,
        "note": "One previously locked checkpoint/threshold; scores are not used to choose a new model",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("lock")
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--development", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--lock", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "lock":
        lock(args.checkpoint, args.spec, args.development, args.output)
    else:
        evaluate(args.lock, args.output_dir)


if __name__ == "__main__":
    main()
