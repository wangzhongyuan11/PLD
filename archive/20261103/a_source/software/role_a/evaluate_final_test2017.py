"""One-time paired final evaluation on ISIC 2017 Task 1 test, including screened subset."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from software.role_a import refine_small as base
from software.role_a.adaptive_inference import score_adaptive
from software.role_a.data import enumerate_isic_pairs
from software.role_a.refine_from_baseline_v3 import atomic_write_json, sha256_file


TEST = base.PROJECT / "data" / "raw" / "isic2017_test"
TRIAL = base.PROJECT / "runs" / "role_a_from_baseline_v3"
SELECTION = TRIAL / "final_selection.json"
EXCLUSIONS = base.PROJECT / "runs" / "role_a_full2017" / "test2017_overlap_exclusions.json"
AUDIT = EXCLUSIONS.with_name("test2017_overlap_audit.json")
OUTPUT = TRIAL / "final_test2017.json"


def validate_inputs() -> tuple[list, set[str], dict, Path]:
    if OUTPUT.exists():
        raise FileExistsError(f"Final holdout already evaluated: {OUTPUT}")
    choice = json.loads(SELECTION.read_text(encoding="utf-8"))
    if choice["frozen_before_official_test2017_evaluation"] is not True:
        raise ValueError("Selection was not frozen before holdout")
    candidate = (TRIAL / choice["candidate"]).resolve()
    if not candidate.is_relative_to(TRIAL.resolve()):
        raise ValueError("Candidate path escaped the trial directory")
    if sha256_file(candidate) != choice["candidate_sha256"]:
        raise ValueError("Frozen candidate hash mismatch")
    provenance = json.loads((TEST / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("complete") is not True or provenance.get("image_count") != 600:
        raise ValueError("Official test set provenance is incomplete")
    items = enumerate_isic_pairs(TEST / "images", TEST / "masks", require_mask=True)
    if len(items) != 600 or {item.stem for item in items} != set(provenance["items"]):
        raise ValueError("Official test pairs do not match verified provenance")
    frozen = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    hits = {pair["left"] for name in ("test2017_vs_train", "test2017_vs_external",
                                      "test2017_vs_test2016")
            for pair in audit[name]["dhash_close_pairs"]}
    excluded = set(frozen["clean_exclusions"])
    if hits != excluded or len(items) - len(excluded) != frozen["clean_count_expected"]:
        raise ValueError("Frozen overlap exclusions do not match pre-test audit")
    if excluded - {item.stem for item in items}:
        raise ValueError("Frozen exclusions include IDs outside the test set")
    return items, excluded, choice, candidate


def grouped_summary(records: list[dict], areas: dict[str, float],
                    excluded: set[str]) -> dict:
    result = {}
    for group, selected in (("all600", records),
                            ("screened583", [row for row in records if row["stem"] not in excluded])):
        area_cuts = base.cuts({row["stem"]: areas[row["stem"]] for row in selected})
        summaries = {}
        for kind in ("baseline", "candidate"):
            rows = [{"stem": row["stem"], "area_ratio": areas[row["stem"]],
                     "band": base.band(areas[row["stem"]], area_cuts), **row[kind]}
                    for row in selected]
            summaries[kind] = base.summarize(rows, area_cuts)
        result[group] = {"count": len(selected), "baseline": summaries["baseline"],
                         "candidate": summaries["candidate"],
                         "accepted": base.acceptance(summaries["candidate"], summaries["baseline"]),
                         "dice_delta": {band: summaries["candidate"][band]["dice"] -
                                        summaries["baseline"][band]["dice"]
                                        for band in ("all", "tiny", "medium", "large")}}
    return result


@torch.inference_mode()
def evaluate(items: list, choice: dict, candidate_path: Path) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline, _ = base.load_model(base.OLD, device)
    candidate, _ = base.load_model(candidate_path, device)
    baseline.eval()
    candidate.eval()
    rule = choice["rule"]
    output = []
    for batch in base.loader(items):
        image = batch["image"].to(device)
        old_logits = baseline(image).cpu()
        new_logits = candidate(image).cpu()
        for index, filename in enumerate(batch["filename"]):
            target, valid = batch["mask"][index], batch["valid"][index]
            output.append({
                "stem": Path(filename).stem,
                "baseline": base.paired_score(old_logits[index], target, valid, 0.5),
                "candidate": score_adaptive(
                    new_logits[index], target, valid,
                    area_cut=float(rule["area_cut"]),
                    small_threshold=float(rule["threshold_if_area_below_cut"]),
                    large_threshold=float(rule["threshold_if_area_at_or_above_cut"])),
            })
    return output


def main(validate_only: bool = False) -> None:
    items, excluded, choice, candidate = validate_inputs()
    print(f"Preflight: 600 verified pairs, {len(excluded)} frozen exclusions, "
          f"candidate SHA256 {choice['candidate_sha256']}", flush=True)
    if validate_only:
        return
    areas = base.area_map(items)
    records = evaluate(items, choice, candidate)
    if len(records) != 600 or {row["stem"] for row in records} != set(areas):
        raise ValueError("Final evaluation records are incomplete")
    summaries = grouped_summary(records, areas, excluded)
    payload = {
        "schema": "role-a-final-test2017-paired-v1",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_sha256": sha256_file(SELECTION),
        "candidate_sha256": sha256_file(candidate),
        "baseline_sha256": sha256_file(base.OLD),
        "test_provenance_sha256": sha256_file(TEST / "provenance.json"),
        "overlap_exclusions_sha256": sha256_file(EXCLUSIONS),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparison": summaries,
        "records": records,
        "interpretation": "One-time final holdout check; screened583 is primary generalization evidence. Do not tune on these results.",
    }
    atomic_write_json(OUTPUT, payload)
    for name, row in summaries.items():
        print(name, "accepted", row["accepted"], "dice_delta", row["dice_delta"], flush=True)
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    main(args.validate_only)
