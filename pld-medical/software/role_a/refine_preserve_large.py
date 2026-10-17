"""Conservative repair trial: retain 2016 large-lesion skill after tiny gain.

The exposed ISIC 2016 test set is explicitly a development gate in this trial.
The official ISIC 2017 test set is reserved as a fresh final holdout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from software.role_a import refine_small as base
from software.role_a.data import enumerate_isic_pairs
from software.role_a.metrics import segmentation_loss
from software.role_a.refine_conservative import evaluate_grid, set_blend
from software.role_a.refine_full2017 import OUT as FIRST_OUT, EXCLUSIONS


OUT = base.PROJECT / "runs" / "role_a_preserve_large"
START = FIRST_OUT / "best.pt"
EPOCHS = 2
LR = 0.00001
BLENDS = (0.25, 0.5, 1.0)
THRESHOLDS = (0.5, 0.53, 0.56, 0.6)
SEED = base.SEED + 3


def loss_with_retention(student, target, valid, micro, original2016, teacher):
    core, _ = segmentation_loss(student, target, valid)
    probability = torch.sigmoid(student)
    axes = (1, 2, 3)
    if bool(micro.any()):
        tp = (probability * target * valid).sum(axes)
        fp = (probability * (1 - target) * valid).sum(axes)
        fn = ((1 - probability) * target * valid).sum(axes)
        tversky = 1 - (tp + 1) / (tp + 0.3 * fp + 0.7 * fn + 1)
        core = core + 0.2 * tversky[micro].mean()
    distill = (F.binary_cross_entropy_with_logits(student, torch.sigmoid(teacher),
                                                   reduction="none") * valid).sum(axes) / valid.sum(axes)
    original_nonmicro = original2016 & ~micro
    supplement_nonmicro = ~original2016 & ~micro
    if bool(original_nonmicro.any()):
        core = core + 2.0 * distill[original_nonmicro].mean()
    if bool(supplement_nonmicro.any()):
        core = core + 0.5 * distill[supplement_nonmicro].mean()
    return core


def _clean_items(items, excluded):
    return [item for item in items if item.stem not in excluded]


def run() -> None:
    base.seed_everything(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    exclusions = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))
    external_excluded = set(exclusions["external_validation"])
    train2016_excluded = set(exclusions["isic2016_training"])
    supplement_excluded = set(exclusions["supplemental_training"])
    test_excluded = set(exclusions["official_test_clean_subset_exclusions"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    external_items, external_areas = base.external_items()
    external_items = _clean_items(external_items, external_excluded)
    external_areas = {stem: area for stem, area in external_areas.items() if stem not in external_excluded}
    external_cuts = base.cuts(external_areas)
    test_items = enumerate_isic_pairs(base.TEST / "images", base.TEST / "masks", require_mask=True)
    if len(test_items) != 379:
        raise ValueError("ISIC 2016 development set changed")
    test_items = _clean_items(test_items, test_excluded)
    test_areas = base.area_map(test_items)
    test_cuts = base.cuts(test_areas)
    first_protocol = json.loads((FIRST_OUT / "protocol.json").read_text(encoding="utf-8"))
    external_baseline = first_protocol["baseline"]
    development_baseline = json.loads((FIRST_OUT / "official_test_comparison.json")
                                      .read_text(encoding="utf-8"))["clean_subset_summaries"]["original"]
    original_items = enumerate_isic_pairs(base.TRAIN / "images", base.TRAIN / "masks", require_mask=True)
    supplement_items = enumerate_isic_pairs(base.SUPPLEMENT / "images", base.SUPPLEMENT / "masks", require_mask=True)
    if len(original_items) != 900 or len(supplement_items) != 949:
        raise ValueError("source training datasets changed")
    original_items = _clean_items(original_items, train2016_excluded)
    supplement_items = _clean_items(supplement_items, supplement_excluded)
    if len(original_items) != 897 or len(supplement_items) != 946:
        raise ValueError("reviewed overlap exclusions changed")
    original_ids = {item.stem for item in original_items}
    train_items = original_items + supplement_items
    train_areas = base.area_map(train_items)
    weights = []
    for item in train_items:
        area = train_areas[item.stem]
        if item.stem in original_ids:
            weights.append(3.0 if area > 0.10 else 2.0)
        else:
            weights.append(3.0 if area <= 0.02 else 0.5)
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(train_items),
                                    replacement=True, generator=torch.Generator().manual_seed(SEED))
    batches = base.loader(train_items, augment=True, sampler=sampler)
    student, source_checkpoint = base.load_model(START, device)
    teacher, _ = base.load_model(base.OLD, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    start_state = {name: value.detach().clone() for name, value in student.state_dict().items()}
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    base._save_json(OUT / "protocol.json", {
        "start_checkpoint_sha256": hashlib.sha256(START.read_bytes()).hexdigest(),
        "old_checkpoint_sha256": hashlib.sha256(base.OLD.read_bytes()).hexdigest(),
        "exclusions_sha256": hashlib.sha256(EXCLUSIONS.read_bytes()).hexdigest(),
        "seed": SEED, "epochs": EPOCHS, "lr": LR,
        "train_count": len(train_items), "external_count": len(external_items),
        "development_count": len(test_items), "sampler": "2016 large 3, 2016 other 2, 2017 tiny 3, 2017 other 0.5",
        "teacher_bce": "2016 nonmicro 2.0, 2017 nonmicro 0.5",
        "micro_tversky_weight": 0.2, "blends": BLENDS, "thresholds": THRESHOLDS,
        "external_baseline": external_baseline, "development_baseline": development_baseline,
        "disclosure": "ISIC 2016 test was exposed before this repair trial and is development feedback",
        "final_holdout": "official ISIC 2017 Task 1 test, never used for training or selection",
    })
    history = []
    best_score = float("-inf")
    for epoch in range(1, EPOCHS + 1):
        student.train()
        total_loss = 0.0
        for batch in batches:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            is_original = torch.tensor([Path(name).stem in original_ids for name in batch["filename"]],
                                       device=device)
            micro = torch.tensor([train_areas[Path(name).stem] <= 0.02 for name in batch["filename"]],
                                 device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(image)
            logits = student(image)
            loss = loss_with_retention(logits, target, valid, micro, is_original, teacher_logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
        trained = {name: value.detach().clone() for name, value in student.state_dict().items()}
        for alpha in BLENDS:
            set_blend(student, trained, start_state, alpha)
            path = OUT / "candidates" / f"epoch{epoch:02d}_alpha{alpha:.2f}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            candidate = {"model_state": {k: v.detach().cpu().clone() for k, v in student.state_dict().items()},
                         "config": source_checkpoint["config"], "epoch": epoch, "alpha": alpha,
                         "source_checkpoint": str(START)}
            torch.save(candidate, path)
            ext_grid = evaluate_grid(student, external_items, external_areas, external_cuts,
                                     device, THRESHOLDS)
            dev_grid = evaluate_grid(student, test_items, test_areas, test_cuts,
                                     device, THRESHOLDS)
            for threshold in THRESHOLDS:
                external_metrics, external_records = ext_grid[threshold]
                development_metrics, development_records = dev_grid[threshold]
                accepted = (base.acceptance(external_metrics, external_baseline) and
                            base.acceptance(development_metrics, development_baseline))
                score = min(external_metrics["tiny"]["dice"] - external_baseline["tiny"]["dice"],
                            development_metrics["tiny"]["dice"] - development_baseline["tiny"]["dice"])
                row = {"epoch": epoch, "alpha": alpha, "threshold": threshold,
                       "train_loss": total_loss / len(batches), "candidate": str(path),
                       "accepted": accepted, "score": score,
                       "external": external_metrics, "development2016": development_metrics}
                history.append(row)
                base._save_json(OUT / "history.json", history)
                print(f"epoch={epoch} alpha={alpha} threshold={threshold} "
                      f"ext tiny={external_metrics['tiny']['dice']:.5f} large={external_metrics['large']['dice']:.5f} "
                      f"dev tiny={development_metrics['tiny']['dice']:.5f} "
                      f"large={development_metrics['large']['dice']:.5f} accepted={accepted}", flush=True)
                if accepted and score > best_score:
                    best_score = score
                    torch.save({**candidate, "threshold": threshold,
                                "selection": "dual development gate; 2017 test remains untouched"}, OUT / "best.pt")
                    base._save_json(OUT / "best_metrics.json", {
                        "epoch": epoch, "alpha": alpha, "threshold": threshold,
                        "external": external_metrics, "external_records": external_records,
                        "development2016": development_metrics,
                        "development_records": development_records})
        set_blend(student, trained, start_state, 1.0)
    base._save_json(OUT / "selection.json", {
        "accepted": (OUT / "best.pt").exists(), "score": best_score if best_score > float("-inf") else None,
        "rule": "external and exposed 2016 development tiny gain; all other area-group Dice nondecreasing",
    })


if __name__ == "__main__":
    run()
