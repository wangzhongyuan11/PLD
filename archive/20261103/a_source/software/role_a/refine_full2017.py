"""Post-train on every non-overlapping 2017 training image, retaining the old core loss."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from software.role_a.data import enumerate_isic_pairs
from software.role_a.metrics import segmentation_loss
from software.role_a import refine_small as base
from software.role_a.refine_conservative import BLENDS, evaluate_grid, set_blend


OUT = base.PROJECT / "runs" / "role_a_full2017"
EXCLUSIONS = OUT / "overlap_exclusions.json"
EPOCHS = 5
LR = 0.00003
MICRO_WEIGHT = 2.0
MICRO_TVERSKY = 0.2
TEACHER_WEIGHT = 0.5
SEED = base.SEED + 2
THRESHOLDS = (0.5, 0.53, 0.56, 0.60, 0.65, 0.70)


def full_loss(student: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
              micro: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Original batch BCE+Dice, plus real micro-lesion recall and teacher retention."""
    core, _ = segmentation_loss(student, target, valid)
    axes = (1, 2, 3)
    probability = torch.sigmoid(student)
    if bool(micro.any()):
        tp = (probability * target * valid).sum(axes)
        fp = (probability * (1 - target) * valid).sum(axes)
        fn = ((1 - probability) * target * valid).sum(axes)
        tversky = 1 - (tp + 1) / (tp + 0.3 * fp + 0.7 * fn + 1)
        core = core + MICRO_TVERSKY * tversky[micro].mean()
    if bool((~micro).any()):
        distill = (F.binary_cross_entropy_with_logits(student, torch.sigmoid(teacher),
                                                       reduction="none") * valid).sum(axes) / valid.sum(axes)
        core = core + TEACHER_WEIGHT * distill[~micro].mean()
    return core


def run() -> None:
    base.seed_everything(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    exclusions = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))
    if exclusions.get("schema") != "role-a-reviewed-visual-overlap-v1":
        raise ValueError("missing reviewed visual-overlap exclusions")
    excluded_external = set(exclusions["external_validation"])
    excluded_training = set(exclusions["supplemental_training"])
    excluded_2016_training = set(exclusions["isic2016_training"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    external_items, external_areas = base.external_items()
    if not excluded_external <= {item.stem for item in external_items}:
        raise ValueError("external overlap exclusion ID absent from verified data")
    external_items = [item for item in external_items if item.stem not in excluded_external]
    external_areas = {stem: area for stem, area in external_areas.items() if stem not in excluded_external}
    area_cuts = base.cuts(external_areas)
    old_model, old_checkpoint = base.load_model(base.OLD, device)
    baseline, _ = base.evaluate(old_model, external_items, external_areas, area_cuts, device)
    train_items = enumerate_isic_pairs(base.TRAIN / "images", base.TRAIN / "masks", require_mask=True)
    supplement_items = enumerate_isic_pairs(base.SUPPLEMENT / "images", base.SUPPLEMENT / "masks", require_mask=True)
    provenance = json.loads((base.SUPPLEMENT / "provenance.json").read_text(encoding="utf-8"))
    if (len(train_items) != 900 or len(supplement_items) != 949 or
            provenance.get("complete") is not True or set(provenance["items"]) != {x.stem for x in supplement_items}):
        raise ValueError("full 2017 supplement is incomplete")
    if not excluded_2016_training <= {item.stem for item in train_items}:
        raise ValueError("2016 training exclusion ID absent from official data")
    train_items = [item for item in train_items if item.stem not in excluded_2016_training]
    if not excluded_training <= {item.stem for item in supplement_items}:
        raise ValueError("training overlap exclusion ID absent from verified data")
    supplement_items = [item for item in supplement_items if item.stem not in excluded_training]
    train_ids = {x.stem for x in train_items}
    supplement_ids = {x.stem for x in supplement_items}
    test_ids = {x.stem for x in enumerate_isic_pairs(base.TEST / "images", base.TEST / "masks", require_mask=True)}
    if supplement_ids & (train_ids | test_ids | {x.stem for x in external_items}):
        raise ValueError("supplement overlaps train, test, or external validation")
    train_items.extend(supplement_items)
    train_areas = base.area_map(train_items)
    micro_count = sum(v <= base.MICRO_TRAIN_MAX for v in train_areas.values())
    weights = torch.tensor([MICRO_WEIGHT if train_areas[x.stem] <= base.MICRO_TRAIN_MAX else 1.0
                            for x in train_items], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, len(train_items), replacement=True,
                                    generator=torch.Generator().manual_seed(SEED))
    batches = base.loader(train_items, augment=True, sampler=sampler)
    student, _ = base.load_model(base.OLD, device)
    teacher = old_model
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    original = {key: value.detach().clone() for key, value in student.state_dict().items()}
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    base._save_json(OUT / "protocol.json", {
        "source_checkpoint_sha256": hashlib.sha256(base.OLD.read_bytes()).hexdigest(),
        "supplement_provenance_sha256": hashlib.sha256((base.SUPPLEMENT / "provenance.json").read_bytes()).hexdigest(),
        "overlap_exclusions_sha256": hashlib.sha256(EXCLUSIONS.read_bytes()).hexdigest(),
        "excluded_external_ids": sorted(excluded_external),
        "excluded_training_ids": sorted(excluded_training),
        "excluded_2016_training_ids": sorted(excluded_2016_training),
        "seed": SEED, "train_count": len(train_items), "micro_count": micro_count,
        "external_count": len(external_items), "external_area_cuts": area_cuts,
        "epochs": EPOCHS, "lr": LR, "micro_sampling_weight": MICRO_WEIGHT,
        "micro_tversky_weight": MICRO_TVERSKY, "teacher_weight": TEACHER_WEIGHT,
        "core_loss": "original batch-level BCE + soft Dice", "blends": BLENDS,
        "thresholds": THRESHOLDS,
        "acceptance": "external tiny Dice > baseline, medium/large/all Dice >= baseline",
        "baseline": baseline})
    history: list[dict[str, object]] = []
    best = baseline["tiny"]["dice"]
    for epoch in range(1, EPOCHS + 1):
        student.train()
        loss_sum = 0.0
        for batch in batches:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            micro = torch.tensor([train_areas[Path(name).stem] <= base.MICRO_TRAIN_MAX
                                  for name in batch["filename"]], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                old_logits = teacher(image)
            logits = student(image)
            loss = full_loss(logits, target, valid, micro, old_logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach())
        trained = {key: value.detach().clone() for key, value in student.state_dict().items()}
        for alpha in BLENDS:
            set_blend(student, trained, original, alpha)
            candidate = {"model_state": {k: v.detach().cpu().clone() for k, v in student.state_dict().items()},
                         "config": old_checkpoint["config"], "epoch": epoch, "alpha": alpha,
                         "source_checkpoint": str(base.OLD)}
            path = OUT / "candidates" / f"epoch{epoch:02d}_alpha{alpha:.2f}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(candidate, path)
            for threshold, (metrics, records) in evaluate_grid(student, external_items, external_areas,
                                                               area_cuts, device, THRESHOLDS).items():
                accepted = base.acceptance(metrics, baseline)
                row = {"epoch": epoch, "alpha": alpha, "threshold": threshold,
                       "train_loss": loss_sum / len(batches), "candidate": str(path),
                       "accepted": accepted, "metrics": metrics}
                history.append(row)
                base._save_json(OUT / "history.json", history)
                print(f"epoch={epoch} alpha={alpha} threshold={threshold} tiny={metrics['tiny']['dice']:.5f} "
                      f"medium={metrics['medium']['dice']:.5f} large={metrics['large']['dice']:.5f} "
                      f"all={metrics['all']['dice']:.5f} accepted={accepted}", flush=True)
                if accepted and metrics["tiny"]["dice"] > best:
                    best = metrics["tiny"]["dice"]
                    chosen = {**candidate, "threshold": threshold,
                              "external_validation": metrics, "selection": "strict area-group gate"}
                    torch.save(chosen, OUT / "best.pt")
                    base._save_json(OUT / "best_external.json", {"epoch": epoch, "alpha": alpha,
                                                                 "threshold": threshold,
                                                                 "metrics": metrics, "records": records})
        set_blend(student, trained, original, 1.0)
    base._save_json(OUT / "selection.json", {"accepted": (OUT / "best.pt").exists(),
                                            "best_tiny_dice": best, "baseline": baseline})


if __name__ == "__main__":
    run()
