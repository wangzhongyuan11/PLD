"""Conservative second post-training trial with paired area-group selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from software.role_a.data import enumerate_isic_pairs
from software.role_a import refine_small as base


OUT = base.PROJECT / "runs" / "role_a_small_balanced_grid"
EPOCHS = 4
LR = 0.00003
MICRO_WEIGHT = 2.0
TVERSKY_WEIGHT = 0.2
DISTILL_WEIGHT = 0.5
BLENDS = (0.25, 0.5, 1.0)
THRESHOLDS = (0.5, 0.53, 0.56, 0.60)


def conservative_loss(student: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                      micro: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    axes = (1, 2, 3)
    probability = torch.sigmoid(student)
    bce = (F.binary_cross_entropy_with_logits(student, target, reduction="none") * valid).sum(axes) / valid.sum(axes)
    tp = (probability * target * valid).sum(axes)
    pred_sum = (probability * valid).sum(axes)
    target_sum = (target * valid).sum(axes)
    dice = 1 - (2 * tp + 1) / (pred_sum + target_sum + 1)
    loss = bce.mean() + dice.mean()
    if bool(micro.any()):
        fp = (probability * (1 - target) * valid).sum(axes)
        fn = ((1 - probability) * target * valid).sum(axes)
        tversky = 1 - (tp + 1) / (tp + 0.3 * fp + 0.7 * fn + 1)
        loss = loss + TVERSKY_WEIGHT * tversky[micro].mean()
    if bool((~micro).any()):
        pixel_distill = (F.binary_cross_entropy_with_logits(student, torch.sigmoid(teacher),
                                                             reduction="none") * valid).sum(axes) / valid.sum(axes)
        loss = loss + DISTILL_WEIGHT * pixel_distill[~micro].mean()
    return loss


def set_blend(model: torch.nn.Module, trained: dict[str, torch.Tensor],
              original: dict[str, torch.Tensor], alpha: float) -> None:
    with torch.no_grad():
        state = model.state_dict()
        for name, value in state.items():
            value.copy_(original[name] + alpha * (trained[name] - original[name]))


@torch.inference_mode()
def evaluate_grid(model: torch.nn.Module, items, areas: dict[str, float],
                  area_cuts: tuple[float, float], device: torch.device,
                  thresholds: tuple[float, ...] = THRESHOLDS):
    """One model forward pass per image for every predeclared binary threshold."""
    model.eval()
    records = {threshold: [] for threshold in thresholds}
    for batch in base.loader(items):
        logits = model(batch["image"].to(device)).cpu()
        for index, filename in enumerate(batch["filename"]):
            stem = Path(filename).stem
            for threshold in thresholds:
                records[threshold].append({"stem": stem, "area_ratio": areas[stem],
                                           "band": base.band(areas[stem], area_cuts),
                                           **base.paired_score(logits[index], batch["mask"][index],
                                                               batch["valid"][index], threshold)})
    return {threshold: (base.summarize(rows, area_cuts), rows)
            for threshold, rows in records.items()}


def run() -> None:
    base.seed_everything(base.SEED + 1)
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext_items, ext_areas = base.external_items()
    thresholds = base.cuts(ext_areas)
    baseline_model, old_checkpoint = base.load_model(base.OLD, device)
    baseline_summary, baseline_records = base.evaluate(baseline_model, ext_items, ext_areas, thresholds, device)
    train_items = enumerate_isic_pairs(base.TRAIN / "images", base.TRAIN / "masks", require_mask=True)
    extra = enumerate_isic_pairs(base.SUPPLEMENT / "images", base.SUPPLEMENT / "masks", require_mask=True)
    supplement_provenance = json.loads((base.SUPPLEMENT / "provenance.json").read_text(encoding="utf-8"))
    if supplement_provenance.get("complete") is not True or set(supplement_provenance["items"]) != {x.stem for x in extra}:
        raise ValueError("supplemental provenance does not match files")
    if len(train_items) != 900 or len(extra) != 190:
        raise ValueError("training datasets are incomplete")
    if {x.stem for x in train_items} & {x.stem for x in extra}:
        raise ValueError("training IDs overlap")
    if {x.stem for x in extra} & {x.stem for x in ext_items}:
        raise ValueError("external validation overlaps training")
    official_test = {x.stem for x in enumerate_isic_pairs(base.TEST / "images", base.TEST / "masks", require_mask=True)}
    if {x.stem for x in extra} & official_test:
        raise ValueError("supplemental data overlap official test")
    train_items.extend(extra)
    train_areas = base.area_map(train_items)
    micro_n = sum(value <= base.MICRO_TRAIN_MAX for value in train_areas.values())
    weights = torch.tensor([MICRO_WEIGHT if train_areas[x.stem] <= base.MICRO_TRAIN_MAX else 1.0
                            for x in train_items], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_items), replacement=True,
                                    generator=torch.Generator().manual_seed(base.SEED + 1))
    batches = base.loader(train_items, augment=True, sampler=sampler)
    student, _ = base.load_model(base.OLD, device)
    teacher = baseline_model
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    original = {name: tensor.detach().clone() for name, tensor in student.state_dict().items()}
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    base._save_json(OUT / "protocol.json", {
        "source_checkpoint_sha256": hashlib.sha256(base.OLD.read_bytes()).hexdigest(),
        "seed": base.SEED + 1, "epochs": EPOCHS, "lr": LR, "micro_area_upper": base.MICRO_TRAIN_MAX,
        "micro_images": micro_n, "train_images": len(train_items), "external_images": len(ext_items),
        "micro_sampling_weight": MICRO_WEIGHT, "micro_tversky_weight": TVERSKY_WEIGHT,
        "non_micro_teacher_weight": DISTILL_WEIGHT, "blend_coefficients": BLENDS,
        "prediction_thresholds": THRESHOLDS,
        "external_group_thresholds": thresholds,
        "acceptance": "tiny Dice > baseline; medium, large, all Dice >= baseline",
        "baseline": baseline_summary})
    history: list[dict[str, object]] = []
    best = baseline_summary["tiny"]["dice"]
    for epoch in range(1, EPOCHS + 1):
        student.train()
        sum_loss = 0.0
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
            loss = conservative_loss(logits, target, valid, micro, old_logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()
            sum_loss += float(loss.detach())
        trained = {name: tensor.detach().clone() for name, tensor in student.state_dict().items()}
        for alpha in BLENDS:
            set_blend(student, trained, original, alpha)
            checkpoint = {"model_state": {name: tensor.detach().cpu().clone()
                                          for name, tensor in student.state_dict().items()},
                          "config": old_checkpoint["config"], "epoch": epoch, "alpha": alpha,
                          "source_checkpoint": str(base.OLD)}
            candidate_path = OUT / "candidates" / f"epoch{epoch:02d}_alpha{alpha:.2f}.pt"
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, candidate_path)
            for threshold, (summary, records) in evaluate_grid(student, ext_items, ext_areas, thresholds, device).items():
                accepted = base.acceptance(summary, baseline_summary)
                entry = {"epoch": epoch, "alpha": alpha, "threshold": threshold,
                         "candidate_path": str(candidate_path), "train_loss": sum_loss / len(batches),
                         "accepted": accepted, "metrics": summary}
                history.append(entry)
                base._save_json(OUT / "history.json", history)
                print(f"epoch={epoch} alpha={alpha} threshold={threshold} tiny={summary['tiny']['dice']:.5f} "
                      f"medium={summary['medium']['dice']:.5f} large={summary['large']['dice']:.5f} "
                      f"all={summary['all']['dice']:.5f} accepted={accepted}", flush=True)
                if accepted and summary["tiny"]["dice"] > best:
                    best = summary["tiny"]["dice"]
                    selected = dict(checkpoint)
                    selected.update({"threshold": threshold, "external_validation": summary,
                                     "selection": "strict area-group gate"})
                    torch.save(selected, OUT / "best.pt")
                    base._save_json(OUT / "best_external.json", {"epoch": epoch, "alpha": alpha,
                                                                 "threshold": threshold,
                                                                 "metrics": summary, "records": records})
        set_blend(student, trained, original, 1.0)
    base._save_json(OUT / "selection.json", {"accepted": (OUT / "best.pt").exists(),
                                            "best_tiny_dice": best, "baseline": baseline_summary})


if __name__ == "__main__":
    run()
