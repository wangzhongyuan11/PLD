"""Area-aware post-training of the unchanged A-role TinyUNet.

Model selection uses disjoint ISIC 2017 Task 1 validation masks. The official
ISIC 2016 test set is accessed only by the explicit ``test`` subcommand.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from software.role_a.data import ISICDataset, IsicPaths, enumerate_isic_pairs, isic_collate
from software.role_a.model import build_model
from software.role_a.train import seed_everything


PROJECT = Path(__file__).resolve().parents[2]
OLD = PROJECT / "runs" / "role_a_tinyunet" / "best.pt"
EXTERNAL = PROJECT / "data" / "raw" / "isic2017_validation"
TRAIN = PROJECT / "data" / "raw" / "isic2016" / "train"
SUPPLEMENT = PROJECT / "data" / "raw" / "isic2017_training"
TEST = PROJECT / "data" / "raw" / "isic2016" / "test"
OUT = PROJECT / "runs" / "role_a_small_refine"
SIZE = 256
SEED = 20260923
MICRO_TRAIN_MAX = 0.02


def area_ratio(mask: Path) -> float:
    with Image.open(mask) as image:
        return float((np.asarray(image.convert("L")) >= 127).mean())


def area_map(items: list[IsicPaths]) -> dict[str, float]:
    return {item.stem: area_ratio(item.mask) for item in items if item.mask is not None}


def cuts(areas: dict[str, float]) -> tuple[float, float]:
    values = np.fromiter(areas.values(), dtype=np.float64)
    if len(values) < 10:
        raise ValueError("at least 10 masks are required for area bands")
    return float(np.quantile(values, 0.1)), float(np.quantile(values, 0.5))


def band(area: float, thresholds: tuple[float, float]) -> str:
    return "tiny" if area <= thresholds[0] else "medium" if area <= thresholds[1] else "large"


def summarize(records: list[dict[str, Any]], thresholds: tuple[float, float]) -> dict[str, Any]:
    result: dict[str, Any] = {"thresholds": {"q10": thresholds[0], "q50": thresholds[1]}}
    for name in ("all", "tiny", "medium", "large"):
        rows = records if name == "all" else [r for r in records if r["band"] == name]
        if not rows:
            raise ValueError(f"validation band {name} has no images")
        result[name] = {"n": len(rows), **{key: float(np.mean([r[key] for r in rows]))
                                            for key in ("dice", "iou", "recall", "precision")},
                        "empty_prediction": sum(int(r["empty_prediction"]) for r in rows)}
    return result


def acceptance(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    """Strict paired mean gate; no tolerance or test-set tuning."""
    return (candidate["tiny"]["dice"] > baseline["tiny"]["dice"] and
            all(candidate[key]["dice"] >= baseline[key]["dice"]
                for key in ("medium", "large", "all")))


def paired_score(logit: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                 threshold: float = 0.5) -> dict[str, Any]:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one")
    keep = valid > 0
    truth = (target > 0) & keep
    prediction = (torch.sigmoid(logit) >= threshold) & keep
    tp = int((truth & prediction).sum())
    fp = int((prediction & ~truth & keep).sum())
    fn = int((truth & ~prediction).sum())
    return {"dice": 1.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn),
            "iou": 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn),
            "recall": 1.0 if tp + fn == 0 else tp / (tp + fn),
            "precision": 1.0 if tp + fp == 0 else tp / (tp + fp),
            "empty_prediction": tp + fp == 0}


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if int(config["size"]) != SIZE:
        raise ValueError("checkpoint size differs from current 256-pixel pipeline")
    model = build_model(base_channels=int(config["base_channels"]),
                        num_pools=int(config["num_pools"]), input_shape=(3, SIZE, SIZE))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device), checkpoint


def external_items() -> tuple[list[IsicPaths], dict[str, float]]:
    provenance = json.loads((EXTERNAL / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("complete") is not True or provenance.get("image_count") != 147:
        raise ValueError("external dataset is not completely verified")
    items = enumerate_isic_pairs(EXTERNAL / "images", EXTERNAL / "masks", require_mask=True)
    observed = {item.stem for item in items}
    if observed != set(provenance["items"]):
        raise ValueError("external images differ from verified provenance")
    existing = {item.stem for role in ("train", "test")
                for item in enumerate_isic_pairs(PROJECT / "data" / "raw" / "isic2016" / role / "images",
                                                 PROJECT / "data" / "raw" / "isic2016" / role / "masks")}
    if observed & existing:
        raise ValueError("external validation overlaps ISIC 2016")
    return items, {stem: float(row["area_ratio"]) for stem, row in provenance["items"].items()}


def loader(items: list[IsicPaths], augment: bool = False,
           sampler: WeightedRandomSampler | None = None) -> DataLoader:
    return DataLoader(ISICDataset(items, size=SIZE, augment=augment), batch_size=4,
                      shuffle=False, sampler=sampler, num_workers=2,
                      collate_fn=isic_collate, pin_memory=True, persistent_workers=True)


@torch.inference_mode()
def evaluate(model: torch.nn.Module, items: list[IsicPaths], areas: dict[str, float],
             thresholds: tuple[float, float], device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    records: list[dict[str, Any]] = []
    for batch in loader(items):
        logits = model(batch["image"].to(device)).cpu()
        for index, filename in enumerate(batch["filename"]):
            stem = Path(filename).stem
            record = {"stem": stem, "area_ratio": areas[stem],
                      "band": band(areas[stem], thresholds),
                      **paired_score(logits[index], batch["mask"][index], batch["valid"][index])}
            records.append(record)
    return summarize(records, thresholds), records


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def baseline(device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    items, areas = external_items()
    model, _ = load_model(OLD, device)
    summary, records = evaluate(model, items, areas, cuts(areas), device)
    _save_json(OUT / "external_baseline.json", {"source_checkpoint": str(OLD), "metrics": summary,
                                                 "records": records})
    return summary, records


def _loss(student: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
          tiny: torch.Tensor, teacher: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    axes = (1, 2, 3)
    probability = torch.sigmoid(student)
    bce = (F.binary_cross_entropy_with_logits(student, target, reduction="none") * valid).sum(axes) / valid.sum(axes)
    intersection = (probability * target * valid).sum(axes)
    pred_sum = (probability * valid).sum(axes)
    true_sum = (target * valid).sum(axes)
    dice = 1 - (2 * intersection + 1) / (pred_sum + true_sum + 1)
    fp = (probability * (1 - target) * valid).sum(axes)
    fn = ((1 - probability) * target * valid).sum(axes)
    tversky = 1 - (intersection + 1) / (intersection + 0.3 * fp + 0.7 * fn + 1)
    distill = (((probability - torch.sigmoid(teacher)) ** 2) * valid).sum(axes) / valid.sum(axes)
    loss = bce.mean() + dice.mean()
    if bool(tiny.any()):
        loss = loss + 0.4 * tversky[tiny].mean()
    if bool((~tiny).any()):
        loss = loss + 0.1 * distill[~tiny].mean()
    return loss, {"bce": float(bce.mean().detach()), "dice_loss": float(dice.mean().detach()),
                  "tiny_tversky": float(tversky[tiny].mean().detach()) if bool(tiny.any()) else 0.0}


def train(epochs: int, lr: float, device: torch.device) -> None:
    seed_everything(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    base_summary, _ = baseline(device)
    ext_items, ext_areas = external_items()
    ext_cuts = cuts(ext_areas)
    train_items = enumerate_isic_pairs(TRAIN / "images", TRAIN / "masks", require_mask=True)
    if len(train_items) != 900:
        raise ValueError("expected all 900 official ISIC 2016 train pairs")
    supplement_provenance = json.loads((SUPPLEMENT / "provenance.json").read_text(encoding="utf-8"))
    if supplement_provenance.get("complete") is not True or supplement_provenance.get("image_count") != 120:
        raise ValueError("supplemental tiny-lesion data are not completely verified")
    supplement_items = enumerate_isic_pairs(SUPPLEMENT / "images", SUPPLEMENT / "masks", require_mask=True)
    if {item.stem for item in supplement_items} != set(supplement_provenance["items"]):
        raise ValueError("supplemental data differ from verified provenance")
    if {item.stem for item in train_items} & {item.stem for item in supplement_items}:
        raise ValueError("supplemental data overlap ISIC 2016 training")
    if {item.stem for item in supplement_items} & {item.stem for item in ext_items}:
        raise ValueError("supplemental data overlap external validation")
    train_items.extend(supplement_items)
    train_areas = area_map(train_items)
    train_cuts = cuts(train_areas)
    weights = torch.tensor([3.0 if train_areas[item.stem] <= MICRO_TRAIN_MAX else 1.0
                            for item in train_items], dtype=torch.double)
    generator = torch.Generator().manual_seed(SEED)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_items), replacement=True, generator=generator)
    train_loader = loader(train_items, augment=True, sampler=sampler)
    model, original = load_model(OLD, device)
    teacher, _ = load_model(OLD, device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, Any]] = []
    best_tiny = base_summary["tiny"]["dice"]
    _save_json(OUT / "protocol.json", {"seed": SEED, "source_checkpoint_sha256": hashlib.sha256(OLD.read_bytes()).hexdigest(),
                                       "train_count": len(train_items), "external_count": len(ext_items),
                                       "training_area_thresholds": train_cuts,
                                       "external_area_thresholds": ext_cuts,
                                       "micro_training_area_upper": MICRO_TRAIN_MAX,
                                       "tiny_sampling_weight": 3.0, "learning_rate": lr,
                                       "planned_epochs": epochs, "selection": "tiny Dice must increase; medium, large, and overall Dice must not decrease on external validation"})
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            tiny = torch.tensor([train_areas[Path(name).stem] <= MICRO_TRAIN_MAX
                                 for name in batch["filename"]], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(image)
            logits = model(image)
            loss, _ = _loss(logits, target, valid, tiny, teacher_logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
        summary, records = evaluate(model, ext_items, ext_areas, ext_cuts, device)
        accepted = acceptance(summary, base_summary)
        row = {"epoch": epoch, "train_loss": total_loss / len(train_loader),
               "accepted": accepted, "metrics": summary}
        history.append(row)
        _save_json(OUT / "history.json", history)
        print(f"epoch={epoch} loss={row['train_loss']:.4f} tiny={summary['tiny']['dice']:.5f} "
              f"medium={summary['medium']['dice']:.5f} large={summary['large']['dice']:.5f} "
              f"all={summary['all']['dice']:.5f} accepted={accepted}", flush=True)
        if accepted and summary["tiny"]["dice"] > best_tiny:
            best_tiny = summary["tiny"]["dice"]
            checkpoint = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "config": original["config"], "epoch": epoch,
                          "external_validation": summary, "selection": "strict area-group gate",
                          "source_checkpoint": str(OLD)}
            torch.save(checkpoint, OUT / "best.pt")
            _save_json(OUT / "best_external.json", {"epoch": epoch, "metrics": summary,
                                                    "records": records})
    _save_json(OUT / "selection.json", {"accepted": (OUT / "best.pt").exists(),
                                            "best_tiny_dice": best_tiny,
                                            "baseline": base_summary})


def final_test(device: torch.device) -> None:
    chosen = OUT / "best.pt"
    if not chosen.exists():
        raise ValueError("no candidate passed the external non-regression gate")
    items = enumerate_isic_pairs(TEST / "images", TEST / "masks", require_mask=True)
    areas = area_map(items)
    thresholds = cuts(areas)  # Fixed from test labels, independent of predictions.
    old_model, _ = load_model(OLD, device)
    new_model, _ = load_model(chosen, device)
    old_summary, old_records = evaluate(old_model, items, areas, thresholds, device)
    new_summary, new_records = evaluate(new_model, items, areas, thresholds, device)
    _save_json(OUT / "official_test_comparison.json", {"old": old_summary, "new": new_summary,
                                                         "old_records": old_records, "new_records": new_records})
    print(json.dumps({"old": old_summary, "new": new_summary}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("baseline", "train", "test"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.00008)
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0:
        raise ValueError("epochs and lr must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.command == "baseline":
        print(json.dumps(baseline(device)[0], ensure_ascii=False, indent=2))
    elif args.command == "train":
        train(args.epochs, args.lr, device)
    else:
        final_test(device)


if __name__ == "__main__":
    main()
