"""Shared test-set geometry and per-image metrics. No hardware dependency."""
import numpy as np
from PIL import Image, ImageOps


def target_letterbox(path, metadata, size=256):
    with Image.open(path) as source:
        mask = ImageOps.exif_transpose(source).convert("L")
    if mask.size != (metadata["original_width"], metadata["original_height"]):
        raise ValueError("image and target dimensions differ")
    filters = getattr(Image, "Resampling", Image)
    mask = mask.resize((metadata["resized_width"], metadata["resized_height"]), filters.NEAREST)
    result = np.zeros((size, size), dtype=np.bool_)
    x, y = metadata["left"], metadata["top"]
    result[y:y+mask.height, x:x+mask.width] = np.asarray(mask) >= 128
    return result


def segmentation_metrics(prediction, target, valid):
    prediction, target, valid = (np.asarray(a, dtype=np.bool_) for a in (prediction, target, valid))
    if prediction.shape != target.shape or target.shape != valid.shape or not valid.any():
        raise ValueError("metric shapes must match and valid region must be nonempty")
    p, t = prediction[valid], target[valid]
    intersection = int(np.count_nonzero(p & t))
    predicted, truth = int(p.sum()), int(t.sum())
    union, total = predicted + truth - intersection, predicted + truth
    return {"dice": 2*intersection/total if total else 1.0,
            "iou": intersection/union if union else 1.0,
            "intersection": intersection, "prediction_pixels": predicted,
            "target_pixels": truth, "valid_pixels": int(valid.sum())}
