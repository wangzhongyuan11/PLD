"""Prepare versioned FP32/INT8 references for a frozen named image collection."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import PIL
from PIL import features
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference.tinyunet import Model, DEFAULT_MODEL_ROOT, FINAL_SPEC_SHA256, digest, threshold_contract
from reference.preprocess import decode_rgb, letterbox_rgb, quantize_input
from reference.evaluation import target_letterbox, segmentation_metrics
from reference.datasets import collection, validate_manifest, CONTRACT_PATH
from reference.fp32 import FloatModel


def index_files(roots, suffix, mask=False):
    result = {}
    for root in roots:
        for path in root.rglob(suffix):
            name = path.stem.removesuffix("_Segmentation").removesuffix("_segmentation") if mask else path.stem
            if name in result and result[name].resolve() != path.resolve():
                raise ValueError("duplicate source ID: " + name)
            result[name] = path
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_ROOT)
    p.add_argument("--collection", default="final_dev638")
    p.add_argument("--images", type=Path, action="append", help="repeat for multiple source directories")
    p.add_argument("--masks", type=Path, action="append", help="repeat for multiple source directories")
    p.add_argument("--limit", type=int, help="smoke subset; never full-collection acceptance")
    args = p.parse_args()
    selected = collection(args.collection)
    if args.limit is not None and not 1 <= args.limit < selected["count"]:
        p.error("limit must be positive and smaller than the complete collection")
    if (args.output/"manifest.json").exists():
        raise FileExistsError("preserve completed references; choose a new output directory")
    model = Model(args.model)
    torch.set_num_threads(2)
    fp32 = FloatModel(model.root.parent/"best.pt")
    images = index_files(args.images or [ROOT/"ISBI2016_ISIC_Part1_Test_Data"], "*.jpg")
    targets = index_files(args.masks or [ROOT/"ISBI2016_ISIC_Part1_Test_GroundTruth"], "*.png", mask=True)
    cases = selected["cases"][:args.limit]
    missing = [r["id"] for r in cases if r["id"] not in images or r["id"] not in targets]
    if missing:
        raise ValueError(f"missing JPEG/GT for {len(missing)} IDs, first={missing[:10]}")
    model.run(model.golden(0, "input"), verify=True)
    thresholds = [threshold_contract(t, model.layers[-1]["output_scale"]) for t in (.45, .5, .53, .56, .6)]
    a_records = {}
    if model.spec_hash == FINAL_SPEC_SHA256:
        for kind in ("fp32", "int8"):
            for source in {r["source"] for r in cases}:
                path = model.root.parent/"metrics"/f"{kind}_{source}.json"
                data = json.loads(path.read_text(encoding="utf-8"))
                if data["identity"]["spec_sha256"] != model.spec_hash:
                    raise ValueError("A metric identity differs from selected model")
                a_records[kind, source] = {r["stem"]: r for r in data["records"]}
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 2, "passed": False, "complete_test_set": False, "records": [],
              "collection": {k: selected[k] for k in ("name", "role", "primary", "count", "ids_sha256")},
              "dataset_contract_sha256": digest(CONTRACT_PATH), "threshold": model.threshold,
              "input_scale": model.layers[0]["input_scale"],
              "protocol": "per-image macro mean; model-resolution letterbox valid region; nearest GT; inclusive threshold; empty/empty=1",
              "model_spec_sha256": model.spec_hash, "model_manifest_sha256": model.manifest_hash,
              "checkpoint_sha256": digest(model.root.parent/"best.pt"),
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__,
                          "jpeg": features.version("jpg"), "torch": torch.__version__},
              "source_hashes": {str(f.relative_to(ROOT)): digest(f) for f in
                                [Path(__file__), ROOT/"reference/fp32.py", ROOT/"reference/evaluation.py",
                                 ROOT/"reference/preprocess.py", ROOT/"reference/tinyunet.py", ROOT/"reference/datasets.py"]}}
    try:
        max_delta = {"fp32": 0., "int8": 0.}
        a_compared = 0
        for index, case in enumerate(cases):
            name = case["id"]
            begin = time.perf_counter()
            image = decode_rgb(images[name])
            decode_s = time.perf_counter()-begin
            begin = time.perf_counter()
            real, valid, meta = letterbox_rgb(image)
            x = quantize_input(real, model.layers[0]["input_scale"])
            prep_s = time.perf_counter()-begin
            target = target_letterbox(targets[name], meta)
            begin = time.perf_counter()
            floating = fp32.run(real)[0, 0]
            fp32_s = time.perf_counter()-begin
            begin = time.perf_counter()
            head = model.run(x)[-1]
            int8_s = time.perf_counter()-begin
            fp32_mask = floating >= model.threshold["threshold_logit"]
            artifact = args.output/(name+".npz")
            np.savez_compressed(artifact, input=x, head=head, valid=valid, target=target, fp32_mask=fp32_mask)
            record = {**case, "file": artifact.name, "sha256": digest(artifact),
                      "image_sha256": digest(images[name]), "target_sha256": digest(targets[name]),
                      "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                      "head_sha256": hashlib.sha256(head.tobytes()).hexdigest(), "geometry": meta,
                      "fp32": segmentation_metrics(fp32_mask, target, valid),
                      "int8": segmentation_metrics(head[0, 0]>=model.threshold["threshold_q"], target, valid),
                      "seconds": {"decode": decode_s, "preprocess": prep_s, "fp32": fp32_s, "int8": int8_s},
                      "threshold_metrics": {str(t["threshold_probability"]): {
                          "fp32": segmentation_metrics(floating>=t["threshold_logit"], target, valid),
                          "int8": segmentation_metrics(head[0, 0]>=t["threshold_q"], target, valid)} for t in thresholds}}
            if a_records and name in a_records["int8", case["source"]]:
                a_compared += 1
                for j, t in enumerate(thresholds):
                    for kind in ("fp32", "int8"):
                        oracle = a_records[kind, case["source"]][name]
                        for metric in ("dice", "iou"):
                            delta = abs(record["threshold_metrics"][str(t["threshold_probability"])][kind][metric]-oracle[metric][j])
                            max_delta[kind] = max(max_delta[kind], delta)
                            if delta > 1e-7 and kind == "int8":
                                raise ValueError(f"A metric mismatch {name}/{kind}/{metric}/p={t['threshold_probability']}: delta={delta}")
            report["records"].append(record)
            (args.output/"progress.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
            print(f"REFERENCE {index+1}/{len(cases)} {name} int8={int8_s:.3f}s dice={record['int8']['dice']:.6f}", flush=True)
        report["complete_test_set"] = len(cases) == selected["count"]
        report["primary_evaluation_complete"] = report["complete_test_set"] and selected["primary"]
        report["a_metrics"] = {"compared_images": a_compared, "max_abs_difference": max_delta,
                               "int8_matched": a_compared > 0 and max_delta["int8"] <= 1e-7,
                               "fp32_matched": a_compared > 0 and max_delta["fp32"] <= 1e-7,
                               "fp32_scope": "independent CPU reconstruction; differences from A are retained, not called exact reproduction"}
        report["means"] = {kind: {metric: float(np.mean([r[kind][metric] for r in report["records"]]))
                                 for metric in ("dice", "iou")} for kind in ("fp32", "int8")}
        report["groups"] = {}
        for field in ("source", "band"):
            for value in sorted({r[field] for r in report["records"] if field in r}):
                rows = [r for r in report["records"] if r.get(field)==value]
                report["groups"][field+":"+value] = {"count": len(rows), **{kind: {
                    metric: float(np.mean([r[kind][metric] for r in rows])) for metric in ("dice", "iou")}
                    for kind in ("fp32", "int8")}}
        report["passed"] = True
        validate_manifest(report, model)
    except Exception as exc:
        report["passed"] = False
        report["error"] = str(exc)
        raise
    finally:
        (args.output/"manifest.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("passed", "complete_test_set", "primary_evaluation_complete", "means")}), flush=True)


if __name__ == "__main__":
    main()
