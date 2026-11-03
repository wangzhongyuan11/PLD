"""Evaluate a frozen FP32/INT8 pair on the three development sets only.

Per-source records are cached by checkpoint/spec hash so an interrupted INT8
evaluation can resume without revisiting already completed sources. The 2018
final holdout is never opened here.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch

from software.role_a import quant, train_overall_v1 as overall


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as stream:
        tmp = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(tmp, path)


def source_records(model: torch.nn.Module, name: str, dev: overall.DevSet,
                   cache: Path, identity: dict[str, str], device: torch.device,
                   batch_size: int) -> list[dict]:
    if cache.is_file():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if payload.get("identity") != identity or payload.get("source") != name:
            raise ValueError(f"Cached evaluation differs from current checkpoint/spec: {cache}")
        rows = payload["records"]
        if len(rows) != len(dev.items):
            raise ValueError(f"Cached count differs from development set: {cache}")
        print(f"reused {cache.name}: {len(rows)} images", flush=True)
        return rows
    rows = overall.evaluate_development(model, {name: dev}, size=256,
                                        batch_size=batch_size, num_workers=0,
                                        device=device)
    atomic_json(cache, {"identity": identity, "source": name, "records": rows})
    print(f"evaluated {cache.name}: {len(rows)} images", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoint_hash = overall.sha256_file(args.checkpoint)
    spec_hash = overall.sha256_file(args.spec)
    fp32, config = quant._load_fp32(args.checkpoint)
    int8, spec = quant._load_quantized(args.spec)
    if spec["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("INT8 export and FP32 checkpoint differ")
    if int(config["size"]) != int(spec["size"]):
        raise ValueError("FP32 and INT8 input sizes differ")
    assembly = overall.assemble(overall.build_default_spec())
    if len(assembly.train_items) != 2309 or sum(len(x.items) for x in assembly.dev_sets.values()) != 638:
        raise ValueError("Frozen training/development counts changed")
    identity = {"checkpoint_sha256": checkpoint_hash, "spec_sha256": spec_hash,
                "development_ids_sha256": assembly.split_digest["development_all"],
                "evaluator_sha256": overall.sha256_file(Path(__file__))}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp32 = fp32.to(device).eval()
    int8 = int8.to("cpu").eval()
    fp_rows: list[dict] = []
    q_rows: list[dict] = []
    # Finish the smaller sources first, so an interruption still leaves useful
    # cached evidence and only the unfinished source has to be repeated.
    for name, dev in sorted(assembly.dev_sets.items(), key=lambda pair: len(pair[1].items)):
        fp_rows.extend(source_records(fp32, name, dev, args.output_dir / f"fp32_{name}.json",
                                      identity, device, batch_size=4))
        q_rows.extend(source_records(int8, name, dev, args.output_dir / f"int8_{name}.json",
                                     identity, torch.device("cpu"), batch_size=1))
    fp_report = overall.build_development_report(fp_rows, assembly.dev_sets)
    q_report = overall.build_development_report(q_rows, assembly.dev_sets)
    selected = float(fp_report["selected_threshold"])
    selected_q_score = q_report["threshold_table"][selected]["dice"]
    stored = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "selected_threshold" in stored and float(stored["selected_threshold"]) != selected:
        raise ValueError("Recomputed FP32 threshold disagrees with checkpoint")
    if "primary_dice" in stored and abs(float(stored["primary_dice"]) - fp_report["primary"]["dice"]) > 1e-6:
        raise ValueError("Recomputed FP32 Dice disagrees with checkpoint")
    result = {
        "schema": "overall-int8-development-v1", "identity": identity,
        "fp32": fp_report, "int8": q_report,
        "int8_at_fp32_threshold": {"threshold": selected, "dice": selected_q_score,
                                    "delta": selected_q_score - fp_report["primary"]["dice"]},
        "int8_with_development_selected_threshold": {
            "threshold": q_report["selected_threshold"], "dice": q_report["primary"]["dice"],
            "delta": q_report["primary"]["dice"] - fp_report["primary"]["dice"]},
        "holdout": "ISIC 2018 Task 1 masks have not been evaluated",
    }
    atomic_json(args.output_dir / "comparison.json", result)
    print(f"FP32={fp_report['primary']['dice']:.6f}@{selected:.2f} "
          f"INT8={q_report['primary']['dice']:.6f}@{q_report['selected_threshold']:.2f}", flush=True)


if __name__ == "__main__":
    main()
