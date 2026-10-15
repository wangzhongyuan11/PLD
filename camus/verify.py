"""Explicit package/full-network/native replay verification for CAMUS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .device import EmulatedTileAccelerator
from .graph import CamusGraph
from .model import DEFAULT_MODEL_ROOT, load_model, load_golden
from .postprocess import apply_island5_hole1
from .reference import ReferenceNetwork
from .result import native_labels_from_head


def _golden_rows(model, captures) -> list[dict[str, object]]:
    rows = []
    for layer in model.layers:
        for kind, suffix in (("input_full", "input"), ("output_full_golden", "output")):
            golden = load_golden(model, layer, kind)
            actual = captures[f"{layer.index:02d}_{layer.name}_{suffix}"]
            equal = bool(np.array_equal(actual, golden))
            mismatch = int(np.size(golden) - np.count_nonzero(actual == golden)) if golden.shape == actual.shape else int(np.size(golden))
            first = None
            if not equal:
                if golden.shape == actual.shape:
                    flat_golden = golden.reshape(-1)
                    flat_actual = actual.reshape(-1)
                    index = int(np.flatnonzero(flat_golden != flat_actual)[0])
                    first = {"flat_index": index, "expected": int(flat_golden[index]), "actual": int(flat_actual[index])}
                else:
                    first = {"expected_shape": list(golden.shape), "actual_shape": list(actual.shape)}
            rows.append({"layer": layer.name, "tensor": suffix, "equal": equal, "mismatches": mismatch, "first_mismatch": first})
    return rows


def verify_full_network(model) -> dict[str, object]:
    first_input = load_golden(model, model.layers[0], "input_full")
    reference = ReferenceNetwork(model).forward(first_input, capture=True)
    ref_rows = _golden_rows(model, reference["captures"])
    device = EmulatedTileAccelerator()
    emulated = CamusGraph(model, device).forward(first_input, capture=True)
    emu_rows = _golden_rows(model, emulated["captures"])
    return {
        "first_input_sha256_model_manifest": "verified_by_load_model",
        "reference_all_equal": all(row["equal"] for row in ref_rows),
        "reference_tensors": ref_rows,
        "emulator_all_equal": all(row["equal"] for row in emu_rows),
        "emulator_tensors": emu_rows,
        "emulator_conv_records": emulated["conv_records"],
        "reference_head_equals_golden": bool(np.array_equal(reference["head"], load_golden(model, model.layers[-1], "output_full_golden"))),
        "emulator_head_equals_reference": bool(np.array_equal(emulated["head"], reference["head"])),
    }


def verify_native_replay(model) -> dict[str, object]:
    case_root = model.root.parent / "evaluations" / "int8" / "cases"
    inputs = sorted(case_root.glob("*_int8.npz"))
    if len(inputs) != 200:
        raise ValueError(f'expected 200 exported dev cases, found {len(inputs)} in {case_root}')
    rows = []
    processed_rows = []
    for path in inputs:
        with np.load(path, allow_pickle=False) as payload:
            logits = np.asarray(payload["logits_int8"])
            expected_raw = np.asarray(payload["labels"])
            scale = float(payload["head_output_scale"])
        if scale != model.head_scale:
            raise ValueError(f"head scale differs in {path.name}")
        labels = native_labels_from_head(logits, expected_raw.shape[0], expected_raw.shape[1], scale)
        raw_equal = bool(np.array_equal(labels, expected_raw))
        rows.append({"case": path.name, "equal": raw_equal, "pixels": int(labels.size)})
        processed_path = path.with_name(path.name[: -len("_int8.npz")] + "_processed.npz")
        with np.load(processed_path, allow_pickle=False) as payload:
            expected_processed = np.asarray(payload["labels"])
        processed = apply_island5_hole1(labels)
        processed_equal = bool(np.array_equal(processed.labels, expected_processed))
        processed_rows.append({"case": path.name, "equal": processed_equal, "pixels": int(labels.size)})
    return {
        "scope": "replay of exported int8 head tensors; not new image inference or PL execution",
        "case_count": len(inputs),
        "native_raw_all_equal": all(row["equal"] for row in rows),
        "native_raw_cases": rows,
        "postprocess_all_equal": all(row["equal"] for row in processed_rows),
        "postprocess_cases": processed_rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m camus.verify")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output", type=Path, default=Path("out/camus/verify.json"))
    parser.add_argument("--skip-full-network", action="store_true")
    parser.add_argument("--skip-native-replay", action="store_true")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    report = {
        "schema": "pld-camus-stage1-verify-v1",
        "evidence_scope": "software package/reference/emulator/replay only; no RTL build or board execution",
    }
    model_load_started = time.perf_counter()
    model = load_model(args.model)
    from .pipeline import protect_output
    protect_output(args.output, model.root)
    report["package"] = {
        "manifest_members": 173,
        "spec_sha256": model.model_sha256,
        "model_load_verify_seconds": time.perf_counter() - model_load_started,
    }
    if args.skip_full_network:
        report["full_network"] = {"skipped": True}
    else:
        full_started = time.perf_counter()
        report["full_network"] = verify_full_network(model)
        report["full_network"]["seconds"] = time.perf_counter() - full_started
    if args.skip_native_replay:
        report["native_replay"] = {"skipped": True}
    else:
        replay_started = time.perf_counter()
        report["native_replay"] = verify_native_replay(model)
        report["native_replay"]["seconds"] = time.perf_counter() - replay_started
    report["total_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    passed = (
        report["package"]["manifest_members"] == 173
        and (args.skip_full_network or (
            report["full_network"]["reference_all_equal"]
            and report["full_network"]["emulator_all_equal"]
            and report["full_network"]["reference_head_equals_golden"]
            and report["full_network"]["emulator_head_equals_reference"]
        ))
        and (args.skip_native_replay or (
            report["native_replay"]["native_raw_all_equal"]
            and report["native_replay"]["postprocess_all_equal"]
        ))
    )
    print(json.dumps({
        "ok": passed,
        "output": args.output.as_posix(),
        "case_count": 0 if args.skip_native_replay else report["native_replay"]["case_count"],
        "total_seconds": report["total_seconds"],
    }))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
