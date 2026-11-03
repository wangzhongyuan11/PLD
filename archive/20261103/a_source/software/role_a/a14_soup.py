"""A14 阶段 S：v4 / v5 / E1 三个同结构权重的线性平均（model soup）筛选。

只做：权重平均 -> 严格一致性校验 -> 638 张 FP32 五阈值评估。
不做训练，不改旧权重；INT8 由外层按阈值条件另行调用现有管线。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from software.role_a import train_overall_v1 as overall
from software.role_a.model import build_model

OUT = Path("runs/role_a_a14_soup")
SOURCES = {
    "v4": (Path("runs/role_a_overall_v4_dice2_base8/best.pt"),
           "65bb913162812ae602aa6ce0f6cda183eb9f8e706af3a1b15a572e7159151b63"),
    "v5": (Path("runs/role_a_overall_v5_rotcos8/best.pt"),
           "0e6e96454b5bd1edf4278c67c1e9461b61466adc24e06a931fe05919942cbf58"),
    "e1": (Path("runs/role_a_a13_rot_only8/best.pt"),
           "11a280590743dac388882ac0a49a541fe1194a988d5c30da6629dc3674e2406a"),
}
BASE = "v4"
ALPHAS = (0.25, 0.50, 0.75)
CANDIDATES = [(f"soup_{name}_a{int(alpha * 100):03d}", name, alpha)
              for name in ("v5", "e1") for alpha in ALPHAS]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name: str) -> dict[str, Any]:
    path, expected = SOURCES[name]
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"{name} 源权重 SHA256 不符: {actual} != {expected}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise SystemExit(f"{name} 缺少 model_state")
    return {"path": path, "sha256": actual, "payload": payload,
            "state": payload["model_state"], "config": payload.get("config", {})}


def check_architecture(loaded: dict[str, Any]) -> dict[str, Any]:
    model = build_model(base_channels=8, num_pools=3, input_shape=(3, 256, 256))
    batchnorm = [name for name, module in model.named_modules()
                 if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    if batchnorm:
        raise SystemExit(f"模型含 BatchNorm，权重平均不适用: {batchnorm}")
    state_keys = set(model.state_dict())
    for name in ("v4", "v5", "e1"):
        keys = set(loaded[name]["state"])
        if keys != state_keys:
            raise SystemExit(f"{name} 键不一致: 多 {sorted(keys - state_keys)[:3]} "
                             f"少 {sorted(state_keys - keys)[:3]}")
    for name in ("v5", "e1"):
        for key, value in loaded[name]["state"].items():
            if tuple(value.shape) != tuple(loaded["v4"]["state"][key].shape):
                raise SystemExit(f"{key} shape 不一致: {name}={tuple(value.shape)} "
                                 f"v4={tuple(loaded['v4']['state'][key].shape)}")
    for name in ("v4", "v5", "e1"):
        config = loaded[name]["config"]
        triple = (int(config.get("base_channels", -1)), int(config.get("num_pools", -1)),
                  int(config.get("size", -1)))
        if triple != (8, 3, 256):
            raise SystemExit(f"{name} 结构配置异常: {triple}")
    return {"state_keys": len(state_keys),
            "parameters": int(sum(v.numel() for v in model.state_dict().values())),
            "batchnorm_modules": 0,
            "batchnorm_absent": True}


def blend(base_state, other_state, alpha: float) -> dict[str, torch.Tensor]:
    alpha = float(alpha)
    return {key: (1.0 - alpha) * base_state[key].to(torch.float64)
            + alpha * other_state[key].to(torch.float64) for key in base_state}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    out = Path(args.output_dir)
    (out / "candidates").mkdir(parents=True, exist_ok=True)

    loaded = {name: load(name) for name in SOURCES}
    info = check_architecture(loaded)
    print(json.dumps({"architecture": info,
                      "source_sha256": {k: v["sha256"] for k, v in loaded.items()}},
                     ensure_ascii=False, indent=2), flush=True)

    base_state = {k: v.to(torch.float64) for k, v in loaded["v4"]["state"].items()}
    # 端点复现：alpha=0 必须逐位等于 v4
    zero = blend(loaded["v4"]["state"], loaded["v4"]["state"], 0.0)
    for key, value in zero.items():
        if not torch.equal(value.to(loaded["v4"]["state"][key].dtype),
                           loaded["v4"]["state"][key]):
            raise SystemExit(f"alpha=0 端点未能复现 v4: {key}")
    for name in ("v5", "e1"):
        one = blend(loaded["v4"]["state"], loaded[name]["state"], 1.0)
        for key, value in one.items():
            if not torch.equal(value.to(loaded[name]["state"][key].dtype),
                               loaded[name]["state"][key]):
                raise SystemExit(f"alpha=1 端点未能复现 {name}: {key}")
    print("端点复现通过: alpha=0 -> v4 逐位一致；alpha=1 -> 源模型逐位一致", flush=True)

    candidates: dict[str, dict[str, Any]] = {}
    for label, name, alpha in CANDIDATES:
        state = blend(loaded["v4"]["state"], loaded[name]["state"], alpha)
        state = {key: value.to(loaded["v4"]["state"][key].dtype) for key, value in state.items()}
        path = out / "candidates" / f"{label}.pt"
        torch.save({"schema": "a14-soup-candidate-v1", "soup": {"base": BASE, "other": name,
                    "alpha": alpha, "base_sha256": loaded["v4"]["sha256"],
                    "other_sha256": loaded[name]["sha256"]},
                    "config": loaded["v4"]["config"], "model_state": state}, path)
        candidates[label] = {"label": label, "other": name, "alpha": alpha,
                             "checkpoint": path.as_posix(),
                             "checkpoint_sha256": sha256(path)}
        print(f"built {label} sha256={candidates[label]['checkpoint_sha256'][:16]}...", flush=True)

    assembly = overall.assemble(overall.build_default_spec())
    if sum(len(x.items) for x in assembly.dev_sets.values()) != 638:
        raise SystemExit("冻结开发集数量变化")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v4_state = loaded["v4"]["state"]
    for label, block in candidates.items():
        model = build_model(base_channels=8, num_pools=3, input_shape=(3, 256, 256))
        payload = torch.load(block["checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model_state"], strict=True)
        model = model.to(device).eval()
        records = overall.evaluate_development(model, assembly.dev_sets, size=256, batch_size=4,
                                               num_workers=0, thresholds=overall.THRESHOLDS,
                                               device=device)
        report = overall.build_development_report(records, assembly.dev_sets,
                                                  overall.THRESHOLDS)
        block["fp32"] = {
            "primary_dice": float(report["primary"]["dice"]),
            "primary_iou": float(report["primary"]["iou"]),
            "selected_threshold": float(report["selected_threshold"]),
            "threshold_dice": {str(t): float(report["threshold_table"][t]["dice"])
                               for t in overall.THRESHOLDS},
            "source_dice": {k: float(v["all"]["dice"]) for k, v in report["sources"].items()},
            "area_groups": {k: {"n": int(v["n"]), "dice": v["dice"]}
                            for k, v in report["area_groups_pooled"].items()},
        }
        print(f"{label} FP32 dice={block['fp32']['primary_dice']:.10f} "
              f"thr={block['fp32']['selected_threshold']:.2f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    v4_fp32 = 0.8431070097803244
    ranked = sorted(candidates.values(), key=lambda b: -b["fp32"]["primary_dice"])
    best = ranked[0]
    within = best["fp32"]["primary_dice"] >= v4_fp32 - 0.001
    above = best["fp32"]["primary_dice"] > v4_fp32
    quantise = [b["label"] for b in ranked[:2]] if (above or within) else []
    summary = {
        "schema": "a14-soup-v1",
        "sources": {k: {"path": v["path"].as_posix(), "sha256": v["sha256"]}
                    for k, v in loaded.items()},
        "architecture": info,
        "v4_fp32_reference": v4_fp32,
        "candidates": candidates,
        "ranking_fp32": [b["label"] for b in ranked],
        "best_fp32": {"label": best["label"], "dice": best["fp32"]["primary_dice"],
                      "above_v4": bool(above), "within_0.001_of_v4": bool(within)},
        "quantise_candidates": quantise,
        "quantise_decision_rule": "最好 FP32 > v4 或距 v4 <= 0.001 才对最多 2 个候选做 INT8；最终只按 INT8 总体 Dice 判定",
        "holdout": "ISIC 2018 Task 1 masks have not been evaluated",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps({"ranking": summary["ranking_fp32"], "best": summary["best_fp32"],
                      "quantise": quantise}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())