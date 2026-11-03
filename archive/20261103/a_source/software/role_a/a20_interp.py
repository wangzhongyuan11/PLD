"""A20 阶段 1：两份源权重的固定比例插值（只做插值 + 硬校验，不做评估）。

方案复用 A14-S 已验收的 soup 方法（同结构权重的线性平均，float64 中间精度、
按 base 的 dtype 落盘、alpha=0/1 端点逐位复现），源只有两份：

* base  = ``runs/role_a_a14_soup/candidates/soup_v5_a050.pt``（SHA256 ``f488719f…67eb77``）
* other = ``runs/role_a_a19_soup_dice3_lr5e5/best.pt`` （SHA256 ``57b78cfc…5e1bf4b53``）

定义：``state = (1 - alpha) * base + alpha * A19``，即 **alpha = A19 权重占比**。
预置三点 alpha = 0.25 / 0.50 / 0.75，不动态扩网格。

本阶段**不读开发集、不做 FP32 638 评估、不导出 INT8、不下载**。
开发集 638 的 FP32 五阈值筛选与三候选 INT8 验收留待下一阶段放行后执行；
届时按同一五阈值在 dev638 筛选，并对三个候选各自做 INT8 导出/验证/评估，
不得仅凭 FP32 提前排除候选。

用法：
    python -m software.role_a.a20_interp --verify-only     # 只做硬校验，不写产物
    python -m software.role_a.a20_interp --output-dir runs/role_a_a20_interp
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from software.role_a.model import build_model

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "runs" / "role_a_a20_interp"

SOURCES: dict[str, tuple[Path, str]] = {
    "soup": (PROJECT / "runs" / "role_a_a14_soup" / "candidates" / "soup_v5_a050.pt",
             "f488719fe6d4d09c9b86be84d06b19d34241d6c14060b87da7bab25d1367eb77"),
    "a19": (PROJECT / "runs" / "role_a_a19_soup_dice3_lr5e5" / "best.pt",
            "57b78cfc6df14b6f2fb53aef42c617836fe6b22c394960a0bcaf95e5e1bf4b53"),
}
BASE = "soup"
OTHER = "a19"
ALPHAS: tuple[float, ...] = (0.25, 0.50, 0.75)
EXPECTED_STRUCTURE = (8, 3, 256)          # base_channels, num_pools, size
CANDIDATE_SCHEMA = "a20-interp-candidate-v1"
SUMMARY_SCHEMA = "a20-interp-stage1-v1"
# 下一阶段冻结口径（本阶段不执行），写进摘要以免放行时口径漂移。
SELECTION_THRESHOLDS: tuple[float, ...] = (0.45, 0.50, 0.53, 0.56, 0.60)
PRIMARY_METRIC = "development_638_mean_per_image_dice"
SELECTION_RULE = ("主指标 = 三套开发集共 638 张的逐图 Dice 算术平均；只在五个统一阈值中取主指标最高者；"
                  "并列（1e-6 内）先比三来源 Dice 均值，再取更小阈值")
CANDIDATE_NAMES = tuple(f"interp_a19_a{int(alpha * 100):03d}" for alpha in ALPHAS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(name: str) -> dict[str, Any]:
    """硬校验完整 SHA256 后加载；不符即抛错，不做任何降级。"""
    path, expected = SOURCES[name]
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"[A20] 源权重 {name} SHA256 不符: 实际 {actual} != 期望 {expected}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise SystemExit(f"[A20] 源权重 {name} 缺少 model_state")
    return {"name": name, "path": path, "sha256": actual, "payload": payload,
            "state": payload["model_state"], "config": dict(payload.get("config", {}))}


def check_sources_finite(loaded: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """逐源、逐张量检查有限性：结论必须覆盖两份源，而不是只查 base。"""
    checked: dict[str, int] = {}
    non_finite: dict[str, list[str]] = {}
    for name, entry in loaded.items():
        bad = [key for key, value in entry["state"].items()
               if not bool(torch.isfinite(value).all())]
        if bad:
            non_finite[name] = bad
        checked[name] = len(entry["state"])
    if non_finite:
        raise SystemExit("[A20] 源权重含非有限张量: "
                         + json.dumps({name: keys[:3] for name, keys in non_finite.items()},
                                      ensure_ascii=False))
    return checked


def check_structure(loaded: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """同 8 通道、同键、同 shape、无 BatchNorm、两源权重全有限，且能严格加载。"""
    model = build_model(base_channels=EXPECTED_STRUCTURE[0], num_pools=EXPECTED_STRUCTURE[1],
                        input_shape=(3, EXPECTED_STRUCTURE[2], EXPECTED_STRUCTURE[2]))
    batchnorm = [name for name, module in model.named_modules()
                 if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    if batchnorm:
        raise SystemExit(f"[A20] 模型含 BatchNorm，权重插值不适用: {batchnorm}")
    reference_keys = set(model.state_dict())
    base_state = loaded[BASE]["state"]
    for name, entry in loaded.items():
        config = entry["config"]
        triple = (int(config.get("base_channels", -1)), int(config.get("num_pools", -1)),
                  int(config.get("size", -1)))
        if triple != EXPECTED_STRUCTURE:
            raise SystemExit(f"[A20] {name} 结构配置异常: {triple} != {EXPECTED_STRUCTURE}")
        keys = set(entry["state"])
        if keys != reference_keys:
            raise SystemExit(f"[A20] {name} 键与模型不一致: 多 {sorted(keys - reference_keys)[:3]} "
                             f"少 {sorted(reference_keys - keys)[:3]}")
        for key, value in entry["state"].items():
            if tuple(value.shape) != tuple(base_state[key].shape):
                raise SystemExit(f"[A20] {name} 键 {key} shape 不一致: {tuple(value.shape)} != "
                                 f"{tuple(base_state[key].shape)}")
        model.load_state_dict(entry["state"], strict=True)
    checked_tensors = check_sources_finite(loaded)
    return {"state_keys": len(reference_keys),
            "parameters": int(sum(v.numel() for v in model.state_dict().values())),
            "base_channels": EXPECTED_STRUCTURE[0], "num_pools": EXPECTED_STRUCTURE[1],
            "size": EXPECTED_STRUCTURE[2], "batchnorm_modules": 0, "batchnorm_absent": True,
            "strict_load_ok": True, "weights_finite": True,
            "finite_checked_per_source": checked_tensors,
            "finite_tensors_total": int(sum(checked_tensors.values())),
            "key_sets_identical": True, "shapes_identical": True}


def blend(base_state: Mapping[str, torch.Tensor], other_state: Mapping[str, torch.Tensor],
          alpha: float) -> dict[str, torch.Tensor]:
    """(1-alpha)*base + alpha*other，float64 中间精度，落回 base 的 dtype。

    与 A14-S 的 ``blend`` 同法；额外要求两态键集相同，避免静默丢键。
    """
    if set(base_state) != set(other_state):
        raise ValueError("blend 需要两份 state 的键集完全相同")
    alpha = float(alpha)
    blended: dict[str, torch.Tensor] = {}
    for key, value in base_state.items():
        other = other_state[key]
        if tuple(value.shape) != tuple(other.shape):
            raise ValueError(f"blend 键 {key} shape 不一致: {tuple(value.shape)} != {tuple(other.shape)}")
        mixed = (1.0 - alpha) * value.to(torch.float64) + alpha * other.to(torch.float64)
        blended[key] = mixed.to(value.dtype)
    return blended


def check_endpoints(base_state: Mapping[str, torch.Tensor],
                    other_state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """端点必须逐位一致：blend(base, other, 0) -> base，blend(base, other, 1) -> other。

    两次都用**真实的双源输入**（而不是把 base 同时当两源），因此也顺带验证
    ``blend`` 在 alpha 取端点时不会因另一源引入任何扰动。
    """
    zero = blend(base_state, other_state, 0.0)
    mismatch = [key for key, value in zero.items() if not torch.equal(value, base_state[key])]
    if mismatch:
        raise SystemExit(f"[A20] alpha=0 端点未能逐位复现 {BASE}: {mismatch[:3]}")
    one = blend(base_state, other_state, 1.0)
    mismatch = [key for key, value in one.items() if not torch.equal(value, other_state[key])]
    if mismatch:
        raise SystemExit(f"[A20] alpha=1 端点未能逐位复现 {OTHER}: {mismatch[:3]}")
    return {"alpha_0_bitwise_equals_base": True, "alpha_1_bitwise_equals_other": True,
            "endpoints_use_both_sources": True, "checked_tensors": len(base_state)}


def build_candidate(loaded: Mapping[str, Mapping[str, Any]], alpha: float,
                    output_dir: Path) -> dict[str, Any]:
    label = f"interp_a19_a{int(round(alpha * 100)):03d}"
    state = blend(loaded[BASE]["state"], loaded[OTHER]["state"], alpha)
    path = output_dir / "candidates" / f"{label}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": CANDIDATE_SCHEMA,
                "interp": {"base": BASE, "base_sha256": loaded[BASE]["sha256"],
                           "other": OTHER, "other_sha256": loaded[OTHER]["sha256"],
                           "alpha_a19_share": float(alpha),
                           "formula": "state = (1 - alpha) * base + alpha * a19 (float64, cast to base dtype)"},
                "config": loaded[BASE]["config"], "model_state": state}, path)
    return {"label": label, "alpha_a19_share": float(alpha), "checkpoint": path.as_posix(),
            "checkpoint_sha256": sha256(path), "checkpoint_bytes": path.stat().st_size,
            "tensors": len(state)}


def check_candidate_compatibility(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """只读兼容冒烟：用下一阶段现成导出器 ``quant._load_fp32`` 读候选。

    只做 schema/config/键集的兼容性检查，**不校准、不前向、不写任何产物**。
    任何不兼容都以非零退出结束，避免下一阶段才发现候选读不进导出器。
    """
    from software.role_a import quant

    path = Path(candidate["checkpoint"])
    model, config = quant._load_fp32(path)
    keys = list(model.state_dict())
    if any(not bool(parameter.isfinite().all()) for parameter in model.state_dict().values()):
        raise SystemExit(f"[A20] 候选 {candidate['label']} 含非有限权重")
    return {"label": candidate["label"], "path": path.as_posix(),
            "readable_by_quant_load_fp32": True, "strict_load_ok": True,
            "config_triple": [int(config["base_channels"]), int(config["num_pools"]), int(config["size"])],
            "state_keys": len(keys), "weights_finite": True,
            "calibration_or_forward_run": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--verify-only", action="store_true",
                        help="只做源哈希/结构/端点硬校验，不写任何候选或摘要")
    args = parser.parse_args(list(argv) if argv is not None else None)

    loaded = {name: load_source(name) for name in SOURCES}
    structure = check_structure(loaded)
    endpoints = check_endpoints(loaded[BASE]["state"], loaded[OTHER]["state"])
    print(json.dumps({"sources": {k: {"path": v["path"].as_posix(), "sha256": v["sha256"]}
                                  for k, v in loaded.items()},
                      "structure": structure, "endpoints": endpoints},
                     ensure_ascii=False, indent=2), flush=True)
    if args.verify_only:
        print("[A20] verify-only 通过：源哈希、结构、端点全部符合；未写任何产物", flush=True)
        return 0

    output_dir = Path(args.output_dir)
    candidates = [build_candidate(loaded, alpha, output_dir) for alpha in ALPHAS]
    for candidate in candidates:
        print(f"[A20] built {candidate['label']} sha256={candidate['checkpoint_sha256'][:16]}... "
              f"bytes={candidate['checkpoint_bytes']}", flush=True)

    # 只读兼容冒烟：确认下一阶段的现成导出器能读候选（不校准、不前向、不写产物）。
    compatibility = [check_candidate_compatibility(candidate) for candidate in candidates]
    for entry in compatibility:
        print(f"[A20] quant._load_fp32 读入 {entry['label']} 通过: 键 {entry['state_keys']} "
              f"config {entry['config_triple']}", flush=True)

    summary = {
        "schema": SUMMARY_SCHEMA,
        "stage": "1 (interpolation only; no development-set evaluation, no INT8, no download)",
        "method": "A14-S soup 线性插值（复用同一 float64 中间精度与端点逐位复现校验）",
        "formula": "state = (1 - alpha) * base + alpha * a19，alpha = A19 权重占比",
        "sources": {k: {"path": v["path"].as_posix(), "sha256": v["sha256"]}
                    for k, v in loaded.items()},
        "structure": structure,
        "endpoints": endpoints,
        "alphas": list(ALPHAS),
        "grid_policy": "预置三点，不动态扩网格",
        "candidates": candidates,
        "candidate_compatibility_smoke": compatibility,
        "frozen_protocol_for_next_stage": {
            "development_ids": "冻结 2309/638 划分（复用 overall v1 固定划分）",
            "thresholds": list(SELECTION_THRESHOLDS),
            "primary_metric": PRIMARY_METRIC,
            "selection_rule": SELECTION_RULE,
            "quantization": "三候选各自 --overall-train 96 张校准导出 + verify_export 15 层 + dev638 evaluate_quant_overall",
            "rule": "不得仅凭 FP32 提前排除候选；三候选都要做 INT8",
        },
        "holdout": "ISIC 2018 Task 1 masks have not been evaluated",
        "budget": {"cpu_only": True, "downloads_bytes": 0, "training": False,
                   "dev638_evaluations": 0, "int8_exports": 0},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidates": [c["label"] for c in candidates],
                      "alphas": list(ALPHAS)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
