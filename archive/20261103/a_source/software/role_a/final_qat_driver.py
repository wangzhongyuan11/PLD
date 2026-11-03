"""A 组最终训练驱动：从 soup_v5_a050 起步的 8 轮联合（输入/权重/输出）量化感知微调。

固定口径：train 2309 / dev 638 / 阈值 0.45,0.50,0.53,0.56,0.60 / 主指标为 INT8 逐图平均 Dice。
每轮流程：训练 → 原子写 latest.pt(phase=trained，含优化器/尺度/RNG) → 保存该轮候选（unwrap）
→ INT8 导出 → verify_export（须 15/15，失败即硬拒绝）→ 同一 dev638 `evaluate_quant_overall`
→ 写 latest.pt(phase=evaluated) + 该轮结果。**任一段非 0 均记录退出码并硬拒绝继续。**

resume：
* 先硬校验源码/配置/输入哈希；再从断点恢复**保存的尺度**并校验其自身摘要（不与"初始尺度"强比），
  因此第 4 轮后重校准过的断点也能恢复；
* 断点 epoch=N 且 phase=evaluated ⇒ 从 N+1 轮继续；phase=trained ⇒ 只补做第 N 轮的导出与评估，
  不重复训练该轮；
* 恢复 Python/NumPy/Torch/CUDA RNG 与初始 seed。

不读取、不评分 ISIC 2018 最终测试掩膜；不修改现有 deliverables。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from software.role_a import evaluate_quant_overall
from software.role_a import final_guard as guard
from software.role_a import qat
from software.role_a import quant
from software.role_a import train_overall_v1 as overall
from software.role_a import verify_export
from software.role_a.metrics import segmentation_loss

PROJECT = Path(__file__).resolve().parents[2]
BASELINE_SOUP = PROJECT / "runs" / "role_a_a14_soup" / "candidates" / "soup_v5_a050.pt"
BASELINE_SOUP_SHA256 = "f488719fe6d4d09c9b86be84d06b19d34241d6c14060b87da7bab25d1367eb77"
BASELINE_INT8_DICE = 0.8420508068333175
SCHEMA = "a-final-qat-checkpoint-v1"
CANDIDATE_SCHEMA = "a-final-qat-candidate-v1"
RECALIBRATE_AFTER_EPOCH = 4          # 第 4 轮后用同一 96 张无增强重算一次，5-8 轮冻结
DEFAULT_EPOCHS = 8
DEFAULT_LR = 3e-5
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEED = 20260926
CALIBRATION_IMAGES = 96
CODE_PATHS = {
    "driver": Path(__file__),
    "qat": PROJECT / "software" / "role_a" / "qat.py",
    "guard": PROJECT / "software" / "role_a" / "final_guard.py",
    "quant": PROJECT / "software" / "role_a" / "quant.py",
    "quant_math": PROJECT / "software" / "role_a" / "quant_math.py",
    "metrics": PROJECT / "software" / "role_a" / "metrics.py",
    "train_overall_v1": PROJECT / "software" / "role_a" / "train_overall_v1.py",
    "data": PROJECT / "software" / "role_a" / "data.py",
    "model": PROJECT / "software" / "role_a" / "model.py",
    "verify_export": PROJECT / "software" / "role_a" / "verify_export.py",
    "evaluate_quant_overall": PROJECT / "software" / "role_a" / "evaluate_quant_overall.py",
}


# ------------------------------------------------------------------ 基础工具
def _fsync_parent(path: Path) -> None:
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_torch_save(payload: Any, path: Path) -> None:
    """同目录临时文件 → flush+fsync → os.replace（原子落盘）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_parent(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def calibration_ids_digest(stems: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(stems).encode("utf-8")).hexdigest()


def scales_digest(scales: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(scales, sort_keys=True).encode()).hexdigest()


def build_config(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "a-final-qat-config-v1",
        "init": "checkpoint", "init_checkpoint": BASELINE_SOUP.as_posix(),
        "init_checkpoint_sha256": BASELINE_SOUP_SHA256,
        "base_channels": int(config["base_channels"]), "num_pools": int(config["num_pools"]),
        "size": int(config["size"]), "epochs": int(args.epochs),
        "batch_size": int(args.batch_size), "lr": float(args.lr),
        "lr_plan": "constant", "optimizer": "torch.optim.Adam",
        "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0,
        "grad_clip": float(args.grad_clip), "bce_weight": 1.0, "dice_weight": 3.0,
        "seed": int(args.seed), "num_workers": int(args.num_workers),
        "augment": True, "rotation": False, "device": str(args.device),
        "qat": "input+weight+output fake quant (STE), deployment-identical rules",
        "qat_activation_scales": "frozen from 96-image soup calibration; recalibrated once "
                                 f"after epoch {RECALIBRATE_AFTER_EPOCH} (no augmentation)",
        "calibration_images": CALIBRATION_IMAGES,
        "thresholds": list(overall.THRESHOLDS),
        "primary_metric": "development_638_mean_per_image_int8_dice",
        "baseline_soup_sha256": BASELINE_SOUP_SHA256,
        "baseline_int8_dice": BASELINE_INT8_DICE,
        "smoke": bool(args.smoke), "limit_batches": int(args.limit_batches or 0),
        "train_total": 2309, "development_total": 638,
    }


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def load_calibration_items(limit: int = CALIBRATION_IMAGES):
    assembly = quant.load_overall_assembly()
    grouped = quant.group_train_items(assembly)
    items, records = quant.select_calibration_items(grouped, limit)
    return assembly, items, records


def recompute_scales(model: torch.nn.Module, items: Sequence[Any], size: int) -> dict[str, Any]:
    dataset = quant.ISICDataset(list(items), root=None, size=size)
    stats = qat.calibration_scales(model, dataset, len(dataset))
    applied = qat.set_scales_from_calibration(model, stats)
    return {"stats": {name: stats[name] for name in stats}, "scales": applied,
            "source": "same 96 calibration images, no augmentation"}


def apply_saved_scales(model: torch.nn.Module, scales: dict[str, dict[str, float]]) -> None:
    """把断点里保存的尺度写回模型（不重算，避免与初始尺度强比）。"""
    stats = {name: {"input_max_abs": entry["input_scale"] * qat.QMAX,
                    "output_max_abs": entry["output_scale"] * qat.QMAX}
             for name, entry in scales.items()}
    applied = qat.set_scales_from_calibration(model, stats)
    if set(applied) != set(scales):
        raise SystemExit("[driver] 断点尺度层集合与模型不一致")


def resume_decision(payload: dict[str, Any]) -> tuple[int, str | None, bool]:
    """由断点决定恢复行为。

    返回 ``(start_epoch, resume_phase, need_training_of_start_epoch)``：
    * ``phase == "evaluated"``：该轮已完成 ⇒ 从 ``epoch+1`` 开始训练；
    * ``phase == "trained"``：该轮只差导出/评估 ⇒ 仍从该轮开始，但**不重复训练**。
    """
    epoch = int(payload["epoch"])
    phase = str(payload.get("phase"))
    if phase == "trained":
        return epoch, phase, False
    return epoch + 1, None, True


def run_development_evaluation(candidate: Path, spec_path: Path, dev_dir: Path) -> int:
    """调用既有 ``evaluate_quant_overall.main()`` 主流程（正常完成返回 0）。

    该 ``main()`` **返回 None**（其第 50 行只做 argparse），因此不能对其返回值取 int：
    正常完成视为 0；抛异常视为非零并把回溯写入 ``dev_dir/evaluate_error.log``。
    """
    saved = list(sys.argv)
    try:
        sys.argv = ["evaluate_quant_overall", "--checkpoint", str(candidate),
                    "--spec", str(spec_path), "--output-dir", str(dev_dir)]
        result = evaluate_quant_overall.main()
        if result not in (None, 0):
            return int(result)
        return 0
    except BaseException as error:          # noqa: BLE001 - 需要把任何失败都落盘
        import traceback

        dev_dir.mkdir(parents=True, exist_ok=True)
        (dev_dir / "evaluate_error.log").write_text(
            "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            encoding="utf-8")
        print(f"[driver] 评估异常: {type(error).__name__}: {error}", flush=True)
        return 1
    finally:
        sys.argv = saved


def export_and_evaluate(candidate: Path, export_dir: Path, dev_dir: Path,
                        output_dir: Path, epoch: int) -> dict[str, Any]:
    """导出 + 15 层验证 + 638 开发集评估；任一段非 0 即硬拒绝并记录退出码。"""
    entry: dict[str, Any] = {"epoch": epoch, "export_dir": export_dir.as_posix()}
    spec_path = export_dir / "model_spec.json"
    entry["export_exit"] = int(quant.command_export(argparse.Namespace(
        tile_size=8, calibration_images=CALIBRATION_IMAGES, overall_train=True,
        checkpoint=candidate, split=None, image_dir=None, mask_dir=None,
        output_dir=export_dir)))
    if entry["export_exit"] != 0:
        raise SystemExit(f"[driver] epoch {epoch} INT8 导出退出码 {entry['export_exit']}，硬拒绝继续")

    entry["verification"] = verify_export.verify(spec_path)
    entry["layers_verified"] = int(entry["verification"]["layers_verified"])
    atomic_json(entry["verification"], export_dir / "verification.json")
    entry["spec_sha256"] = guard.sha256_file(spec_path)
    if entry["layers_verified"] != 15:
        raise SystemExit(f"[driver] epoch {epoch} verify_export 只有 {entry['layers_verified']} 层，硬拒绝")
    evaluate_exit = run_development_evaluation(candidate, spec_path, dev_dir)
    entry["evaluate_exit"] = evaluate_exit
    if evaluate_exit != 0:
        raise SystemExit(f"[driver] epoch {epoch} 评估退出码 {evaluate_exit}，硬拒绝继续")

    comparison = json.loads((dev_dir / "comparison.json").read_text(encoding="utf-8"))
    entry["fp32_dice"] = comparison["fp32"]["primary"]["dice"]
    entry["fp32_threshold"] = comparison["fp32"]["selected_threshold"]
    entry["int8_dice"] = comparison["int8"]["primary"]["dice"]
    entry["int8_threshold"] = comparison["int8"]["selected_threshold"]
    entry["int8_areas"] = {k: v["dice"] for k, v in
                           comparison["int8"]["area_groups_pooled"].items()}
    entry["int8_sources"] = {k: v["all"]["dice"] for k, v in comparison["int8"]["sources"].items()}
    entry["quantization_loss"] = entry["int8_dice"] - entry["fp32_dice"]
    entry["int8_minus_baseline"] = entry["int8_dice"] - BASELINE_INT8_DICE
    entry["comparison_sha256"] = guard.sha256_file(dev_dir / "comparison.json")
    print(f"[driver] epoch {epoch} FP32={entry['fp32_dice']:.10f} INT8={entry['int8_dice']:.10f} "
          f"(vs baseline {entry['int8_minus_baseline']:+.10f})", flush=True)
    return entry


# ------------------------------------------------------------------------ 主流程
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT / "runs" / "role_a_final_qat8")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="烟测模式：允许 --limit-batches/--skip-eval")
    parser.add_argument("--limit-batches", type=int, default=0,
                        help="仅烟测：每轮最多跑多少 batch（0 = 全部）")
    parser.add_argument("--skip-eval", action="store_true",
                        help="仅烟测：跳过 INT8 导出与评估")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.smoke and (args.limit_batches or args.skip_eval):
        raise SystemExit("[driver] 正式训练不允许 limit-batches/skip-eval（禁止静默降级）")
    if args.smoke and args.epochs > DEFAULT_EPOCHS:
        raise SystemExit("[driver] 烟测不允许超过 8 轮")

    start = time.time()
    output_dir: Path = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "latest.pt"

    actual_init = guard.sha256_file(BASELINE_SOUP)
    if actual_init != BASELINE_SOUP_SHA256:
        raise SystemExit(f"[driver] 起点权重哈希不符: {actual_init} != {BASELINE_SOUP_SHA256}")

    assembly, calibration_items, _records = load_calibration_items()
    if int(assembly.counts["train_total"]) != 2309 or int(assembly.counts["development_total"]) != 638:
        raise SystemExit("[driver] 冻结划分不是 2309/638，拒绝运行")
    stems = [item.stem for item in calibration_items]
    if len(set(stems)) != len(stems) or len(stems) != CALIBRATION_IMAGES:
        raise SystemExit("[driver] 校准 ID 异常（重复或数量不符）")
    dev_ids = {stem for ids in assembly.dev_ids.values() for stem in ids}
    if any(stem in dev_ids for stem in stems):
        raise SystemExit("[driver] 校准集与开发集有交集，拒绝运行")

    model, config = quant._load_fp32(BASELINE_SOUP)
    if (int(config["base_channels"]), int(config["num_pools"]), int(config["size"])) != (8, 3, 256):
        raise SystemExit(f"[driver] 起点结构异常: {config}")
    config_dict = build_config(args, config)
    config_digest = config_hash(config_dict)
    input_hashes = {
        "init_checkpoint": BASELINE_SOUP_SHA256,
        "split_train_ids": str(assembly.split_digest["train_ids"]),
        "split_development_all": str(assembly.split_digest["development_all"]),
        "calibration_ids": calibration_ids_digest(stems),
        "frozen_exposed_split": str(assembly.split_digest["reference_split_file"]),
    }
    code_hashes = guard.hash_files(CODE_PATHS)

    # 设备与 RNG：先 to(device)，再建 optimizer
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    seed_everything(int(args.seed))
    model = model.to(device)
    qat_names = qat.enable_weight_qat(model)
    if len(qat_names) != 15:
        raise SystemExit(f"[driver] QAT 层数不是 15: {len(qat_names)}")
    initial = recompute_scales(model, calibration_items, int(config["size"]))
    initial_scales = qat.scales_snapshot(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)

    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    start_epoch = 1
    resume_phase: str | None = None
    scales = initial_scales
    scales_digest_value = scales_digest(scales)

    if args.resume:
        if not checkpoint_path.is_file():
            raise SystemExit(f"[driver] 找不到断点: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != SCHEMA:
            raise SystemExit(f"[driver] 断点 schema 不支持: {payload.get('schema')!r}")
        guard.verify_guard(payload.get("guard", {}), code_paths=CODE_PATHS,
                           config_hash=config_digest, inputs=input_hashes)   # 硬拒绝
        saved_scales = payload.get("scales")
        if not isinstance(saved_scales, dict) or not saved_scales:
            raise SystemExit("[driver] 断点缺少尺度记录，拒绝 resume")
        if payload.get("scales_digest") != scales_digest(saved_scales):
            raise SystemExit("[driver] 断点尺度自身摘要不一致，拒绝 resume")
        if payload.get("calibration_ids_sha256") != input_hashes["calibration_ids"]:
            raise SystemExit("[driver] 断点校准 ID 与当前不一致，拒绝 resume")
        model.load_state_dict(payload["model_state"], strict=True)
        apply_saved_scales(model, saved_scales)
        scales = saved_scales
        scales_digest_value = scales_digest(scales)
        optimizer.load_state_dict(payload["optimizer_state"])
        restore_rng(payload["rng"])
        history = list(payload.get("history", []))
        best = payload.get("best")
        start_epoch, resume_phase, need = resume_decision(payload)
        if not need and start_epoch != int(payload["epoch"]):
            raise SystemExit("[driver] 断点恢复判定异常")
        print(f"[driver] resume 门禁通过：断点在 epoch {payload['epoch']} phase={payload.get('phase')}，"
              f"从 epoch {start_epoch} 继续（{'仅补评估' if not need else '正常训练'}，"
              f"尺度摘要 {scales_digest_value[:12]}）", flush=True)

    def persist(epoch: int, phase: str, lr: float, train_metrics: dict[str, Any],
                evaluated: dict[str, Any] | None = None) -> None:
        record = guard.build_guard(code_paths=CODE_PATHS, config_hash=config_digest,
                                   inputs=input_hashes)
        record["scales_digest"] = scales_digest_value
        record["calibration_ids_sha256"] = input_hashes["calibration_ids"]
        payload = {
            "schema": SCHEMA, "seed": int(args.seed), "epoch": int(epoch), "phase": phase,
            "lr": float(lr), "config": config_dict, "config_hash": config_digest,
            "guard": record, "input_hashes": input_hashes, "code_hashes": code_hashes,
            "scales": scales, "scales_digest": scales_digest_value,
            "calibration_ids_sha256": input_hashes["calibration_ids"],
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "optimizer_class": type(optimizer).__name__, "rng": rng_state(),
            "history": history, "best": best, "train_metrics": train_metrics,
            "evaluated": evaluated, "device": str(device),
            "elapsed_seconds": time.time() - start,
            "resume_scope": "phase=trained 写在评估前、phase=evaluated 写在评估后；"
                            "含优化器/尺度/校准 ID/Python+NumPy+Torch+CUDA RNG",
        }
        atomic_torch_save(payload, checkpoint_path)

    epochs = int(args.epochs)
    train_metrics: dict[str, Any] = {}
    for epoch in range(start_epoch, epochs + 1):
        need_training = not (resume_phase == "trained" and epoch == start_epoch)
        if epoch == RECALIBRATE_AFTER_EPOCH + 1 and need_training:
            recalibrated = recompute_scales(model, calibration_items, int(config["size"]))
            new_scales = qat.scales_snapshot(model)
            drift = {name: {"input_ratio": new_scales[name]["input_scale"] /
                                      scales[name]["input_scale"],
                            "output_ratio": new_scales[name]["output_scale"] /
                                       scales[name]["output_scale"]}
                     for name in scales}
            scales = new_scales
            scales_digest_value = scales_digest(scales)
            atomic_json({"epoch": epoch, "recalibrated": recalibrated, "drift_ratios": drift,
                         "scales_digest": scales_digest_value},
                        output_dir / "calibration_recompute.json")
            print(f"[driver] 第 {epoch} 轮起使用重算并冻结的尺度（最大比值 "
                  f"{max(max(v.values()) for v in drift.values()):.4f}）", flush=True)

        candidate = output_dir / "candidates" / f"epoch_{epoch:02d}.pt"
        if need_training:
            dataset = overall.TrainDataset(assembly.train_items, size=int(config["size"]),
                                           seed=int(args.seed), augment=True, epoch=epoch)
            loader = overall.make_train_loader(dataset, batch_size=int(args.batch_size),
                                               num_workers=int(args.num_workers),
                                               seed=int(args.seed), epoch=epoch)
            model.train(True)
            totals = {"loss": 0.0, "bce": 0.0, "soft_dice": 0.0}
            batches = 0
            for batch in loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                valids = batch["valid"].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss, parts = segmentation_loss(logits, masks, valids, bce_weight=1.0,
                                                dice_weight=3.0)
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                optimizer.step()
                totals["loss"] += float(loss.detach())
                totals["bce"] += float(parts["bce"])
                totals["soft_dice"] += float(parts["soft_dice"])
                batches += 1
                if args.limit_batches and batches >= int(args.limit_batches):
                    break
            if batches == 0:
                raise SystemExit("[driver] 训练 loader 为空")
            train_metrics = {key: value / batches for key, value in totals.items()}
            train_metrics["batches"] = batches
            train_metrics["lr"] = float(args.lr)
            history.append({"epoch": epoch, "phase": "trained", **train_metrics})
            persist(epoch, "trained", float(args.lr), train_metrics)
            print(f"[driver] epoch {epoch}/{epochs} trained loss={train_metrics['loss']:.6f} "
                  f"({train_metrics['batches']} batches, {time.time() - start:.0f}s)", flush=True)

            plain = qat.unwrap(model)
            atomic_torch_save({"schema": CANDIDATE_SCHEMA, "epoch": epoch,
                               "config": config_dict, "model_state": plain.state_dict(),
                               "init_checkpoint_sha256": BASELINE_SOUP_SHA256,
                               "scales": scales, "scales_digest": scales_digest_value}, candidate)
            del plain
        elif not candidate.is_file():
            raise SystemExit(f"[driver] 断点为 trained 阶段但找不到候选 {candidate}")

        entry: dict[str, Any] = {"epoch": epoch, "candidate": candidate.as_posix(),
                                 "candidate_sha256": guard.sha256_file(candidate),
                                 "train_metrics": train_metrics}
        if not args.skip_eval:
            evaluated = export_and_evaluate(candidate, output_dir / "int8" / f"epoch_{epoch:02d}",
                                            output_dir / "development" / f"epoch_{epoch:02d}",
                                            output_dir, epoch)
            entry.update({k: v for k, v in evaluated.items() if k != "epoch"})
        history.append({"epoch": epoch, "phase": "evaluated",
                        **{k: v for k, v in entry.items() if k != "train_metrics"}})
        if entry.get("int8_dice") is not None and (best is None or
                                                   entry["int8_dice"] > best["int8_dice"]):
            best = dict(entry)
        persist(epoch, "evaluated", float(args.lr), train_metrics, evaluated=entry)
        resume_phase = None

    atomic_json({"schema": "a-final-qat-history-v1", "config": config_dict,
                 "config_hash": config_digest, "input_hashes": input_hashes,
                 "code_hashes": code_hashes, "initial_scales": initial_scales,
                 "scales": scales, "scales_digest": scales_digest_value,
                 "history": history, "best": best,
                 "baseline_int8_dice": BASELINE_INT8_DICE,
                 "adopted": bool(best and best.get("int8_dice", 0) > BASELINE_INT8_DICE),
                 "elapsed_seconds": time.time() - start}, output_dir / "history.json")
    print(json.dumps({"best": best, "baseline": BASELINE_INT8_DICE,
                      "adopted": bool(best and best.get("int8_dice", 0) > BASELINE_INT8_DICE),
                      "elapsed_seconds": round(time.time() - start, 1)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
