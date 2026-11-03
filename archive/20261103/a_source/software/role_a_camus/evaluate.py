"""CAMUS 正样态结果建模。

定义三种语义级别，避免"任意插值结果被当作临床结论"：

* :class:`Measured` —— 直接依据输入像素/物理量测得的量（例如已存在的面积，单位 mm²）。
* :class:`Modelled` —— 明确定义为**建模**结果、由某种约定插值得到（例如官方 notebook 的 LV 体积）。
* :data:`IGNORED` —— 因输入缺失、被排除或算法被配置为忽略而得出的结果。

所有类都会在自身不适用时返回 ``None`` 值并给出原因，绝不以 0 或默认值代替。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from . import geometry as geo
from . import metrics as camus_metrics

CITATION = "Leclerc et al. 2019, IEEE TMI (CAMUS); code reference: camus_ef_reference (official)"
LICENSE = "CC BY-NC-SA 4.0 (non-commercial scientific research)"


class EvaluationRefused(RuntimeError):
    """测试集或其他受限读取被拒绝。"""


def assert_split_allowed(split: str, *, allow_test: bool = False) -> None:
    """测试集默认封条：只有显式 ``allow_test=True`` 才允许继续。"""
    if split == "testing" and not allow_test:
        raise EvaluationRefused("测试集默认封条：需 owner 显式放行后才可评估")


def evaluate_dataset(frames: Sequence[Mapping[str, Any]], split: str, *,
                     allow_test: bool = False) -> dict[str, Any]:
    """汇总已算好的正样态记录；测试集默认拒绝。"""
    assert_split_allowed(split, allow_test=allow_test)
    summary = camus_metrics.evaluate_frames(frames)
    summary.update({"split": split, "citation": CITATION, "license": LICENSE,
                    "note": "所有指标来自正样态 argmax；Dice 不等于发现率"})
    return summary


@dataclass(frozen=True)
class VolumeResult:
    """体积/射血分数结果；``value`` 为 ``None`` 时给出 ``reason``。"""

    value: float | None
    reason: str | None = None
    details: Mapping[str, Any] | None = None

    @property
    def valid(self) -> bool:
        return self.value is not None

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "reason": self.reason, "details": dict(self.details or {})}


@dataclass(frozen=True)
class EjectionFraction(VolumeResult):
    """射血分数（%）及其派生量。"""

    edv_ml: float | None = None
    esv_ml: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "reason": self.reason, "details": dict(self.details or {}),
                "edv_ml": self.edv_ml, "esv_ml": self.esv_ml}


@dataclass(frozen=True)
class GeometryUse:
    """几何量的使用方式，用于区分正样态/插值/忽略三种语义。"""

    kind: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("measured", "modelled", "ignored"):
            raise ValueError(f"未知几何使用类型: {self.kind!r}")


def approximate_chamber_area_mm2(mask_mm2: float | None, *, note: str = "") -> VolumeResult:
    """**近似面积**（单位 mm²）：把二值掩膜面积按像素物理尺寸相加。

    这是直接量测，不是临床定义的心腔面积；命名与报告都必须标注为近似。
    """
    if mask_mm2 is None:
        return VolumeResult(None, "掩膜为空或几何无效，无法给出近似面积")
    if not (mask_mm2 >= 0) or mask_mm2 != mask_mm2:
        return VolumeResult(None, f"非法面积: {mask_mm2!r}")
    return VolumeResult(float(mask_mm2), None,
                        {"kind": "measured", "note": note or "二值掩膜面积求和（近似，非临床定义）"})


def approximate_volume_disc_method(area_mm2: float | None, length_mm: float | None
                                   ) -> VolumeResult:
    """**近似**单平面长度法体积（mL）：``V = 8A²/(3πL)``。

    该式是历史估算方法，**不是官方 biplane Simpson**；正式报告请使用
    :mod:`software.role_a_camus.official_volume`。
    """
    if area_mm2 is None or length_mm is None:
        return VolumeResult(None, "缺少面积或长度，无法给出长度法近似体积")
    if not (area_mm2 == area_mm2 and length_mm == length_mm) or area_mm2 <= 0 or length_mm <= 0:
        return VolumeResult(None, f"非法输入 area={area_mm2!r} length={length_mm!r}")
    volume_mm3 = 8.0 * area_mm2 ** 2 / (3.0 * math_pi() * length_mm)
    return VolumeResult(volume_mm3 / 1000.0, None,
                        {"method": "area-length approximation (NOT official biplane Simpson)",
                         "area_mm2": area_mm2, "length_mm": length_mm, "volume_mm3": volume_mm3})


def math_pi() -> float:
    import math

    return math.pi


def approximate_biplane_simpson(area_2ch_mm2: float | None, length_2ch_mm: float | None,
                               area_4ch_mm2: float | None, length_4ch_mm: float | None
                               ) -> VolumeResult:
    """**近似**双平面 Simpson 体积（mL）：两视图面积与平均长度的解析式。

    与官方 notebook 的 20-disk 逐层求和**不等价**；仅作对照，正式报告使用官方实现。
    """
    values = (area_2ch_mm2, area_4ch_mm2, length_2ch_mm, length_4ch_mm)
    if any(value is None for value in values):
        return VolumeResult(None, "缺少任一视图的面积或长度，无法给出近似双平面体积")
    if any((value != value) or value <= 0 for value in values):
        return VolumeResult(None, f"非法输入: {values!r}")
    length_mean = 0.5 * (length_2ch_mm + length_4ch_mm)
    volume_mm3 = (8.0 / (3.0 * math_pi())) * (area_2ch_mm2 * area_4ch_mm2) / length_mean
    return VolumeResult(volume_mm3 / 1000.0, None,
                        {"method": "analytic biplane approximation (NOT official 20-disk Simpson)",
                         "length_mean_mm": length_mean, "volume_mm3": volume_mm3})


def ejection_fraction(end_diastolic_ml: float | None, end_systolic_ml: float | None) -> EjectionFraction:
    """EF(%) = (EDV−ESV)/EDV×100；缺项、EDV≤0、ESV>EDV 时返回 ``None`` + 原因。"""
    if end_diastolic_ml is None or end_systolic_ml is None:
        return EjectionFraction(None, "缺少 EDV 或 ESV，无法计算 EF",
                                edv_ml=end_diastolic_ml, esv_ml=end_systolic_ml)
    if end_diastolic_ml != end_diastolic_ml or end_systolic_ml != end_systolic_ml:
        return EjectionFraction(None, "EDV/ESV 含非有限值", edv_ml=end_diastolic_ml,
                                esv_ml=end_systolic_ml)
    if end_diastolic_ml <= 0:
        return EjectionFraction(None, f"EDV 非正: {end_diastolic_ml!r}", edv_ml=end_diastolic_ml,
                                esv_ml=end_systolic_ml)
    if end_systolic_ml > end_diastolic_ml:
        return EjectionFraction(None, f"ESV({end_systolic_ml}) > EDV({end_diastolic_ml})，几何不可信",
                                edv_ml=end_diastolic_ml, esv_ml=end_systolic_ml)
    fraction = (end_diastolic_ml - end_systolic_ml) / end_diastolic_ml * 100.0
    return EjectionFraction(fraction, None, {"edv_ml": end_diastolic_ml, "esv_ml": end_systolic_ml},
                            edv_ml=end_diastolic_ml, esv_ml=end_systolic_ml)


def lv_length_mm(mask: Any, spacing_y_mm: float, spacing_x_mm: float, *,
                 axis: str = "vertical") -> VolumeResult:
    """LV 腔掩膜沿主轴的像素跨度换算为 mm（mm）；空掩膜/退化形状给 ``None``。"""
    import numpy as np

    array = np.asarray(mask, dtype=bool)
    if not array.any():
        return VolumeResult(None, "LV 腔掩膜为空，无法测量长度")
    if axis not in ("vertical", "horizontal"):
        return VolumeResult(None, f"未知主轴: {axis!r}")
    spacing = spacing_y_mm if axis == "vertical" else spacing_x_mm
    if not (spacing == spacing) or spacing <= 0:
        return VolumeResult(None, f"非法 spacing: {spacing!r}")
    rows = np.where(array.any(axis=1))[0]
    cols = np.where(array.any(axis=0))[0]
    extent = float(rows[-1] - rows[0] + 1) if axis == "vertical" else float(cols[-1] - cols[0] + 1)
    length = extent * float(spacing)
    if length <= 0:
        return VolumeResult(None, "退化长度")
    return VolumeResult(length, None, {"pixels": extent, "spacing_mm": float(spacing), "axis": axis})


def geometry_valid(spacing_mm: Sequence[float] | None, height: int, width: int) -> VolumeResult:
    """几何有效性：spacing 缺失/非正或尺寸非法时返回 ``None`` + 原因。"""
    if spacing_mm is None or len(spacing_mm) < 2:
        return VolumeResult(None, "缺少 spacing，无法做物理量换算")
    spacing_y, spacing_x = float(spacing_mm[0]), float(spacing_mm[1])
    if height <= 0 or width <= 0:
        return VolumeResult(None, f"非法尺寸 {height}x{width}")
    if min(spacing_y, spacing_x) <= 0 or spacing_y != spacing_y or spacing_x != spacing_x:
        return VolumeResult(None, f"非法 spacing: {spacing_mm!r}")
    return VolumeResult(float(spacing_y * spacing_x), None,
                        {"spacing_y_mm": spacing_y, "spacing_x_mm": spacing_x})


def official_biplane_report(lv_masks: Mapping[str, Any], spacing_by_key: Mapping[str, Sequence[float]]
                            ) -> dict[str, Any]:
    """官方 20-disk 双平面体积/EF 的**薄封装**（数学边界由 Codex 接管的 ``official_volume`` 决定）。

    ``lv_masks`` 与 ``spacing_by_key`` 都以 ``2CH_ED/2CH_ES/4CH_ED/4CH_ES`` 为 key；mask 为
    **原尺寸二值**数组，spacing 为该视图自己的 ``(H, W)`` mm。失败（空/非单连通/NaN spacing/
    ED-ES 几何不一致等）由官方实现返回 ``null`` + ``reason``，本函数不猜测、不挑最大连通域。
    """
    from . import official_volume as official

    report = official.biplane_report(lv_masks, spacing_by_key)
    report.setdefault("fallback_used", False)
    return report


# --------------------------------------------------------------------------- CLI

def evaluate_run(checkpoint: Path, data_manifest: Path, split: str, out_dir: Path, *,
                 allow_test: bool = False, device: str = "cpu", make_figures: bool = False,
                 compute_volumes: bool = True) -> dict[str, Any]:
    """冻结 best 之后的一次完整评估：**native GT** 指标 + HD95/面积 + 低覆盖 + 官方体积/EF。

    所有指标基于原生 GT（缓存中的 ``native_label``）；模型输出经 letterbox → 双线性回原网格
    → argmax 后与原生 GT 比较，**不使用 256 标签逆 resize**。
    """
    import torch

    from . import geometry as geo
    from . import metrics as camus_metrics
    from . import model as camus_model
    from . import train as camus_train

    assert_split_allowed(split, allow_test=allow_test)
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    config_payload = payload.get("config", {})
    model = camus_model.build_model(base_channels=config_payload.get("base_channels", 16),
                                    num_pools=config_payload.get("num_pools", 3),
                                    size=config_payload.get("size", 256))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    torch_device = torch.device(device)
    model.to(torch_device)
    frames = camus_train.load_cache_frames(Path(data_manifest), split)
    records: list[dict[str, Any]] = []
    native_pairs: list[dict[str, Any]] = []
    for frame in frames:
        with torch.no_grad():
            logits = model(torch.from_numpy(np.asarray(frame.image, dtype=np.float32))[None, None]
                           .to(torch_device))
        native_prediction = geo.argmax_on_native_torch(logits.cpu(), frame.meta)[0].numpy()
        if frame.native_label is None:
            records.append({"frame_id": f"{frame.patient}_{frame.view}_{frame.phase}",
                            "patient": frame.patient, "view": frame.view, "phase": frame.phase,
                            "quality": frame.quality, "per_class": {},
                            "foreground_macro_dice": None, "hd95_mm": {}, "gt_area_mm2": {},
                            "pred_area_mm2": {}, "reason": "native GT 缺失，拒绝代替"})
            continue
        record = camus_metrics.native_frame_scores(
            native_prediction, frame.native_label, frame.spacing_mm, quality=frame.quality,
            patient=frame.patient, view=frame.view, phase=frame.phase,
            frame_id=f"{frame.patient}_{frame.view}_{frame.phase}")
        records.append(record)
        native_pairs.append({"frame": frame, "prediction": native_prediction,
                             "truth": frame.native_label, "record": record})
    summary = evaluate_dataset(records, split, allow_test=allow_test)
    summary["quality_source"] = "official Info_2CH.cfg / Info_4CH.cfg ImageQuality"
    if compute_volumes:
        summary["official_volumes"] = _official_volume_section(native_pairs)
    if make_figures:
        from . import visualize

        out_dir = Path(out_dir)
        by_patient: dict[str, dict[str, Any]] = {}
        for entry in native_pairs:
            by_patient.setdefault(entry["frame"].patient, {})[
                f"{entry['frame'].view}_{entry['frame'].phase}"] = entry
        for patient, entries in sorted(by_patient.items()):
            for view in geo.VIEWS:
                ed, es = entries.get(f"{view}_ED"), entries.get(f"{view}_ES")
                if ed is None or es is None:
                    continue
                figure = visualize.ed_es_pair(
                    ed["frame"].native_image, ed["truth"], ed["prediction"],
                    es["frame"].native_image, es["truth"], es["prediction"],
                    title=f"{patient} {view} (dashed=GT, solid=pred)")
                visualize.save_figure(figure, out_dir / f"{patient}_{view}_EDES.png")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"evaluate_{split}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _official_volume_section(native_pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """逐患者的官方 20-disk 双平面体积：**预测与 GT 各一份**，每 key 用**该视图自己的 spacing**。"""
    from . import geometry as geo

    per_patient: dict[str, dict[str, Any]] = {}
    for entry in native_pairs:
        frame = entry["frame"]
        per_patient.setdefault(frame.patient, {})[f"{frame.view}_{frame.phase}"] = {
            "view": frame.view, "phase": frame.phase, "spacing_mm": tuple(frame.spacing_mm),
            "pred_lv": entry["prediction"] == geo.LV_CAVITY,
            "gt_lv": entry["truth"] == geo.LV_CAVITY}
    result: dict[str, Any] = {}
    for patient, entries in sorted(per_patient.items()):
        record: dict[str, Any] = {"prediction": None, "ground_truth": None, "reason": None,
                                  "spacing_by_key": {}}
        required = [f"{view}_{phase}" for view in geo.VIEWS for phase in geo.PHASES]
        missing = [key for key in required if key not in entries]
        if missing:
            record["reason"] = f"缺少端点: {missing}"
            result[patient] = record
            continue
        spacing_by_key = {key: entries[key]["spacing_mm"] for key in required}
        record["spacing_by_key"] = {key: list(value) for key, value in spacing_by_key.items()}
        record["prediction"] = official_biplane_report(
            {key: entries[key]["pred_lv"] for key in required}, spacing_by_key)
        record["ground_truth"] = official_biplane_report(
            {key: entries[key]["gt_lv"] for key in required}, spacing_by_key)
        result[patient] = record
    return {"method": "official 20-disk biplane Simpson (pred vs GT, per-key native spacing)",
            "patients": result,
            "note": "预测与 GT 各一份报告；每个 key 使用自己的原生 (H, W) spacing；不平均单视图体积"}


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAMUS 冻结模型评估（native 指标 + 官方体积）")
    parser.add_argument("--checkpoint", required=True, help="best.pt 绝对路径")
    parser.add_argument("--data-manifest", required=True, help="prepare 缓存清单 JSON")
    parser.add_argument("--split", default="validation", choices=["training", "validation", "testing"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--no-volumes", action="store_true")
    parser.add_argument("--allow-test", action="store_true",
                        help="仅在 owner 一次性放行测试 GT 后使用")
    return parser


def cli_main(argv: Sequence[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv)
    summary = evaluate_run(Path(args.checkpoint), Path(args.data_manifest), args.split,
                           Path(args.out_dir), allow_test=args.allow_test, device=args.device,
                           make_figures=args.figures, compute_volumes=not args.no_volumes)
    print(json.dumps({"split": summary["split"],
                      "primary_metric_patient_balanced": summary.get("primary_metric_patient_balanced"),
                      "frames": summary["frame_count"]}, ensure_ascii=False), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m software.role_a_camus.evaluate`` 入口。"""
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
