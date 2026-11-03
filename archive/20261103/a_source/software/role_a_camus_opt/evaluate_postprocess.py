"""在**获 owner 批准后**对已保存的 dev 预测做后处理评估（不读 test、不重推理）。

严格绑定（缺失或空通配一律拒绝）：

* 许可必须给出**完整前缀身份**：``base_source_identity``（10 个原基础源）与
  ``opt_source_identity``（5 个 opt 顶层源），且与磁盘逐项一致；**禁止**用同名 dict 互相覆盖、
  也**禁止**空 dict 通配；
* 许可绑定协议 SHA（``PROTOCOL.json``）、输入 ``validation01`` 的 ``SUMMARY.json``/``RECORDS.jsonl``
  与 **200 个 NPZ** 的 SHA、**官方 50 名 dev 四端点精确名单**、输出目录、新 STOP 路径、
  ``max_seconds<=900`` 与 40 GB 预算，以及 fast/official 两个 helper 的源码身份；
* ``test_read_approved`` 与 ``inference_approved`` **必须存在且为 false**（缺失或 True 都拒绝）。

评估行为：

* 每帧与逐患者都检查时间与 STOP；固定间隔复查 40 GB（失败按错误处理，不当成空间充足）；
* fast 体积仅在许可要求优化且等价检查通过时使用，并额外校验 ED/ES 各视图 shape/spacing 一致、
  输出 finite、``EDV>0``、``ESV>=0``；异常一律 ``null`` + reason（不逃逸、不填 0）；
* 200 帧原 native NPZ 与逐帧记录、逐患者 volume 报告**逐步落盘**，失败也留证；
* 至少对代表例/最差例/改变最多例绘制 ``原 native 灰度 + GT + raw + processed`` 对照图
  （matplotlib Agg + 项目 plot_cache，绘后 close）；
* **共同有效 MAE**（协议要求）= raw、候选、GT **同一患者交集**、逐字段 finite；同时报告 raw 侧与
  候选侧的 MAE 与分母、名单；不得用"各候选与 GT 的不同子集"冒充配对比较；
* 判定门槛按协议：Dice >= raw、LV HD95 <= raw、EF 成功 > 24/50、共同有效 EDV/ESV/EF 不变差；
  报告全部病例与无效计数，**不自行改门槛**。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT = Path(r"E:/嵌赛/pld-medical")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from software.role_a_camus import data as base_data             # noqa: E402
from software.role_a_camus import geometry as base_geo          # noqa: E402
from software.role_a_camus import metrics as base_metrics        # noqa: E402
from software.role_a_camus import official_volume as official    # noqa: E402
from software.role_a_camus_opt import postprocess as pp          # noqa: E402

RUN = PROJECT / "runs/role_a_camus/20261012"
RUN_OPT = PROJECT / "runs/role_a_camus/20261012_opt01"
VALIDATION01 = RUN / "evaluations/validation01"
PROTOCOL = RUN_OPT / "PROTOCOL.json"
CAMUS_ROOT = PROJECT / "datasets/camus_public/official/CAMUS_public"
FAST_HELPER = RUN / "execution/final_eval01/contour_diameters_fast.py"
OFFICIAL_HELPER = PROJECT / "software/role_a_camus/official_volume.py"
BASE_PACKAGE = PROJECT / "software/role_a_camus"
OPT_PACKAGE = PROJECT / "software/role_a_camus_opt"
BASE_SOURCES = ("__init__.py", "data.py", "evaluate.py", "geometry.py", "metrics.py", "model.py",
                "official_volume.py", "prepare.py", "train.py", "visualize.py")
OPT_SOURCES = ("__init__.py", "postprocess.py", "boundary_loss.py", "train.py",
               "evaluate_postprocess.py")
EXPECTED_FRAMES = 200
EXPECTED_PATIENTS = 50
ENDPOINTS = tuple((view, phase) for view in ("2CH", "4CH") for phase in ("ED", "ES"))
MAX_SECONDS_LIMIT = 900
PROJECT_BUDGET = 40_000_000_000
REPARSE_FLAG = 0x400
EF_VALID_MIN = 24          # 协议：EF 成功患者必须 > 24/50


class PermitRejected(RuntimeError):
    """缺少 owner 许可或绑定不匹配——一律拒绝运行。"""


class BudgetExceeded(RuntimeError):
    """STOP 或时间/空间预算触发。"""


class VolumeFailed(RuntimeError):
    """体积计算失败（由调用方转成 null + reason，不逃逸、不填 0）。"""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def scan(root: Path) -> dict[str, int]:
    """os.scandir 遍历并跳过全部重解析条目（不跟随链接）。"""
    files = dirs = skipped = errors = 0
    total = 0
    stack = [Path(root)]
    while stack:
        current = stack.pop()
        try:
            entries = os.scandir(current)
        except OSError:
            errors += 1
            continue
        with entries:
            for entry in entries:
                try:
                    stat = entry.stat(follow_symlinks=False)
                    if entry.is_symlink() or stat.st_file_attributes & REPARSE_FLAG:
                        skipped += 1
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        dirs += 1
                        stack.append(Path(entry.path))
                    else:
                        files += 1
                        total += stat.st_size
                except OSError:
                    errors += 1
    return {"logical_bytes": total, "files": files, "directories": dirs,
            "skipped_reparse": skipped, "errors": errors}


def _disk_identity(package: Path, names: Sequence[str]) -> dict[str, str]:
    return {name: sha256_file(package / name) for name in names}


def official_dev_patients() -> tuple[str, ...]:
    """官方 dev(validation) 50 人精确名单（只读 database_split，不解码任何体素）。"""
    split = base_data.load_split(CAMUS_ROOT / "database_split")
    return tuple(sorted(split.validation))


def _require_full_identity(permit: Mapping[str, Any], key: str, package: Path,
                           names: Sequence[str], label: str) -> dict[str, str]:
    declared = permit.get(key)
    if not isinstance(declared, dict) or not declared:
        raise PermitRejected(f"许可缺少 {key}（{label}，不允许空通配）")
    if set(declared) != set(names):
        raise PermitRejected(f"{key} 必须精确包含 {len(names)} 个文件，收到 {sorted(declared)}")
    expected = {name: sha256_file(package / name) for name in names}
    if dict(declared) != expected:
        mismatched = sorted(name for name in names if declared.get(name) != expected[name])
        raise PermitRejected(f"{label} 源身份不符: {mismatched}")
    return expected


def load_permit(path: Path) -> dict[str, Any]:
    """owner 许可校验（含完整前缀身份、协议/输入/名单/输出/预算/helper 绑定）。"""
    path = Path(path)
    if not path.is_file():
        raise PermitRejected(f"缺少 owner 许可: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "camus-postprocess-eval-permit-v1":
        raise PermitRejected(f"许可 schema 非法: {payload.get('schema')!r}")
    if payload.get("approved") is not True:
        raise PermitRejected("许可未 approved")
    for flag in ("inference_approved", "test_read_approved"):
        if flag not in payload:
            raise PermitRejected(f"许可必须显式声明 {flag}=false（缺失即拒绝）")
        if payload[flag] is not False:
            raise PermitRejected(f"{flag} 必须为 false（本入口不重推理、不读 test）")

    base_expected = _require_full_identity(payload, "base_source_identity", BASE_PACKAGE,
                                           BASE_SOURCES, "原基础源")
    opt_expected = _require_full_identity(payload, "opt_source_identity", OPT_PACKAGE,
                                          OPT_SOURCES, "opt 源")
    # Separate base/opt dictionaries are distinct namespaces; shared filenames are expected.

    if sha256_file(PROTOCOL) != payload.get("protocol_sha256"):
        raise PermitRejected("协议 SHA 不符")
    if Path(payload.get("validation_dir", "")).resolve() != VALIDATION01.resolve():
        raise PermitRejected(f"绑定的 dev 目录不符: {payload.get('validation_dir')!r}")
    inputs = payload.get("inputs") or {}
    for name in ("SUMMARY.json", "RECORDS.jsonl"):
        recorded = inputs.get(name)
        if not recorded or sha256_file(VALIDATION01 / name) != recorded:
            raise PermitRejected(f"输入 {name} SHA 不符/缺失")
    npz_expected = inputs.get("npz") or {}
    npz_actual = {path.name: sha256_file(path) for path in sorted((VALIDATION01 / "cases").glob("*_pred.npz"))}
    if len(npz_actual) != EXPECTED_FRAMES or dict(npz_expected) != npz_actual:
        raise PermitRejected(f"200 个 NPZ SHA 与许可不符（实际 {len(npz_actual)}）")
    patients = tuple(payload.get("dev_patients") or ())
    if patients != official_dev_patients() or len(patients) != EXPECTED_PATIENTS:
        raise PermitRejected("许可中的 dev 患者名单必须与官方 validation 50 人完全一致")
    endpoints = payload.get("dev_endpoints") or {}
    for patient in patients:
        if tuple(tuple(item) for item in endpoints.get(patient, ())) != ENDPOINTS:
            raise PermitRejected(f"{patient} 的四端点名单不符（必须 {ENDPOINTS}）")
    opt_run = PROTOCOL.parent
    try:
        Path(payload.get("out_dir", "")).resolve().relative_to((opt_run / "evaluations").resolve())
    except ValueError as error:
        raise PermitRejected("输出目录必须位于本新run/evaluations内") from error
    if Path(payload.get("new_stop_file", "")).resolve() != (opt_run / "STOP_REQUEST").resolve():
        raise PermitRejected("必须绑定本新run的STOP_REQUEST")
    max_seconds = int(payload.get("max_seconds") or 0)
    if not 0 < max_seconds <= MAX_SECONDS_LIMIT:
        raise PermitRejected(f"max_seconds 必须在 (0, {MAX_SECONDS_LIMIT}]，收到 {max_seconds}")
    if int(payload.get("project_max_bytes") or 0) != PROJECT_BUDGET:
        raise PermitRejected("许可必须绑定 40GB 项目预算")
    helpers = payload.get("helper_source_identity") or {}
    for name, helper in (("contour_diameters_fast.py", FAST_HELPER),
                         ("official_volume.py", OFFICIAL_HELPER)):
        if not helper.is_file() or helpers.get(name) != sha256_file(helper):
            raise PermitRejected(f"helper 源身份不符/缺失: {name}")
    return {**payload, "_base_identity": base_expected, "_opt_identity": opt_expected}


def ensure_fresh_dir(path: Path) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise PermitRejected(f"输出目录已存在且非空: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _records() -> list[dict[str, Any]]:
    path = VALIDATION01 / "RECORDS.jsonl"
    if not path.is_file():
        raise PermitRejected(f"缺少 RECORDS.jsonl: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _case_files() -> dict[str, Path]:
    return {path.name.replace("_pred.npz", ""): path
            for path in sorted((VALIDATION01 / "cases").glob("*_pred.npz"))}


def _check_budget(started: float, max_seconds: int, stop_file: Path | None,
                  last_disk_check: float, *, interval: int = 50) -> float:
    """每帧/每患者的时间与 STOP 检查；固定间隔复查 40 GB（失败按错误处理）。"""
    if stop_file is not None and Path(stop_file).exists():
        raise BudgetExceeded(f"STOP_REQUEST: {stop_file}")
    if time.monotonic() - started > max_seconds:
        raise BudgetExceeded(f"超过 max_seconds={max_seconds}")
    now = time.monotonic()
    if now - last_disk_check > interval:
        stats = scan(PROJECT)
        if stats["errors"]:
            raise BudgetExceeded(f"预算扫描错误 {stats['errors']}，不得视为空间充足")
        if stats["logical_bytes"] >= PROJECT_BUDGET:
            raise BudgetExceeded(f"项目预算超限: {stats['logical_bytes']}")
        return now
    return last_disk_check


def _fast_volume(masks: Mapping[str, np.ndarray], spacing_by_key: Mapping[str, Sequence[float]],
                 fast: Any) -> dict[str, Any]:
    """等价 fast 量径 + 官方公式；异常一律 null + reason（不逃逸、不填 0）。"""
    report: dict[str, Any] = {"method": "official 20-disk biplane Simpson (fast diameters)",
                              "edv_ml": None, "esv_ml": None, "ef_percent": None, "reason": None}
    try:
        clean, spaces = {}, {}
        for key in ("2CH_ED", "2CH_ES", "4CH_ED", "4CH_ES"):
            clean[key], spaces[key] = official.validate_lv_mask(masks.get(key),
                                                               spacing_by_key.get(key))
        for view in ("2CH", "4CH"):
            if clean[f"{view}_ED"].shape != clean[f"{view}_ES"].shape:
                raise VolumeFailed(f"{view}: ED/ES shape 不一致")
            if not np.allclose(spaces[f"{view}_ED"], spaces[f"{view}_ES"], rtol=0, atol=1e-5):
                raise VolumeFailed(f"{view}: ED/ES spacing 不一致")
        outcomes = {}
        for key in ("2CH_ED", "2CH_ES", "4CH_ED", "4CH_ES"):
            outcome = fast.contour_diameters_fast(clean[key], spaces[key])
            if not outcome.ok:
                raise VolumeFailed(f"{key}: {outcome.reason}")
            outcomes[key] = outcome
        step = max(outcomes[key].step_size for key in outcomes)
        edv = official.volumes_by_instant(outcomes["2CH_ED"].diameters,
                                          outcomes["4CH_ED"].diameters, step)
        esv = official.volumes_by_instant(outcomes["2CH_ES"].diameters,
                                          outcomes["4CH_ES"].diameters, step)
        if not np.isfinite(edv) or not np.isfinite(esv):
            raise VolumeFailed("非有限体积")
        if edv <= 0 or esv < 0:
            raise VolumeFailed(f"非法体积 edv={edv} esv={esv}")
        report["edv_ml"], report["esv_ml"] = float(edv), float(esv)
        if esv > edv:
            report["reason"] = "ESV exceeds EDV; EF undefined for this geometry"
        else:
            report["ef_percent"] = 100.0 * (edv - esv) / edv
    except Exception as error:  # noqa: BLE001 - 官方口径失败一律 null + reason
        report["reason"] = f"{type(error).__name__}: {error}"
    return report


def _official_volume(masks: Mapping[str, np.ndarray], spacing_by_key: Mapping[str, Sequence[float]],
                     *, fast: Any | None, optimize: bool) -> dict[str, Any]:
    if optimize and fast is not None:
        return _fast_volume(masks, spacing_by_key, fast)
    try:
        return official.biplane_report(masks, spacing_by_key)
    except Exception as error:  # noqa: BLE001
        return {"method": "official 20-disk biplane Simpson", "edv_ml": None, "esv_ml": None,
                "ef_percent": None, "reason": f"{type(error).__name__}: {error}"}


def _draw_contrast(out_dir: Path, name: str, gray: np.ndarray, gt: np.ndarray,
                   raw: np.ndarray, processed: np.ndarray) -> str:
    """原 native 灰度 + GT + raw + processed 四联对照图（Agg、绘后 close）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from software.role_a_camus import visualize as base_visual

    figure, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("native gray")
    base_visual.contour_overlay(gray, gt, None, axis=axes[1], title="GT")
    base_visual.contour_overlay(gray, None, raw, axis=axes[2], title="raw prediction")
    base_visual.contour_overlay(gray, gt, processed, axis=axes[3], title="processed")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(name)
    path = out_dir / f"{name}.png"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return path.as_posix()


def paired_volume_mae(raw_patients, candidate_patients):
    """Per-field errors on the same finite raw/candidate/GT patient intersection."""
    result = {}
    for field in ("edv_ml", "esv_ml", "ef_percent"):
        ids, raw_errors, candidate_errors = [], [], []
        for patient in sorted(set(raw_patients) & set(candidate_patients)):
            r, c = raw_patients[patient], candidate_patients[patient]
            reports = [r.get('prediction') or {}, r.get('ground_truth') or {},
                       c.get('prediction') or {}, c.get('ground_truth') or {}]
            # EF failure is retained, never converted to a finite error.
            if any(v.get('ef_percent') is None or not np.isfinite(v['ef_percent']) for v in reports):
                continue
            values = [v.get(field) for v in reports]
            if any(v is None or not np.isfinite(v) for v in values):
                continue
            rp, rg, cp, cg = map(float, values)
            if abs(rg-cg) > 1e-9:
                raise PermitRejected('paired ground truth values differ')
            ids.append(patient);raw_errors.append(abs(rp-rg));candidate_errors.append(abs(cp-cg))
        result[field] = {'patients':ids, 'denominator':len(ids),
                         'raw_MAE':float(np.mean(raw_errors)) if ids else None,
                         'candidate_MAE':float(np.mean(candidate_errors)) if ids else None}
    return result


def evaluate_variants(out_dir: Path, *, permit: Mapping[str, Any], max_seconds: int = 900
                      ) -> dict[str, Any]:
    """三版本对比评估：逐帧/逐患者落盘、共同有效 MAE、代表例图件。"""
    started = time.monotonic()
    last_disk_check = started
    stop_file = Path(permit["new_stop_file"]) if permit.get("new_stop_file") else None
    out_dir = Path(out_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    (out_dir / "cases").mkdir(parents=True, exist_ok=True)
    (out_dir / "volumes").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    records = _records()
    cases = _case_files()
    if len(records) != EXPECTED_FRAMES or len(cases) != EXPECTED_FRAMES:
        raise PermitRejected(f"dev 帧数不符: records={len(records)} cases={len(cases)}")
    expected_ids = {f'{patient}_{view}_{phase}' for patient in permit['dev_patients']
                    for view, phase in ENDPOINTS}
    if len({r['frame_id'] for r in records}) != EXPECTED_FRAMES or {r['frame_id'] for r in records} != expected_ids or set(cases) != expected_ids:
        raise PermitRejected('saved dev frames must be exactly50patients x4unique endpoints')
    fast = None
    if permit.get("optimize_geometry") and permit.get("equivalence_checked"):
        sys.path.insert(0, str(FAST_HELPER.parent))
        import contour_diameters_fast as fast  # noqa: PLC0415
    optimize = bool(permit.get("optimize_geometry"))

    rows: dict[str, list[dict[str, Any]]] = {variant: [] for variant in pp.VARIANTS}
    volumes: dict[str, dict[str, Any]] = {variant: {} for variant in pp.VARIANTS}
    changed_total: dict[str, int] = {variant: 0 for variant in pp.VARIANTS}
    changed_rank: list[tuple[int, str, str]] = []
    allowed_patients = set(permit["dev_patients"])

    for index, record in enumerate(records, start=1):
        last_disk_check = _check_budget(started, max_seconds, stop_file, last_disk_check)
        frame_id = str(record.get("frame_id"))
        patient = str(record.get("patient"))
        if patient not in allowed_patients:
            raise PermitRejected(f"帧 {frame_id} 的患者 {patient} 不在许可名单内")
        path = cases.get(frame_id)
        if path is None:
            raise PermitRejected(f"缺少该帧的保存预测: {frame_id}")
        with np.load(path) as payload:
            raw_labels = np.asarray(payload["labels"])
            gt = np.asarray(payload["gt_labels"]) if "gt_labels" in payload else None
        spacing = record.get("spacing_mm")
        if gt is None or not spacing or len(spacing) < 2:
            raise PermitRejected(f"该帧缺少 GT 或 spacing: {frame_id}")
        spacing = (float(spacing[0]), float(spacing[1]))
        outcomes = pp.process_all(raw_labels)
        for variant, outcome in outcomes.items():
            labels = outcome.labels
            np.savez_compressed(out_dir / "cases" / f"{frame_id}_{variant}.npz",
                                labels=labels.astype(np.uint8), gt_labels=gt.astype(np.uint8))
            changed = int((labels != raw_labels).sum())
            changed_total[variant] += changed
            scores = base_metrics.native_frame_scores(
                labels, gt, spacing, quality=str(record.get("quality")), patient=patient,
                view=record.get("view"), phase=record.get("phase"), frame_id=frame_id)
            scores["changed_pixels_vs_raw"] = changed
            scores["postprocess"] = outcome.as_dict()
            rows[variant].append(scores)
            (out_dir / "frames" / f"{frame_id}_{variant}.json").write_text(
                json.dumps(scores, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if variant == "island5_hole1":
                changed_rank.append((changed, frame_id, variant))
        for variant, outcome in outcomes.items():
            volumes[variant].setdefault(patient, {})[f"{record.get('view')}_{record.get('phase')}"] = {
                "spacing_mm": spacing, "pred_lv": outcome.labels == 1, "gt_lv": gt == 1}
        if index % 20 == 0:
            (out_dir / "PROGRESS.json").write_text(json.dumps(
                {"done_frames": index, "total": EXPECTED_FRAMES, "utc": utc_now(),
                 "elapsed_seconds": time.monotonic() - started}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")

    volume_reports: dict[str, dict[str, Any]] = {}
    for variant in pp.VARIANTS:
        per_patient: dict[str, Any] = {}
        for patient, entries in sorted(volumes[variant].items()):
            last_disk_check = _check_budget(started, max_seconds, stop_file, last_disk_check)
            if any(key not in entries for key in [f"{v}_{p}" for v, p in ENDPOINTS]):
                per_patient[patient] = {"prediction": None, "ground_truth": None, "error": None,
                                        "reason": "缺少四张 LV 掩膜"}
            else:
                spacing_by_key = {key: entries[key]["spacing_mm"]
                                  for key in [f"{v}_{p}" for v, p in ENDPOINTS]}
                pred = _official_volume({key: entries[key]["pred_lv"]
                                         for key in spacing_by_key}, spacing_by_key,
                                        fast=fast, optimize=optimize)
                truth = _official_volume({key: entries[key]["gt_lv"]
                                          for key in spacing_by_key}, spacing_by_key,
                                         fast=fast, optimize=optimize)
                error = {}
                for field in ("edv_ml", "esv_ml", "ef_percent"):
                    pv, gv = pred.get(field), truth.get(field)
                    error[field] = (float(pv) - float(gv)
                                    if pv is not None and gv is not None
                                    and np.isfinite(pv) and np.isfinite(gv) else None)
                per_patient[patient] = {"prediction": pred, "ground_truth": truth, "error": error,
                                        "reason": pred.get("reason") or truth.get("reason")}
            (out_dir / "volumes" / f"{variant}_{patient}.json").write_text(
                json.dumps(per_patient[patient], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
        pred_valid = [p for p, entry in per_patient.items()
                      if entry["prediction"] and entry["prediction"].get("ef_percent") is not None]
        gt_valid = [p for p, entry in per_patient.items()
                    if entry["ground_truth"] and entry["ground_truth"].get("ef_percent") is not None]
        common = sorted(set(pred_valid) & set(gt_valid))
        if variant != "raw_reference":
            raw_patients = volume_reports["raw_reference"]["prediction_valid_patients_list"]
            common = sorted(set(common) & set(raw_patients))
        mae: dict[str, float | None] = {}
        for field in ("edv_ml", "esv_ml", "ef_percent"):
            values = [abs(entry["error"][field]) for p, entry in per_patient.items()
                      if p in common and entry["error"] and entry["error"][field] is not None]
            mae[field] = float(np.mean(values)) if values else None
        volume_reports[variant] = {
            "patients": per_patient,
            "prediction_valid_patients": len(pred_valid),
            "prediction_valid_ratio": f"{len(pred_valid)}/{len(per_patient)}",
            "prediction_valid_patients_list": sorted(pred_valid),
            "ground_truth_valid_patients": len(gt_valid),
            "common_valid_patients": len(common),
            "common_valid_denominator": len(common),
            "common_valid_list": common,
            "mae_common_valid": mae,
            "invalid_counts": {"prediction_invalid": len(per_patient) - len(pred_valid),
                               "ground_truth_invalid": len(per_patient) - len(gt_valid)},
        }

    summary: dict[str, Any] = {"schema": "camus-postprocess-eval-v1", "utc": utc_now(),
                               "variants": {}, "changed_pixels_total": changed_total,
                               "frames": len(records), "volumes": volume_reports,
                               "ef_valid_min_required": EF_VALID_MIN,
                               "budget": {"max_seconds": max_seconds,
                                          "project_max_bytes": PROJECT_BUDGET},
                               "reads_test": False, "ran_inference": False}
    raw_primary = None
    for variant in pp.VARIANTS:
        frames = rows[variant]
        aggregate = base_metrics.evaluate_frames(frames)
        aggregate["changed_pixels_total"] = changed_total[variant]
        aggregate["changed_pixels_mean_per_frame"] = changed_total[variant] / len(frames)
        summary["variants"][variant] = aggregate
        with (out_dir / f"RECORDS_{variant}.jsonl").open("w", encoding="utf-8") as handle:
            for frame in frames:
                handle.write(json.dumps(frame, ensure_ascii=False) + "\n")
        if variant == "raw_reference":
            raw_primary = aggregate.get("primary_metric_patient_balanced")

    verdict: dict[str, Any] = {}
    raw_volume = volume_reports["raw_reference"]
    for variant in pp.VARIANTS:
        aggregate = summary["variants"][variant]
        volume = volume_reports[variant]
        dice = aggregate.get("primary_metric_patient_balanced")
        hd95 = (aggregate.get("hd95_mm_frame_balanced") or {}).get("lv_cavity")
        raw_hd95 = (summary["variants"]["raw_reference"].get("hd95_mm_frame_balanced")
                    or {}).get("lv_cavity")
        checks = {
            "dice_not_below_raw": (dice is not None and raw_primary is not None
                                   and dice >= raw_primary - 1e-12),
            "lv_hd95_not_worse_than_raw": (hd95 is not None and raw_hd95 is not None
                                           and hd95 <= raw_hd95 + 1e-9),
            "ef_valid_above_24_of_50": volume["prediction_valid_patients"] > EF_VALID_MIN,
            "common_valid_mae_not_worse": None,
        }
        if variant != "raw_reference":
            paired = paired_volume_mae(raw_volume['patients'], volume['patients'])
            volume['paired_vs_raw'] = paired
            worse = [field for field, item in paired.items()
                     if item['raw_MAE'] is None or item['candidate_MAE'] is None
                     or item['candidate_MAE'] > item['raw_MAE'] + 1e-9]
            checks['common_valid_mae_not_worse'] = not worse
            checks['common_valid_mae_worse_fields'] = worse
        verdict[variant] = checks
    summary["verdict"] = verdict
    summary["acceptance_rule"] = ("patient-balanced native macro Dice >= raw; LV HD95 <= raw; "
                                  "EF valid patients > 24/50; common-valid EDV/ESV/EF MAE not worse")
    summary["all_failures_retained"] = True
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")

    # 代表例 / 最差例 / 改变最多例 四联对照图
    figures: list[str] = []
    worst = min(((frame["foreground_macro_dice"], frame["frame_id"]) for frame in rows["island5_hole1"]
                 if frame.get("foreground_macro_dice") is not None), default=(None, None))
    if changed_rank:
        changed_rank.sort(key=lambda item: (-item[0], item[1]))
        picks = {"most_changed": changed_rank[0][1]}
        if worst[1]:
            picks["worst"] = worst[1]
        valid = [frame for frame in rows["island5_hole1"]
                 if frame.get("foreground_macro_dice") is not None]
        if valid:
            values = sorted(frame["foreground_macro_dice"] for frame in valid)
            median = values[len(values) // 2]
            picks["representative"] = min(valid, key=lambda frame: abs(
                frame["foreground_macro_dice"] - median))["frame_id"]
        for label, frame_id in picks.items():
            path = cases.get(frame_id)
            if path is None:
                continue
            with np.load(path) as payload:
                raw_labels = np.asarray(payload["labels"])
                gt = np.asarray(payload["gt_labels"])
            processed = pp.process(raw_labels, "island5_hole1").labels
            cache_manifest=PROJECT/'runs/role_a_camus/20261012/data/cache04/COMBINED_MANIFEST.json'
            frozen_protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'))
            if sha256_file(cache_manifest)!=frozen_protocol['cache_SHA256']:raise PermitRejected('native gray cache manifest identity mismatch')
            cache_records=json.loads(cache_manifest.read_text(encoding='utf-8'))['records']
            cache_record=next((q for q in cache_records if q['split']=='validation' and f"{q['patient']}_{q['view']}_{q['phase']}"==frame_id),None)
            if cache_record is None:raise PermitRejected('native gray validation image missing')
            cache_file=Path(cache_record['cache_file'])
            if sha256_file(cache_file)!=cache_record['cache_sha256']:raise PermitRejected('native gray cache SHA mismatch')
            cached=base_data.load_cached_frame(cache_file)
            gray=np.asarray(cached['native_image'],dtype=np.float32)
            if gray.shape!=raw_labels.shape or not np.array_equal(cached['native_label'],gt):raise PermitRejected('figure native geometry/GT differs from evaluated case')
            figures.append(_draw_contrast(out_dir / "figures", f"{label}_{frame_id}", gray, gt,
                                          raw_labels, processed))
    summary["figures"] = figures
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    (out_dir / "RESULT.json").write_text(json.dumps({
        "schema": "camus-postprocess-eval-result-v1", "utc": utc_now(), "frames": len(records),
        "primary": {variant: summary["variants"][variant]["primary_metric_patient_balanced"]
                    for variant in pp.VARIANTS},
        "changed_pixels_total": changed_total, "verdict": verdict,
        "elapsed_seconds": time.monotonic() - started, "figures": figures,
        "reads_test": False, "ran_inference": False}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAMUS 后处理变体评估（需 owner 许可；不重推理、不读 test）")
    parser.add_argument("--permit", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-seconds", type=int, default=900)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    permit = load_permit(Path(args.permit))
    max_seconds = min(int(args.max_seconds), int(permit["max_seconds"]))
    out_dir = Path(args.out_dir)
    if Path(permit["out_dir"]) != out_dir:
        raise PermitRejected("命令行 out-dir 必须等于许可绑定的 out_dir")
    ensure_fresh_dir(out_dir)
    out_dir.joinpath("START.json").write_text(json.dumps({
        "schema": "camus-postprocess-eval-start-v1", "utc": utc_now(),
        "permit": Path(args.permit).as_posix(), "out_dir": out_dir.as_posix(),
        "max_seconds": max_seconds,
        "base_source_identity": permit["_base_identity"],
        "opt_source_identity": permit["_opt_identity"]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    summary = evaluate_variants(out_dir, permit=permit, max_seconds=max_seconds)
    print(json.dumps({"variants": list(pp.VARIANTS),
                      "primary": {variant: summary["variants"][variant]
                                  ["primary_metric_patient_balanced"]
                                  for variant in pp.VARIANTS},
                      "verdict": summary["verdict"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
