"""CAMUS 数据集层：官方名单强绑定、端点枚举、封条与缓存。

安全规则（修正上一版"仅靠调用者字符串"的漏洞）：

* 划分只能来自官方名单文件并经 :class:`PatientSplit` 校验；任何"患者 ↔ 路径"不一致、
  跨组或越界，都在 ``nib.load`` **之前**拒绝。
* 测试组 GT 在 owner 一次性放行文件存在前**永远拒绝**；不存在 ``read_test_labels=True`` 快捷开关。
* 只枚举 train/validation 名单 × 2CH/4CH × ED/ES 四端点，并要求每人**恰好 4 帧**；
  未知 ID、错误 token、重复、缺失一律报错；**不按 ID 区间猜划分**。
* 测试患者只清点路径与哈希，不解码 GT。
* 真实样例语义：图像与 GT 均为二维 ``(549, 389)``（末尾 singleton 允许），float32 图像 0..255、
  标签 0..3，spacing 单位 mm、affine 匹配；**真正的序列（多帧）被拒绝**。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import geometry as geo

TRAIN_SPLIT = "training"
DEV_SPLIT = "validation"
TEST_SPLIT = "testing"
SPLIT_FILES = {TRAIN_SPLIT: "subgroup_training.txt", DEV_SPLIT: "subgroup_validation.txt",
               TEST_SPLIT: "subgroup_testing.txt"}
EXPECTED_COUNTS = {TRAIN_SPLIT: 400, DEV_SPLIT: 50, TEST_SPLIT: 50}
ENDPOINTS: tuple[tuple[str, str], ...] = tuple((view, phase) for view in geo.VIEWS
                                               for phase in geo.PHASES)
NUM_CLASSES = 4
TEST_RELEASE_FILE = "OWNER_TEST_RELEASE.json"
# 官方每个患者目录的 12 个 NIfTI：8 个端点（图像 + GT）+ 4 个 half_sequence 文件。
# 端点用于训练/评估；half_sequence 只允许存在、不参与枚举、训练与评估。
FILENAME_PATTERN = re.compile(
    r"^(?P<patient>patient\d{4})_(?P<view>2CH|4CH)_(?P<phase>ED|ES|half_sequence)"
    r"(?P<kind>_gt|)$")
OFFICIAL_KINDS = ("", "_gt", "_half_sequence", "_half_sequence_gt")
HALF_SEQUENCE_KINDS = ("_half_sequence", "_half_sequence_gt")
QUALITY_LEVELS = ("Good", "Medium", "Poor")
QUALITY_CONFIGS = {"2CH": "Info_2CH.cfg", "4CH": "Info_4CH.cfg"}


class SplitError(ValueError):
    """划分名单或端点结构不符合官方约定。"""


class TestSealedError(RuntimeError):
    """测试组 GT 处于封条状态。"""


@dataclass(frozen=True)
class FileIdentity:
    """官方文件名解析结果。"""

    patient: str
    view: str
    phase: str | None
    kind: str

    @property
    def is_endpoint_image(self) -> bool:
        return self.kind == ""

    @property
    def is_endpoint_gt(self) -> bool:
        return self.kind == "_gt"

    @property
    def is_half_sequence(self) -> bool:
        return self.kind in HALF_SEQUENCE_KINDS

    @property
    def endpoint_key(self) -> str:
        return f"{self.view}_{self.phase}"


def parse_filename(path: Path) -> FileIdentity:
    """严格解析官方文件名；未知 token 一律报错（不猜、不按 ID 区间）。"""
    name = Path(path).name
    if not name.endswith(".nii.gz"):
        raise SplitError(f"不是 .nii.gz 文件: {name}")
    stem = name[: -len(".nii.gz")]
    match = FILENAME_PATTERN.match(stem)
    if match is None:
        raise SplitError(f"文件名不符合官方 token 约定: {name}")
    return FileIdentity(patient=match.group("patient"), view=match.group("view"),
                        phase=None if match.group("phase") == "half_sequence" else match.group("phase"),
                        kind=("_half_sequence" + match.group("kind")) if match.group("phase") == "half_sequence" else match.group("kind"))


@dataclass(frozen=True)
class PatientSplit:
    training: tuple[str, ...]
    validation: tuple[str, ...]
    testing: tuple[str, ...]
    source_sha256: Mapping[str, str] = field(default_factory=dict)

    def groups(self) -> dict[str, tuple[str, ...]]:
        return {TRAIN_SPLIT: self.training, DEV_SPLIT: self.validation, TEST_SPLIT: self.testing}

    def validate(self, *, expect: Mapping[str, int] | None = None) -> None:
        expect = dict(expect or EXPECTED_COUNTS)
        for name, members in self.groups().items():
            if len(set(members)) != len(members):
                raise SplitError(f"{name} 名单存在重复患者")
            expected = expect.get(name)
            if expected is not None and len(members) != expected:
                raise SplitError(f"{name} 患者数 {len(members)} != 期望 {expected}")
            for patient in members:
                if not patient.startswith("patient") or not patient[7:].isdigit():
                    raise SplitError(f"{name} 含非法患者 token: {patient!r}")
        seen: dict[str, str] = {}
        for name, members in self.groups().items():
            for patient in members:
                if patient in seen:
                    raise SplitError(f"患者 {patient} 同时出现在 {seen[patient]} 与 {name}")
                seen[patient] = name

    def owner_of(self, patient: str) -> str:
        for name, members in self.groups().items():
            if patient in members:
                return name
        raise SplitError(f"患者 {patient} 不在官方名单中")

    def digest(self) -> str:
        payload = json.dumps({name: sorted(members) for name, members in sorted(self.groups().items())},
                             ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {"counts": {name: len(members) for name, members in self.groups().items()},
                "digest": self.digest(), "source_sha256": dict(self.source_sha256),
                "lists": {name: list(members) for name, members in self.groups().items()}}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_patient_list(text: str) -> tuple[str, ...]:
    """解析官方名单：每行首个 token 的患者名；忽略注释与空行；报错含行号。"""
    patients: list[str] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0].strip().strip("/")
        name = Path(token).name
        if not name.startswith("patient"):
            raise SplitError(f"名单第 {number} 行不是患者 token: {raw!r}")
        patients.append(name)
    if not patients:
        raise SplitError("名单文件未解析出任何患者")
    return tuple(patients)


def load_split(root: Path, *, expect: Mapping[str, int] | None = None) -> PatientSplit:
    """读取官方 ``subgroup_*.txt`` 并校验 400/50/50 与互斥性；记录名单文件 SHA。"""
    root = Path(root)
    groups: dict[str, tuple[str, ...]] = {}
    digests: dict[str, str] = {}
    for split, filename in SPLIT_FILES.items():
        path = root / filename
        if not path.is_file():
            raise SplitError(f"缺少官方名单文件: {path}")
        groups[split] = parse_patient_list(path.read_text(encoding="utf-8"))
        digests[filename] = sha256_file(path)
    split = PatientSplit(training=groups[TRAIN_SPLIT], validation=groups[DEV_SPLIT],
                         testing=groups[TEST_SPLIT], source_sha256=digests)
    split.validate(expect=expect)
    return split


def owner_release_file(camus_root: Path) -> Path:
    return Path(camus_root) / TEST_RELEASE_FILE


def test_release_granted(camus_root: Path) -> bool:
    """测试 GT 放行：仅当 owner 一次性放行文件存在且声明 ``allow_test_gt=true``。"""
    release = owner_release_file(camus_root)
    if not release.is_file():
        return False
    try:
        payload = json.loads(release.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("allow_test_gt") is True


def assert_test_gt_allowed(camus_root: Path, patient: str) -> None:
    """任何测试 GT 读取前调用；未放行立即拒绝。"""
    if not test_release_granted(camus_root):
        raise TestSealedError(f"测试 GT 封条未解除（缺少 {TEST_RELEASE_FILE}），拒绝读取 {patient}")


def expected_paths(patient_dir: Path, patient: str) -> list[tuple[str, str, Path, Path]]:
    """给定患者目录，返回四端点的 ``(view, phase, image_path, gt_path)``。"""
    entries: list[tuple[str, str, Path, Path]] = []
    for view, phase in ENDPOINTS:
        entries.append((view, phase,
                        patient_dir / f"{patient}_{view}_{phase}.nii.gz",
                        patient_dir / f"{patient}_{view}_{phase}_gt.nii.gz"))
    return entries


def validate_patient_dir(patient_dir: Path, patient: str) -> dict[str, Any]:
    """校验患者目录：8 个端点齐全；允许官方 4 个 half_sequence 文件存在但不枚举；
    仅真正未知的文件名才拒绝。返回 ``{"endpoints": [...], "half_sequence": [...], "extra": []}``。
    """
    patient_dir = Path(patient_dir)
    if not patient_dir.is_dir():
        raise SplitError(f"患者目录不存在: {patient_dir}")
    if patient_dir.name != patient:
        raise SplitError(f"目录名 {patient_dir.name} 与患者 {patient} 不一致")
    endpoints = expected_paths(patient_dir, patient)
    missing = [f"{view}_{phase}" for view, phase, image, gt in endpoints
               if not image.is_file() or not gt.is_file()]
    if missing:
        raise SplitError(f"{patient} 缺少端点文件: {missing}")
    half_sequence: list[Path] = []
    for path in sorted(patient_dir.glob("*.nii.gz")):
        identity = parse_filename(path)
        if identity.patient != patient:
            raise SplitError(f"{path.name} 的患者 token 与目录 {patient} 不一致")
        if identity.is_half_sequence:
            half_sequence.append(path)
        elif identity.kind not in ("", "_gt"):
            raise SplitError(f"{path.name} 属于未知 token")
    known = {path.name for _v, _p, image, gt in endpoints for path in (image, gt)} \
        | {path.name for path in half_sequence}
    extra = sorted(path.name for path in patient_dir.glob("*.nii.gz") if path.name not in known)
    if extra:
        raise SplitError(f"{patient} 存在未知 NIfTI 文件: {extra}")
    for view in geo.VIEWS:
        config = patient_dir / QUALITY_CONFIGS[view]
        if not config.is_file():
            raise SplitError(f"{patient} 缺少官方质量配置文件: {config.name}")
    return {"endpoints": endpoints, "half_sequence": half_sequence, "extra": []}


def parse_quality_config(path: Path) -> dict[str, Any]:
    """读取官方 ``Info_2CH.cfg`` / ``Info_4CH.cfg`` 原字段（含 ImageQuality 与 spacing）。"""
    path = Path(path)
    if not path.is_file():
        raise SplitError(f"质量配置文件不存在: {path}")
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        values[key.strip()] = value.strip()
    quality = values.get("ImageQuality")
    if quality not in QUALITY_LEVELS:
        raise SplitError(f"{path.name} 的 ImageQuality 非法/缺失: {quality!r}")
    spacing: tuple[float, float] | None = None
    for key in ("PixelSpacing", "Spacing", "VoxelSpacing"):
        if key in values:
            numbers = [float(token) for token in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?",
                                                            values[key])]
            if len(numbers) >= 2:
                spacing = (numbers[0], numbers[1])
                break
    return {"path": path.as_posix(), "sha256": sha256_file(path), "ImageQuality": quality,
            "spacing_cfg": spacing, "fields": dict(sorted(values.items()))}


def patient_quality(patient_dir: Path) -> dict[str, Any]:
    """两个视角的官方质量字段（Good/Medium/Poor）；缺失即报错，不用亮度替代。"""
    result: dict[str, Any] = {}
    for view, filename in QUALITY_CONFIGS.items():
        result[view] = parse_quality_config(Path(patient_dir) / filename)
    return result


@dataclass
class CamusFrame:
    """一个端点的缓存条目：**保留原生数据**，模型输入是额外的 letterbox 视图。"""

    patient: str
    split: str
    view: str
    phase: str
    image: np.ndarray
    valid: np.ndarray
    label: np.ndarray | None
    meta: dict[str, Any]
    spacing_mm: tuple[float, float]
    native_image: np.ndarray
    native_label: np.ndarray | None
    native_shape: tuple[int, int]
    affine: tuple[tuple[float, ...], ...]
    source_sha256: Mapping[str, str]
    quality: str = "unknown"
    quality_source: str | None = None
    histogram: Mapping[str, int] = field(default_factory=dict)
    half_sequence_present: bool = False

    def as_record(self) -> dict[str, Any]:
        return {"patient": self.patient, "split": self.split, "view": self.view, "phase": self.phase,
                "native_shape": list(self.native_shape), "spacing_mm": list(self.spacing_mm),
                "letterbox_shape": [int(self.image.shape[0]), int(self.image.shape[1])],
                "affine": [list(row) for row in self.affine],
                "quality": self.quality, "quality_source": self.quality_source,
                "histogram": dict(self.histogram),
                "source_sha256": dict(self.source_sha256), "letterbox": dict(self.meta),
                "half_sequence_present": bool(self.half_sequence_present)}


def _load_nifti_array(path: Path) -> tuple[np.ndarray, Any]:
    import nibabel as nib

    handle = nib.load(str(path))
    array = np.asarray(handle.dataobj)
    return array, handle


def _require_2d(array: np.ndarray, what: str) -> np.ndarray:
    squeezed = np.squeeze(np.asarray(array))
    if squeezed.ndim != 2:
        raise SplitError(f"{what} 不是可用的二维端点（shape={array.shape}）；序列帧请另行走时序流程")
    return squeezed


def quality_tag(image: np.ndarray) -> str:
    """**已停用**：亮度分档不是官方质量标签。

    正式质量必须来自 ``Info_2CH.cfg`` / ``Info_4CH.cfg`` 的 ``ImageQuality`` 原字段
    （见 :func:`patient_quality`）。保留本函数只为在误用时立刻暴露问题。
    """
    raise SplitError("quality_tag 已停用：请使用官方 Info_*.cfg 的 ImageQuality（patient_quality）")


def verify_split_binding(camus_root: Path, split_object: PatientSplit) -> None:
    """强绑定：名单必须与 ``camus_root/database_split`` 下的官方文件逐一 SHA 相符。

    调用者无法通过临时构造的名单绕过（含测试封条）。
    """
    split_root = Path(camus_root) / "database_split"
    if not split_object.source_sha256:
        raise SplitError("PatientSplit 缺少官方来源 SHA，拒绝使用（必须由 load_split 产生）")
    for split, filename in SPLIT_FILES.items():
        path = split_root / filename
        if not path.is_file():
            raise SplitError(f"缺少官方名单文件: {path}")
        recorded = split_object.source_sha256.get(filename)
        if recorded is None:
            raise SplitError(f"PatientSplit 未记录 {filename} 的 SHA")
        if recorded != sha256_file(path):
            raise SplitError(f"{filename} 与磁盘 SHA 不一致，拒绝使用该名单")
    if list(split_object.groups()) != [TRAIN_SPLIT, DEV_SPLIT, TEST_SPLIT]:
        raise SplitError("PatientSplit 结构异常")
    split_object.validate()


def load_frame(image_path: Path, gt_path: Path | None, *, split: str, patient: str,
               split_object: PatientSplit, camus_root: Path, size: int = 256,
               num_classes: int = NUM_CLASSES) -> CamusFrame:
    """读取一个端点：**原生图像/GT 与几何全部保留**，另生成 letterbox 模型输入。

    顺序（全部在 ``nib.load`` 之前）：split 合法 → 与实际官方名单强绑定（含源 SHA 复核）
    → 患者归属 → 文件名严格解析 → image/GT 的 view/phase/patient 完全一致 →
    目录与 camus_root 一致 → 官方质量字段 → 测试封条 → 才读取数据。
    """
    if split not in (TRAIN_SPLIT, DEV_SPLIT, TEST_SPLIT):
        raise SplitError(f"未知 split: {split!r}")
    camus_root = Path(camus_root)
    verify_split_binding(camus_root, split_object)
    owner = split_object.owner_of(patient)
    if owner != split:
        raise SplitError(f"患者 {patient} 属 {owner}，拒绝以 {split} 读取（跨组/越界）")
    image_path = Path(image_path)
    gt_path = Path(gt_path) if gt_path is not None else None
    nifti_root = camus_root / "database_nifti"
    if image_path.parent.resolve() != (nifti_root / patient).resolve():
        raise SplitError(f"图像不在官方患者目录内: {image_path}")
    image_identity = parse_filename(image_path)
    if not image_identity.is_endpoint_image or image_identity.patient != patient:
        raise SplitError(f"图像 token 非法或患者不符: {image_path.name}")
    if gt_path is not None:
        gt_identity = parse_filename(gt_path)
        if not gt_identity.is_endpoint_gt or gt_identity.patient != patient:
            raise SplitError(f"GT token 非法或患者不符: {gt_path.name}")
        if (gt_identity.view, gt_identity.phase) != (image_identity.view, image_identity.phase):
            raise SplitError("image/GT 的 view/phase 不匹配: "
                             f"{image_path.name} vs {gt_path.name}")
    if split == TEST_SPLIT:
        assert_test_gt_allowed(camus_root, patient)
    quality = patient_quality(nifti_root / patient)
    quality_value = quality[image_identity.view]["ImageQuality"]
    listing = validate_patient_dir(nifti_root / patient, patient)
    array, handle = _load_nifti_array(image_path)
    data = _require_2d(array, f"{image_path.name}")
    geometry = geo.geometry_from_nifti(handle.header, patient=patient,
                                       phase=image_identity.phase, view=image_identity.view,
                                       source_path=image_path.as_posix())
    display = geo.to_display_array(data)
    normalized = geo.normalize_gray(display)
    canvas, valid, meta = geo.letterbox(normalized, size=size)
    source_hashes = {"image": sha256_file(image_path)}
    native_label = None
    label_canvas = None
    histogram: dict[str, int] = {}
    if gt_path is not None:
        gt_array, gt_handle = _load_nifti_array(gt_path)
        gt_data = _require_2d(gt_array, f"{gt_path.name}")
        if not np.allclose(np.asarray(handle.affine), np.asarray(gt_handle.affine)):
            raise SplitError(f"图像与 GT affine 不一致: {image_path.name}")
        if gt_data.shape != data.shape:
            raise SplitError(f"图像与 GT shape 不一致: {data.shape} vs {gt_data.shape}")
        native_label = geo.to_display_array(gt_data).astype(np.int64)
        geo.validate_label(native_label, num_classes=num_classes)
        label_canvas = geo.letterbox_labels(native_label, meta, num_classes=num_classes)
        source_hashes["gt"] = sha256_file(gt_path)
        histogram = {name: int((native_label == index).sum())
                     for index, name in geo.CLASS_NAMES.items()}
    return CamusFrame(patient=patient, split=split, view=image_identity.view,
                      phase=image_identity.phase, image=canvas, valid=valid, label=label_canvas,
                      meta=meta, spacing_mm=(geometry.spacing_y_mm, geometry.spacing_x_mm),
                      native_image=normalized, native_label=native_label,
                      native_shape=(normalized.shape[0], normalized.shape[1]),
                      affine=tuple(tuple(float(v) for v in row) for row in np.asarray(handle.affine)),
                      source_sha256=source_hashes, quality=quality_value,
                      quality_source=quality[image_identity.view]["path"], histogram=histogram,
                      half_sequence_present=bool(listing["half_sequence"]))


def enumerate_split(camus_root: Path, split: str, patient_split: PatientSplit, *,
                    verify_endpoints: bool = True) -> dict[str, Any]:
    """只按本名单枚举 **8 个端点**（4 视图/时相 × 图像+GT），校验每人 4 端点齐全。

    官方 half_sequence 文件允许存在但**不进入枚举**；测试组只清点路径与哈希。
    """
    camus_root = Path(camus_root)
    verify_split_binding(camus_root, patient_split)
    members = {TRAIN_SPLIT: patient_split.training, DEV_SPLIT: patient_split.validation,
               TEST_SPLIT: patient_split.testing}.get(split)
    if members is None:
        raise SplitError(f"未知 split: {split!r}")
    nifti_root = camus_root / "database_nifti"
    found: list[dict[str, Any]] = []
    half_sequence: list[dict[str, Any]] = []
    for patient in members:
        patient_dir = nifti_root / patient
        if not patient_dir.is_dir():
            raise SplitError(f"{split} 名单中的患者目录缺失: {patient_dir}")
        listing = validate_patient_dir(patient_dir, patient) if verify_endpoints else {
            "endpoints": expected_paths(patient_dir, patient),
            "half_sequence": sorted(patient_dir.glob("*half_sequence*.nii.gz"))}
        if len(listing["endpoints"]) != 4:
            raise SplitError(f"{patient} 端点数 {len(listing['endpoints'])} != 4")
        for view, phase, image, gt in listing["endpoints"]:
            image_identity, gt_identity = parse_filename(image), parse_filename(gt)
            if (image_identity.view, image_identity.phase) != (view, phase) or \
                    (gt_identity.view, gt_identity.phase) != (view, phase):
                raise SplitError(f"{patient} 端点 token 与预期不符: {view}_{phase}")
            if split == TEST_SPLIT and not test_release_granted(camus_root):
                found.append({"patient": patient, "view": view, "phase": phase,
                              "image": image.as_posix(), "gt": gt.as_posix(),
                              "image_sha256": sha256_file(image), "gt_sha256": None,
                              "gt_sealed": True})
            else:
                found.append({"patient": patient, "view": view, "phase": phase,
                              "image": image.as_posix(), "gt": gt.as_posix(),
                              "image_sha256": sha256_file(image), "gt_sha256": sha256_file(gt),
                              "gt_sealed": False})
        for path in listing["half_sequence"]:
            half_sequence.append({"patient": patient, "path": path.as_posix(),
                                  "sha256": sha256_file(path), "used_for_training": False})
    if verify_endpoints and len(found) != 4 * len(members):
        raise SplitError(f"{split} 端点总数 {len(found)} != 4×{len(members)}")
    return {"split": split, "patients": len(members), "frames": len(found),
            "half_sequence_files": len(half_sequence), "entries": found,
            "half_sequence": half_sequence,
            "note": "half_sequence 文件仅清点，不参与枚举/训练/评估；测试 GT 未放行时不计算 SHA"}


def patient_level_index(frames: Iterable[CamusFrame]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for position, frame in enumerate(frames):
        index.setdefault(frame.patient, []).append(position)
    return index


def assert_no_patient_cross_split(split: PatientSplit, assignments: Mapping[str, str]) -> None:
    """Each actual patient must be assigned to its one official split."""
    for patient, assigned in assignments.items():
        if not isinstance(assigned, str) or assigned not in (TRAIN_SPLIT, DEV_SPLIT, TEST_SPLIT):
            raise SplitError(f"Invalid or multiple split assignment for {patient}: {assigned!r}")
        owner = split.owner_of(patient)
        if owner != assigned:
            raise SplitError(f"Patient {patient} belongs to {owner}, not {assigned}")



def cache_frames(frames: Sequence[CamusFrame], out_dir: Path, *, config: Mapping[str, Any] | None = None
                 ) -> dict[str, Any]:
    """缓存**原生**图像/GT（原尺寸、原 spacing、完整 affine）与 letterbox 输入，并写审计清单。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for frame in frames:
        payload: dict[str, Any] = {
            "image": frame.image.astype(np.float32),
            "valid": frame.valid.astype(np.float32),
            "native_image": frame.native_image.astype(np.float32),
            "meta": np.array(json.dumps(frame.meta), dtype=object),
            "spacing_mm": np.asarray(frame.spacing_mm, dtype=np.float64),
            "affine": np.asarray(frame.affine, dtype=np.float64),
            "quality_source": np.array(frame.quality_source or "", dtype=object)}
        if frame.native_label is not None:
            payload["native_label"] = frame.native_label.astype(np.int16)
        if frame.label is not None:
            payload["label"] = frame.label.astype(np.int16)
        name = f"{frame.patient}_{frame.view}_{frame.phase}.npz"
        path = out_dir / name
        np.savez_compressed(path, **payload)
        record = frame.as_record()
        record.update({"cache_file": path.as_posix(), "cache_sha256": sha256_file(path),
                       "cache_bytes": path.stat().st_size,
                       "native_pixels": int(frame.native_image.size),
                       "native_label_present": frame.native_label is not None})
        records.append(record)
    manifest = {"schema": "camus-cache-manifest-v2", "frames": len(records),
                "config": dict(config or {}), "records": records, "native_gt_retained": True,
                "note": "保留原生 GT/图像/spacing/affine；指标必须基于 native GT，"
                        "禁止把 256 标签逆 resize 冒充原 GT"}
    (out_dir / "CACHE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_cached_frame(path: Path) -> dict[str, Any]:
    """读取缓存条目（原生图像/GT、spacing、affine、quality 来源与 letterbox 元数据）。"""
    with np.load(Path(path), allow_pickle=True) as payload:
        return {"image": payload["image"], "valid": payload["valid"],
                "native_image": payload["native_image"],
                "meta": json.loads(str(payload["meta"])),
                "spacing_mm": tuple(float(v) for v in payload["spacing_mm"]),
                "affine": payload["affine"].tolist(),
                "quality_source": (str(payload["quality_source"])
                                   if "quality_source" in payload else None),
                "label": payload["label"] if "label" in payload else None,
                "native_label": payload["native_label"] if "native_label" in payload else None}
