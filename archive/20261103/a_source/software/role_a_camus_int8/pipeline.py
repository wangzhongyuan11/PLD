"""CAMUS INT8 流水线：校准（固定 roster，只读 image）→ 导出（CAMUS v1）→ 校验（verify/loader/CLI）。

只**适配** owner 已审的 :mod:`integer` 与 :mod:`graph` 稳定接口，不复制其数学。

导出产物（每层一个目录，共 15 个）：``weight.bin``(int8 OIHW)、``bias.bin``(<i4)、
``weight_scales.npy``(float64 per-channel)、``multipliers.bin``(<i4)、``tile/{input.bin 8x8,
input_full.bin, acc_golden.bin <i4, output_golden.bin fused-relu s8, output_full_golden.bin}``、
``layer.json``（shape/dtype/bytes/SHA/packed 连续偏移）。另有 ``concat/level*/``（skip/up 原始、
各 rescale、skip-first merged、info.json）、``full_sample.json``（仅 JSON 元数据，无 ndarray）、
``model_spec.json``（绑定 code identity / protocol / roster / cache / 源）。

``source_kind``：``synthetic`` 仅用于 CPU 合成检查（**不读真实 checkpoint/数据、不贴真实源 SHA**）；
``frozen`` 才加载真实 checkpoint（``weights_only=True``）并做批准与身份核验。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

PROJECT = Path(r"E:/嵌赛/pld-medical")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from software.role_a_camus import geometry as base_geo            # noqa: E402
from software.role_a_camus_int8 import graph as i8_graph          # noqa: E402
from software.role_a_camus_int8 import integer as i8              # noqa: E402

RUN = PROJECT / "runs/role_a_camus/20261012"
RUN_INT8 = PROJECT / "runs/role_a_camus/20261012_int8_01"
PROTOCOL = RUN_INT8 / "PROTOCOL.json"
ROSTER = RUN_INT8 / "CALIBRATION_ROSTER.json"
CACHE_MANIFEST = RUN / "data/cache04/COMBINED_MANIFEST.json"
FROZEN_BEST = RUN / "final_selection/FROZEN_BEST.pt"
FP32_DELIVERY = PROJECT / "deliverables/role_a_camus_20261012/model_state_fp32.pt"
EXPECTED_CHECKPOINT_SHA = "dcfe16a59cdde4821180561aa2d8c72393f1f53e6c8bbbcdade0541413a1e616"
EXPECTED_FP32_DELIVERY_SHA = "74d0da6f13b48528b221e6c295d90de73b6e16f1320a860bc086a05e3c2b774f"
EXPECTED_ROSTER_SHA = "a81b739655e32ef1dcfc9ea8e4e995e9d553c99cb28435eba7c597960062936a"
EXPECTED_CACHE_SHA = "3112df3f46856ec36a5b42a6fa1752063d8851ef982c07c9b9fec5769cb4b217"
EXPORT_SCHEMA = "camus-int8-export-v1"
CALIBRATION_FRAMES = 128
CALIBRATION_PATIENTS = 32
TOTAL_CONVS = 15
HIDDEN_CONVS = 14
TILE = 8
LAYER_NAMES = i8_graph.LAYER_NAMES
CONCAT_SKIP_UP = ((2, 64, 128), (1, 32, 64), (0, 16, 32))   # (level, skip_ch, up_ch)
SOURCE_KINDS = ("synthetic", "frozen")
GUARD = None
def guard():
    if GUARD is not None:GUARD()


class PipelineError(RuntimeError):
    """前置条件不满足（身份、路径、SHA、形状等）。"""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def code_identity() -> dict[str,str]:
    identity={f'int8/{p.name}':sha256_file(p) for p in sorted((PROJECT/'software/role_a_camus_int8').glob('*.py'))}
    identity.update({f'base/{p.name}':sha256_file(p) for p in sorted((PROJECT/'software/role_a_camus').glob('*.py'))})
    identity['role_a/quant_math.py']=sha256_file(PROJECT/'software/role_a/quant_math.py')
    for p in sorted((RUN_INT8/'execution').glob('*.py')):identity[f'execution/{p.name}']=sha256_file(p)
    for n in ('final_eval.py','contour_diameters_fast.py'):
        q=RUN/'execution/final_eval01'/n
        if q.is_file():identity[f'helper/{n}']=sha256_file(q)
    return identity


# --------------------------------------------------------------------------- 模型与校准

def conv_modules(model: nn.Module) -> list[tuple[str, nn.Conv2d]]:
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, nn.Conv2d)]


def load_fp32_model(checkpoint: Path = FROZEN_BEST, *, source_kind: str = "frozen"
                    ) -> tuple[nn.Module, dict[str, Any]]:
    """CAMUS 专用加载器：``source_kind=frozen`` 时严格核验身份并纯 FP32 加载。"""
    if source_kind != "frozen":
        raise PipelineError("load_fp32_model 只允许 source_kind='frozen'")
    from software.role_a_camus import model as camus_model

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise PipelineError(f"检查点不存在: {checkpoint}")
    digest = sha256_file(checkpoint)
    if digest != EXPECTED_CHECKPOINT_SHA:
        raise PipelineError(f"检查点 SHA 不符: {digest}")
    if not PROTOCOL.is_file() or sha256_file(ROSTER) != EXPECTED_ROSTER_SHA:
        raise PipelineError("protocol/roster 绑定异常")
    if not CACHE_MANIFEST.is_file() or sha256_file(CACHE_MANIFEST) != EXPECTED_CACHE_SHA:
        raise PipelineError("cache manifest 绑定异常")
    if not FP32_DELIVERY.is_file() or sha256_file(FP32_DELIVERY)!=EXPECTED_FP32_DELIVERY_SHA:raise PipelineError("pure FP32 source identity failed")
    payload = torch.load(FP32_DELIVERY, map_location="cpu", weights_only=True)
    if payload.get("selection",{}).get("epoch")!=94:raise PipelineError("pure FP32 source must be frozen epoch94")
    config = dict(payload.get("config") or {})
    if (int(config.get("base_channels", 0)), int(config.get("num_pools", 0)),
            int(config.get("size", 0))) != (16, 3, 256):
        raise PipelineError(f"期望 base16/3pool/256，收到 {config}")
    model = camus_model.build_model(base_channels=16, num_pools=3, size=256)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    tensors = list(model.state_dict().values())
    if len(tensors) != 30 or any(t.dtype != torch.float32 or not bool(torch.isfinite(t).all())
                                 for t in tensors):
        raise PipelineError("模型必须为 30 个有限 FP32 张量")
    if len(conv_modules(model)) != TOTAL_CONVS:
        raise PipelineError(f"Conv 层数 {len(conv_modules(model))} != {TOTAL_CONVS}")
    if not FP32_DELIVERY.is_file() or sha256_file(FP32_DELIVERY) != EXPECTED_FP32_DELIVERY_SHA:
        raise PipelineError("FP32 交付 SHA 不符")
    delivery=torch.load(FP32_DELIVERY,map_location="cpu",weights_only=True)["model_state"]
    if set(delivery)!=set(model.state_dict()) or any(not torch.equal(delivery[n],v) for n,v in model.state_dict().items()):raise PipelineError("frozen and delivered FP32 tensors differ")
    return model, config


def load_roster(path: Path = ROSTER) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise PipelineError(f"缺少校准名单: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "camus-int8-calibration-roster-v1":
        raise PipelineError(f"名单 schema 非法: {payload.get('schema')!r}")
    records = payload.get("records") or []
    if len(records) != CALIBRATION_FRAMES:
        raise PipelineError(f"名单帧数 {len(records)} != {CALIBRATION_FRAMES}")
    if len({record["patient"] for record in records}) != CALIBRATION_PATIENTS:
        raise PipelineError("名单患者数必须为 32")
    endpoints = {("2CH", "ED"), ("2CH", "ES"), ("4CH", "ED"), ("4CH", "ES")}
    for record in records:
        if (record["view"], record["phase"]) not in endpoints:
            raise PipelineError(f"非法端点: {record}")
        if "label" in record:
            raise PipelineError("名单不得包含 label 字段")
    return payload


def load_calibration_images(roster: Mapping[str, Any], *, verify_hashes: bool = True
                            ) -> list[tuple[str, np.ndarray]]:
    """只读 ``image`` 键（**不读 label**、不增强），逐帧核对缓存 SHA。"""
    images: list[tuple[str, np.ndarray]] = []
    for record in roster["records"]:
        guard()
        if record.get("split")!="training":raise PipelineError("calibration requires training split")
        path = Path(record["cache_file"])
        if not path.is_file():
            raise PipelineError(f"缓存缺失: {path}")
        if verify_hashes and sha256_file(path) != record["cache_sha256"]:
            raise PipelineError(f"缓存 SHA 不符: {path.name}")
        with np.load(path, allow_pickle=False) as payload:
            if "image" not in payload:
                raise PipelineError(f"缓存缺少 image 键: {path.name}")
            image = np.asarray(payload["image"], dtype=np.float32)
        images.append((f"{record['patient']}_{record['view']}_{record['phase']}", image))
    if len(images) != CALIBRATION_FRAMES:
        raise PipelineError(f"实际载入 {len(images)} 帧 != {CALIBRATION_FRAMES}")
    return images


def calibrate(model: nn.Module, images: Sequence[tuple[str, np.ndarray]]) -> dict[str, Any]:
    """逐层 hook：hidden Conv 取 fused ReLU 后 ``max|·|``；head 取 4 logits 全局 ``max|·|``。"""
    convs = conv_modules(model)
    if [name for name, _m in convs] != list(LAYER_NAMES):
        raise PipelineError("CAMUS 必须为精确的 15-Conv 拓扑")
    stats = {name: {"input_max_abs": 0.0, "output_max_abs": 0.0} for name in LAYER_NAMES[:-1]}
    head_stats = {"input_max_abs": 0.0, "output_max_abs": 0.0}
    handles = []
    for name, module in convs[:-1]:
        def before(_module, arguments, key: str = name) -> None:
            stats[key]["input_max_abs"] = max(stats[key]["input_max_abs"],
                                              float(arguments[0].abs().amax()))
        def after(_module, _arguments, output, key: str = name) -> None:
            clamped = output.clamp(min=0)
            stats[key]["output_max_abs"] = max(stats[key]["output_max_abs"],
                                               float(clamped.abs().amax()))
        handles.extend((module.register_forward_pre_hook(before),
                        module.register_forward_hook(after)))
    head_module = convs[-1][1]
    def head_before(_module, arguments) -> None:
        head_stats["input_max_abs"] = max(head_stats["input_max_abs"],
                                          float(arguments[0].abs().amax()))
    def head_after(_module, _arguments, output) -> None:
        head_stats["output_max_abs"] = max(head_stats["output_max_abs"],
                                           float(output.abs().amax()))
    handles.extend((head_module.register_forward_pre_hook(head_before),
                    head_module.register_forward_hook(head_after)))
    try:
        with torch.no_grad():
            for _name, image in images:
                model(torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None])
    finally:
        for handle in handles:
            handle.remove()
    return {"input_scale": i8.INPUT_SCALE, "layers": stats, "head": head_stats,
            "frames": len(images), "head_name": convs[-1][0]}


def scales_from_calibration(model: nn.Module, calibration: Mapping[str, Any]):
    """**委托** owner 已审的 :func:`graph.scales_from_calibration`（concat 显式取两支 max）。"""
    return i8_graph.scales_from_calibration(model, calibration)


def build_layers(model: nn.Module, calibration: Mapping[str, Any]) -> list[i8.QuantLayer]:
    input_scales, output_scales, _head = scales_from_calibration(model, calibration)
    layers = []
    for name, module in conv_modules(model):
        layers.append(i8.build_layer(module, name, input_scale=input_scales[name],
                                     output_scale=output_scales[name]))
    if [layer.name for layer in layers] != list(LAYER_NAMES):
        raise PipelineError("层名与 owner 已审拓扑不一致")
    return layers


# --------------------------------------------------------------------------- 导出

def _safe_dir(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not out_dir.is_dir() or any(out_dir.iterdir()):
            raise PipelineError(f"导出目录必须全新（允许已存在空目录）: {out_dir}")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_array(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.ascontiguousarray(array).tobytes(order="C"))
    return {"file": path.name, "bytes": int(path.stat().st_size), "dtype": str(array.dtype),
            "shape": list(array.shape), "sha256": sha256_file(path)}


def _layer_dir_name(index: int, name: str) -> str:
    return f"{index:02d}_{name.replace('.', '_')}"


def _oracle_tile(input_tile: np.ndarray, layer: i8.QuantLayer, *, relu: bool
                 ) -> tuple[np.ndarray, np.ndarray]:
    """8×8 tile 的 same-padding 真整数 oracle（不误用 full 边界卷积）。"""
    acc = i8.qm.conv2d_int32(input_tile, layer.q_weight, layer.q_bias, layer.padding)
    output = i8.qm.requantize_int8(acc, layer.multiplier)
    if relu:
        output = np.maximum(output, 0).astype(np.int8)
    return acc, output


def export_graph(model: nn.Module, calibration: Mapping[str, Any],
                 layers: Sequence[i8.QuantLayer], out_dir: Path, *,
                 sample_image: np.ndarray, source_kind: str = "synthetic",
                 source_meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """导出 CAMUS 独立 v1 schema（逐层目录 + tile 黄金 + 3 concat 黄金 + full 例元数据）。"""
    if source_kind not in SOURCE_KINDS:
        raise PipelineError(f"source_kind 必须为 {SOURCE_KINDS}")
    if tuple(layer.name for layer in layers) != tuple(LAYER_NAMES):
        raise PipelineError("必须提供 owner 已审的 15 层顺序")
    if source_kind=="frozen" and (not source_meta or source_meta.get("approved") is not True):raise PipelineError("frozen export requires approved provenance")
    out_dir = _safe_dir(out_dir)
    sample = np.asarray(sample_image, dtype=np.float32)
    if sample.ndim != 2 or min(sample.shape) < TILE or any(size % 8 for size in sample.shape):
        raise PipelineError("样本尺寸必须是 8 的倍数且 ≥ 8")

    weight_offset = bias_offset = 0
    layer_records: list[dict[str, Any]] = []
    for index, layer in enumerate(layers):
        guard()
        directory = out_dir / _layer_dir_name(index, layer.name)
        directory.mkdir(parents=True, exist_ok=True)
        weight_info = _write_array(directory / "weight.bin", layer.q_weight.astype(np.int8))
        bias_info = _write_array(directory / "bias.bin", layer.q_bias.astype("<i4"))
        scales = (np.asarray(layer.weight_scales, dtype=np.float64)
                  if layer.weight_scales is not None
                  else i8.qm.quantize_weights(layer.q_weight)[1].astype(np.float64))
        np.save(directory / "weight_scales.npy", scales)
        multiplier_info = _write_array(directory / "multipliers.bin",
                                      layer.multiplier.astype("<i4"))
        bound = i8.require_bounds(layer.q_weight, layer.q_bias)
        if not np.array_equal(bound, layer.conservative_bound):
            raise PipelineError(f"{layer.name}: 保守界不一致")
        record = {"index": index, "name": layer.name, "directory": directory.name,
                  "kernel": int(layer.kernel), "padding": int(layer.padding),
                  "input_scale": float(layer.input_scale), "output_scale": float(layer.output_scale),
                  "weight_shape": list(layer.q_weight.shape), "bias_shape": list(layer.q_bias.shape),
                  "weight_scales": scales.tolist(),
                  "multiplier_q24": layer.multiplier.astype(int).tolist(),
                  "conservative_bound_max": int(bound.max()), "bound_ok": True,
                  "weight_file": weight_info, "bias_file": bias_info,
                  "multiplier_file": multiplier_info,
                  "weight_scales_file": {"file": "weight_scales.npy",
                                         "bytes": (directory / "weight_scales.npy").stat().st_size,
                                         "dtype": "float64", "shape": [len(scales)],
                                         "sha256": sha256_file(directory / "weight_scales.npy")},
                  "packed_weight_offset": weight_offset,
                  "packed_weight_count": int(layer.q_weight.size),
                  "packed_bias_offset": bias_offset,
                  "packed_bias_count": int(layer.q_bias.size)}
        weight_offset += int(layer.q_weight.size)
        bias_offset += int(layer.q_bias.size)
        layer_records.append(record)

    result = i8_graph.forward_graph(model, layers, sample, valid=np.ones_like(sample, dtype=np.float32),
                                    capture=True, oracle=False)
    captures = result["captures"]
    full_shapes: dict[str, list[int]] = {}
    for index, layer in enumerate(layers):
        guard()
        directory = out_dir / _layer_dir_name(index, layer.name)
        capture = captures.get(layer.name, {})
        if "input_int8" not in capture:
            raise PipelineError(f"{layer.name}: 缺失输入捕获")
        full_input = np.asarray(capture["input_int8"], dtype=np.int8)
        captured_acc = np.asarray(capture["acc_int32"])
        captured_output = np.asarray(capture["output_int8"])
        relu = index < HIDDEN_CONVS
        tile_input = np.ascontiguousarray(full_input[:, :, :TILE, :TILE])
        tile_acc, tile_output = _oracle_tile(tile_input, layer, relu=relu)
        # A standalone tile has zero padding on its right/bottom border; it must
        # not be compared with the crop of a convolution of the larger image.
        tile_fast=i8.conv2d_int8_float64(i8.Int8Feature(tile_input,layer.input_scale),layer,relu=relu,oracle=True)
        if not np.array_equal(tile_fast.data,tile_output):raise PipelineError('tile simulator mismatch')
        tile_dir = directory / "tile"
        goldens = {"input": _write_array(tile_dir / "input.bin", tile_input),
                   "input_full": _write_array(tile_dir / "input_full.bin", full_input),
                   "acc_golden": _write_array(tile_dir / "acc_golden.bin", tile_acc.astype("<i4")),
                   "output_golden": _write_array(tile_dir / "output_golden.bin", tile_output),
                   "output_full_golden": _write_array(tile_dir / "output_full_golden.bin",
                                                      captured_output.astype(np.int8))}
        layer_records[index]["tile"] = goldens
        layer_records[index]["tile_relu"] = bool(relu)
        layer_records[index]["full_input_shape"] = list(full_input.shape)
        layer_records[index]["full_output_shape"] = list(captured_output.shape)
        full_shapes[layer.name] = list(full_input.shape)
        (directory / "layer.json").write_text(
            json.dumps(layer_records[index], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    concat_records = []
    for metadata in result["concat_info"]:
        guard()
        info={**metadata,**captures[f"concat_level{metadata['level']}"]}
        directory = out_dir / "concat" / f"level{info['level']}"
        directory.mkdir(parents=True, exist_ok=True)
        files = {"skip_int8": _write_array(directory / "skip_int8.bin",
                                          np.asarray(info["skip_int8"], dtype=np.int8)),
                 "up_int8": _write_array(directory / "up_int8.bin",
                                        np.asarray(info["up_int8"], dtype=np.int8)),
                 "skip_rescaled": _write_array(directory / "skip_rescaled.bin",
                                               np.asarray(info["skip_rescaled_int8"], dtype=np.int8)),
                 "up_rescaled": _write_array(directory / "up_rescaled.bin",
                                             np.asarray(info["up_rescaled_int8"], dtype=np.int8)),
                 "merged_int8": _write_array(directory / "merged_int8.bin",
                                            np.asarray(info["output_int8"], dtype=np.int8))}
        payload = {"level": info["level"], "decoder": info["decoder"], "skip_first": True,
                   "source_scales": info["source_scales"], "common_scale": info["common_scale"],
                   "multiplier_q24": info["multiplier_q24"],
                   "output_channels": info["output_channels"], "files": files}
        (directory / "info.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
        concat_records.append(payload)

    quant_logits = np.asarray(result["quant_logits"], dtype=np.int8)
    entry_input = np.asarray(captures[LAYER_NAMES[0]]["input_int8"])
    full_sample = {"input_scale": float(result["input_scale"]),
                   "head_output_scale": float(result["head_output_scale"]),
                   "input_int8_shape": list(entry_input.shape),
                   "input_int8_head256": entry_input.astype(int).reshape(-1).tolist()[:256],
                   "final_logits_int8_shape": list(quant_logits.shape),
                   "final_logits_int8_head256": quant_logits.astype(int).reshape(-1).tolist()[:256],
                   "layer_input_shapes": full_shapes,
                   "note": "full 例仅 JSON 元数据；逐层 .bin 为权威黄金，无 ndarray 内联"}
    (out_dir / "full_sample.json").write_text(json.dumps(full_sample, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    source: dict[str, Any] = {"kind": source_kind}
    if source_kind == "synthetic":
        source.update({"synthetic": True, "training_source": False, "checkpoint_sha256": None,
                       "fp32_delivery_sha256": None})
    else:
        if not source_meta or not source_meta.get("approved"):
            raise PipelineError("frozen 导出必须提供已批准来源元数据")
        source.update({"synthetic": False, "training_source": True,
                       "checkpoint_sha256": sha256_file(FROZEN_BEST),
                       "fp32_delivery_sha256": (sha256_file(FP32_DELIVERY)
                                                if FP32_DELIVERY.is_file() else None)})
    weights_blob=b''.join(layer.q_weight.tobytes(order='C') for layer in layers)
    bias_blob=b''.join(layer.q_bias.astype('<i4').tobytes(order='C') for layer in layers)
    (out_dir/'weights_int8.bin').write_bytes(weights_blob);(out_dir/'bias_int32.bin').write_bytes(bias_blob)
    spec = {"schema": EXPORT_SCHEMA, "utc": utc_now(), "source": source,
            "graph": {"input_channels": 1, "num_classes": 4, "base_channels": 16, "num_pools": 3,
                      "convs": TOTAL_CONVS, "hidden_convs": HIDDEN_CONVS,
                      "layer_names": list(LAYER_NAMES),
                      "concat_skip_up": [list(pair) for pair in CONCAT_SKIP_UP],
                      "skip_first": True, "padding": "same (1 for 3x3, 0 for 1x1)"},
            "quantization": {"signed_range": [i8.QMIN, i8.QMAX], "zero_point": 0,
                             "qfrac": i8.QFRAC, "rounding": "ties_even", "weight_layout": "OIHW",
                             "activation_layout": "NCHW", "shared_head_output_scale": True,
                             "no_sigmoid": True, "input_scale": float(i8.INPUT_SCALE),
                             "software_simulation_only": True, "not_native_int8_speed": True},
            "layers": layer_records, "concat": concat_records, "full_sample": full_sample,
            "packed": {"weights_int8": {"file":"weights_int8.bin","sha256":sha256_file(out_dir/"weights_int8.bin"),"bytes": weight_offset, "elements": weight_offset},
                       "bias_int32": {"file":"bias_int32.bin","sha256":sha256_file(out_dir/"bias_int32.bin"),"bytes": bias_offset * 4, "elements": bias_offset}},
            "code_identity": code_identity(),
            "protocol_sha256": sha256_file(PROTOCOL) if PROTOCOL.is_file() else None,
            "roster_sha256": sha256_file(ROSTER) if ROSTER.is_file() else None,
            "cache_manifest_sha256": (sha256_file(CACHE_MANIFEST) if CACHE_MANIFEST.is_file()
                                      else None),
            "calibration": {"frames": int(calibration.get("frames") or 0),
                            "input_scale": float(calibration.get("input_scale") or 0.0)}}
    (out_dir / "model_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    return spec


# --------------------------------------------------------------------------- 校验 / 装载

def _safe_member_path(root: Path, candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute() or ".." in path.parts:
        raise PipelineError(f"非法路径（绝对或含 ..）: {candidate}")
    resolved = (root / path).resolve()
    if root.resolve() not in resolved.parents and resolved != root.resolve():
        raise PipelineError(f"路径逃逸导出目录: {candidate}")
    return resolved


def _read_array(root,info,*,npy=False):
    path=_safe_member_path(root,info['file'])
    if not path.is_file() or path.stat().st_size!=int(info['bytes']) or sha256_file(path)!=info['sha256']:raise PipelineError(f'file integrity failed: {path}')
    if npy:arr=np.load(path,allow_pickle=False)
    else:
        dtype={'int8':np.dtype('i1'),'int32':np.dtype('<i4'),'float64':np.dtype('<f8')}.get(info['dtype'])
        if dtype is None:raise PipelineError('unsupported binary dtype')
        arr=np.frombuffer(path.read_bytes(),dtype=dtype).reshape(info['shape'])
    if arr.shape!=tuple(info['shape']) or str(arr.dtype)!=info['dtype']:raise PipelineError('dtype/shape metadata mismatch')
    return arr

def _read_layers(root,spec):
    result=[]
    for record in spec['layers']:
        d=_safe_member_path(root,record['directory'])
        w=_read_array(d,record['weight_file']);b=_read_array(d,record['bias_file']);m=_read_array(d,record['multiplier_file']);sc=_read_array(d,record['weight_scales_file'],npy=True)
        if w.shape!=tuple(record['weight_shape']) or b.shape!=tuple(record['bias_shape']) or sc.shape!=b.shape or not np.isfinite(sc).all() or np.any(sc<=0):raise PipelineError('invalid parameter shape/scales')
        bound=i8.require_bounds(w,b)
        layer=i8.QuantLayer(record['name'],w,b,m,float(record['input_scale']),float(record['output_scale']),int(record['padding']),int(record['kernel']),bound,True,sc)
        i8.check_layer(layer)
        if not np.array_equal(i8.qm.q24_multipliers(layer.input_scale,sc,layer.output_scale),m) or m.tolist()!=record['multiplier_q24'] or int(bound.max())!=record['conservative_bound_max']:raise PipelineError('Q24/bound metadata mismatch')
        result.append(layer)
    return result

def verify_export(spec_path:Path,*,expected_source_kind:str|None=None):
    spec_path=Path(spec_path);root=spec_path.parent;spec=json.loads(spec_path.read_text(encoding='utf-8'));source=spec.get('source',{})
    if spec.get('schema')!=EXPORT_SCHEMA or source.get('kind') not in SOURCE_KINDS:raise PipelineError('invalid export schema/source')
    if expected_source_kind and source['kind']!=expected_source_kind:raise PipelineError('source kind mismatch')
    records=spec.get('layers',[])
    if [q['name'] for q in records]!=list(LAYER_NAMES) or spec['graph'].get('layer_names')!=list(LAYER_NAMES) or spec['graph'].get('num_classes')!=4:raise PipelineError('exact CAMUS topology required')
    quant=spec['quantization']
    if quant.get('input_scale')!=i8.INPUT_SCALE or quant.get('signed_range')!=[-127,127] or quant.get('qfrac')!=24 or quant.get('shared_head_output_scale') is not True or quant.get('no_sigmoid') is not True:raise PipelineError('integer interface mismatch')
    if source['kind']=='frozen':
        if source.get('synthetic') is not False or source.get('training_source') is not True or source.get('checkpoint_sha256')!=EXPECTED_CHECKPOINT_SHA or source.get('fp32_delivery_sha256')!=EXPECTED_FP32_DELIVERY_SHA:raise PipelineError('frozen lineage missing')
        fp32,_=load_fp32_model()
    else:
        if source.get('training_source') is not False or source.get('synthetic') is not True or source.get('checkpoint_sha256') is not None:raise PipelineError('synthetic cannot claim frozen lineage')
        fp32=None
    packed={}
    for key in ('weights_int8','bias_int32'):
        info=spec['packed'][key];path=_safe_member_path(root,info['file'])
        if not path.is_file() or path.stat().st_size!=info['bytes'] or sha256_file(path)!=info['sha256']:raise PipelineError('packed binary integrity failed')
        packed[key]=path.read_bytes()
    layers=_read_layers(root,spec);wo=bo=0;seen=set();rows=[]
    for index,(rec,layer) in enumerate(zip(records,layers)):
        guard();d=_safe_member_path(root,rec['directory'])
        if rec['index']!=index or rec['packed_weight_offset']!=wo or rec['packed_bias_offset']!=bo or rec['packed_weight_count']!=layer.q_weight.size or rec['packed_bias_count']!=layer.q_bias.size:raise PipelineError('packed offset/count mismatch')
        wb=layer.q_weight.tobytes();bb=layer.q_bias.astype('<i4').tobytes()
        if packed['weights_int8'][wo:wo+len(wb)]!=wb or packed['bias_int32'][bo*4:bo*4+len(bb)]!=bb:raise PipelineError('packed/layer bytes differ')
        wo+=len(wb);bo+=layer.q_bias.size
        for info in list(rec['tile'].values())+[rec[k] for k in ('weight_file','bias_file','multiplier_file','weight_scales_file')]:
            loc=d/'tile' if info in list(rec['tile'].values()) else d
            path=_safe_member_path(loc,info['file'])
            if str(path) in seen:raise PipelineError('duplicate member')
            seen.add(str(path));_read_array(loc,info,npy=info['file'].endswith('.npy'))
        tile=rec['tile'];x=_read_array(d/'tile',tile['input']);acc=_read_array(d/'tile',tile['acc_golden']);q=_read_array(d/'tile',tile['output_golden'])
        oracle_acc,oracle_q=_oracle_tile(x,layer,relu=index<14)
        if not np.array_equal(acc,oracle_acc) or not np.array_equal(q,oracle_q):raise PipelineError('integer tile oracle mismatch')
        if fp32 is not None:
            conv=fp32.get_submodule(layer.name);qw,sc=i8.qm.quantize_weights(conv.weight.detach().numpy());qb=i8.qm.quantize_bias(conv.bias.detach().numpy(),layer.input_scale,sc)
            if not np.array_equal(qw,layer.q_weight) or not np.array_equal(sc,layer.weight_scales) or not np.array_equal(qb,layer.q_bias):raise PipelineError('quantized parameters differ from frozen FP32 source')
        rows.append({'name':layer.name,'tile_bitexact':True,'weight_int8_bytes':layer.q_weight.size,'bias_int32_bytes':layer.q_bias.nbytes})
    if wo!=len(packed['weights_int8']) or bo*4!=len(packed['bias_int32']) or wo!=486352 or bo!=708:raise PipelineError('parameter totals mismatch')
    entry=_read_array(root/records[0]['directory']/'tile',records[0]['tile']['input_full'])
    # Shared first-input scale reconstructs the same grid exactly at the quantizer.
    run=i8_graph.forward_graph(None,layers,entry[0,0].astype(np.float32)*i8.INPUT_SCALE,capture=True,oracle=False)
    for rec in records:
        guard();d=root/rec['directory']/'tile';cap=run['captures'][rec['name']]
        if not np.array_equal(_read_array(d,rec['tile']['input_full']),cap['input_int8']) or not np.array_equal(_read_array(d,rec['tile']['output_full_golden']),cap['output_int8']):raise PipelineError('full-layer topology golden mismatch')
    concats=spec.get('concat',[])
    if [c['level'] for c in concats]!=[2,1,0]:raise PipelineError('three ordered concat entries required')
    names={'skip_int8':'skip_int8','up_int8':'up_int8','skip_rescaled':'skip_rescaled_int8','up_rescaled':'up_rescaled_int8','merged_int8':'output_int8'}
    for c in concats:
        guard();d=root/'concat'/f"level{c['level']}";cap=run['captures'][f"concat_level{c['level']}"]
        if c['source_scales']!=cap['source_scales'] or c['common_scale']!=cap['common_scale'] or c['multiplier_q24']!=cap['multiplier_q24']:raise PipelineError('concat scale metadata differs')
        for k,target in names.items():
            if not np.array_equal(_read_array(d,c['files'][k]),cap[target]):raise PipelineError('concat branch/full integer golden mismatch')
    return {'schema':'camus-int8-verify-v2','utc':utc_now(),'layers':rows,'concat':[{'level':c['level'],'branch_bitexact':True} for c in concats],'checks':{'layer_count':15,'concat_count':3,'weight_elements':wo,'bias_elements':bo,'weights_int8_bytes':wo,'bias_int32_bytes':bo*4,'paths_safe':True,'hashes_ok':True,'packed_bitexact':True,'tiles_bitexact':True,'full_graph_bitexact':True,'source_fp32_parameters_verified':fp32 is not None,'weights_only_safe_fp32_load':True,'source_kind':source['kind']}}


def _rescale_with_multiplier(values: np.ndarray, multiplier: int) -> np.ndarray:
    """用给定 Q24 乘子做整数重定标（ties-even、饱和）——与 graph 的乘法口径一致。"""
    products = values.astype(np.int64) * int(multiplier)
    magnitude = np.abs(products)
    quotient = magnitude >> 24
    remainder = magnitude & ((1 << 24) - 1)
    half = 1 << 23
    increment = (remainder > half) | ((remainder == half) & ((quotient & 1) == 1))
    rounded = quotient + increment.astype(np.int64)
    signed = np.where(products < 0, -rounded, rounded)
    return np.clip(signed, i8.QMIN, i8.QMAX).astype(np.int8)


def load_export(spec_path:Path):
    verify_export(spec_path)
    spec=json.loads(Path(spec_path).read_text(encoding='utf-8'))
    return None,_read_layers(Path(spec_path).parent,spec),spec


def synthetic_model(seed: int = 20261012, size: int = 64) -> nn.Module:
    """合成随机模型（仅 CPU 检查；**不冒充真实 CAMUS 权重**）。"""
    from software.role_a_camus import model as camus_model

    torch.manual_seed(seed)
    model = camus_model.build_model(base_channels=16, num_pools=3, size=size)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.05)
    model.eval()
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAMUS INT8 导出/校验（默认 synthetic，不读真实数据）")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="导出 CAMUS INT8 v1（synthetic 仅 CPU 合成）")
    export.add_argument("--out-dir", required=True)
    export.add_argument("--source-kind", default="synthetic", choices=list(SOURCE_KINDS))
    export.add_argument("--size", type=int, default=64)
    verify = sub.add_parser("verify", help="校验导出目录")
    verify.add_argument("--spec", required=True)
    verify.add_argument("--expect-source-kind", default=None, choices=list(SOURCE_KINDS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export":
        if args.source_kind == "frozen":
            raise PipelineError("frozen 导出需要 owner 批准入口（本 CLI 仅 synthetic）")
        model = synthetic_model(size=args.size)
        image = np.linspace(0.0, 1.0, args.size * args.size, dtype=np.float32).reshape(args.size,
                                                                                     args.size)
        calibration = calibrate(model, [("synthetic", image)])
        layers = build_layers(model, calibration)
        spec = export_graph(model, calibration, layers, Path(args.out_dir), sample_image=image,
                            source_kind="synthetic")
        print(json.dumps({"schema": spec["schema"], "layers": len(spec["layers"]),
                          "source_kind": spec["source"]["kind"],
                          "out_dir": Path(args.out_dir).as_posix()}, ensure_ascii=False), flush=True)
        return 0
    report = verify_export(Path(args.spec), expected_source_kind=args.expect_source_kind)
    print(json.dumps(report["checks"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def forward_graph(model,layers,image,*,valid=None,capture=False):
    return i8_graph.forward_graph(model,layers,image,valid=valid,capture=capture)

def native_labels_from_logits(logits,meta):
    return base_geo.argmax_on_native_torch(torch.from_numpy(np.asarray(logits,dtype=np.float32))[None],meta)[0].numpy()
