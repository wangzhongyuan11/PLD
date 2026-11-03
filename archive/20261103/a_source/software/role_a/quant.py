"""Tiny U-Net 的可复现 INT8 导出、逐层黄金向量与 FP32/INT8 对比。

量化是从已训练 FP32 checkpoint 和训练集样本校准的；没有真实 checkpoint 时仅可
用合成数据验证程序，不能把产物称为有效病灶分割模型。

``export`` 有两条互斥路径：

* 旧路径 ``--split --image-dir --mask-dir``：只读给定划分的 train 角色，行为与产物
  格式保持不变；
* ``--overall-train``：从冻结的 ``train_overall_v1`` 装配（三来源 2309 训练图）按
  来源确定性均匀取样校准，绝不读取 ISIC 2018 最终留出集。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

if __package__:
    from .data import ISICDataset, IsicPaths, enumerate_isic_pairs, load_split
    from .model import TinyUNet, build_model
    from .quant_math import (
        QFRAC, QMAX, QMIN, conv2d_int32, q24_multipliers, quantize_bias,
        quantize_weights, requantize_int8, symmetric_scale,
    )
    from .train import _items_for, evaluate, make_loader
else:  # Historical direct-script and test imports remain supported.
    from data import ISICDataset, IsicPaths, enumerate_isic_pairs, load_split
    from model import TinyUNet, build_model
    from quant_math import (
        QFRAC, QMAX, QMIN, conv2d_int32, q24_multipliers, quantize_bias,
        quantize_weights, requantize_int8, symmetric_scale,
    )
    from train import _items_for, evaluate, make_loader


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_fp32(path: Path) -> tuple[TinyUNet, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint or "config" not in checkpoint:
        raise ValueError("checkpoint must contain model_state and config")
    config = checkpoint["config"]
    size = int(config["size"])
    model = build_model(base_channels=int(config["base_channels"]),
                        num_pools=int(config["num_pools"]), input_shape=(3, size, size))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.eval(), config


def _round_divide_even_torch(products: Tensor) -> Tensor:
    magnitude = products.abs()
    quotient = magnitude >> QFRAC
    remainder = magnitude & ((1 << QFRAC) - 1)
    half = 1 << (QFRAC - 1)
    increment = (remainder > half) | ((remainder == half) & ((quotient & 1) == 1))
    rounded = quotient + increment.to(torch.int64)
    return torch.where(products < 0, -rounded, rounded)


class QuantConv2d(nn.Module):
    """独立卷积输入与输出均为零点 0 的 signed INT8，内部 INT32/Q24。"""

    def __init__(self, q_weight: np.ndarray, q_bias: np.ndarray, multiplier: np.ndarray,
                 input_scale: float, output_scale: float, padding: int) -> None:
        super().__init__()
        self.register_buffer("q_weight", torch.from_numpy(np.ascontiguousarray(q_weight, dtype=np.int8)))
        self.register_buffer("q_bias", torch.from_numpy(np.ascontiguousarray(q_bias, dtype=np.int32)))
        self.register_buffer("multiplier", torch.from_numpy(np.ascontiguousarray(multiplier, dtype=np.int32)))
        self.input_scale = float(input_scale)
        self.output_scale = float(output_scale)
        self.padding = int(padding)
        self.capture_tile = 0
        self.capture_full = False
        self.captured_input: np.ndarray | None = None
        self.captured_full_input: np.ndarray | None = None
        self.captured_full_output: np.ndarray | None = None

    @classmethod
    def from_fp32(cls, conv: nn.Conv2d, input_scale: float,
                  output_scale: float) -> tuple["QuantConv2d", np.ndarray]:
        if conv.groups != 1 or conv.stride != (1, 1) or conv.dilation != (1, 1):
            raise ValueError("only groups=1, stride=1, dilation=1 Conv2d is supported")
        if conv.kernel_size not in ((1, 1), (3, 3)) or conv.padding not in ((0, 0), (1, 1)):
            raise ValueError("only 1x1 or 3x3 convolution with 0/1 padding is supported")
        q_weight, weight_scales = quantize_weights(conv.weight.detach().cpu().numpy())
        bias = (conv.bias.detach().cpu().numpy() if conv.bias is not None
                else np.zeros(conv.out_channels, dtype=np.float64))
        q_bias = quantize_bias(bias, input_scale, weight_scales)
        multipliers = q24_multipliers(input_scale, weight_scales, output_scale)
        return cls(q_weight, q_bias, multipliers, input_scale, output_scale,
                   conv.padding[0]), weight_scales

    def forward(self, input_tensor: Tensor) -> Tensor:
        q_input = torch.round(input_tensor / self.input_scale).clamp(QMIN, QMAX).to(torch.int8)
        if self.capture_tile:
            tile = self.capture_tile
            self.captured_input = q_input[:1, :, :tile, :tile].cpu().numpy().copy()
        if self.capture_full:
            self.captured_full_input = q_input.cpu().numpy().copy()
        # 双精度卷积只对整数做乘加；本模型每层乘加数远小于 2**53，故整数结果精确。
        acc = F.conv2d(q_input.to(torch.float64), self.q_weight.to(torch.float64),
                       self.q_bias.to(torch.float64), padding=self.padding).to(torch.int64)
        products = acc * self.multiplier.to(torch.int64).view(1, -1, 1, 1)
        q_output = _round_divide_even_torch(products).clamp(QMIN, QMAX).to(torch.int8)
        if self.capture_full:
            self.captured_full_output = q_output.cpu().numpy().copy()
        return q_output.to(input_tensor.dtype) * self.output_scale


def _conv_modules(model: nn.Module) -> list[tuple[str, nn.Conv2d]]:
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, nn.Conv2d)]


def calibrate(model: TinyUNet, dataset: ISICDataset, limit: int) -> dict[str, dict[str, Any]]:
    if limit < 1 or limit > len(dataset):
        raise ValueError("calibration limit must be between 1 and dataset length")
    stats = {name: {"input_max_abs": 0.0, "output_max_abs": 0.0}
             for name, _ in _conv_modules(model)}
    handles = []
    for name, module in _conv_modules(model):
        def before(_module: nn.Module, arguments: tuple[Tensor, ...], key: str = name) -> None:
            stats[key]["input_max_abs"] = max(stats[key]["input_max_abs"],
                                                float(arguments[0].abs().amax()))
            stats[key]["input_shape"] = list(arguments[0].shape)

        def after(_module: nn.Module, _arguments: tuple[Tensor, ...], output: Tensor,
                  key: str = name) -> None:
            stats[key]["output_max_abs"] = max(stats[key]["output_max_abs"],
                                                 float(output.abs().amax()))
            stats[key]["output_shape"] = list(output.shape)

        handles.extend((module.register_forward_pre_hook(before),
                        module.register_forward_hook(after)))
    try:
        with torch.no_grad():
            for index in range(limit):
                model(dataset[index].image.unsqueeze(0))
    finally:
        for handle in handles:
            handle.remove()
    return stats


def _replace(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parent_name, _, leaf = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    if isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def _quantized_model(model: TinyUNet, stats: dict[str, dict[str, Any]]) -> tuple[TinyUNet, list[dict[str, Any]]]:
    quantized = copy.deepcopy(model)
    layers: list[dict[str, Any]] = []
    for name, conv in _conv_modules(model):
        input_scale = symmetric_scale(stats[name]["input_max_abs"])
        output_scale = symmetric_scale(stats[name]["output_max_abs"])
        qconv, weight_scales = QuantConv2d.from_fp32(conv, input_scale, output_scale)
        _replace(quantized, name, qconv)
        layers.append({
            "name": name, "directory": name.replace(".", "_"),
            "input_shape": stats[name]["input_shape"],
            "output_shape": stats[name]["output_shape"],
            "weight_shape": list(qconv.q_weight.shape),
            "padding": qconv.padding, "stride": 1,
            "input_scale": input_scale, "weight_scales": weight_scales.tolist(),
            "output_scale": output_scale, "multiplier_q24": qconv.multiplier.tolist(),
            "weight_file": "weight.bin", "bias_file": "bias.bin",
        })
    return quantized.eval(), layers


def _save_layer_golden(directory: Path, module: QuantConv2d, layer: dict[str, Any]) -> dict[str, Any]:
    if (module.captured_input is None or module.captured_full_input is None or
            module.captured_full_output is None):
        raise RuntimeError(f"no input captured for {layer['name']}")
    x = module.captured_input
    w = module.q_weight.cpu().numpy()
    b = module.q_bias.cpu().numpy()
    m = module.multiplier.cpu().numpy()
    acc = conv2d_int32(x, w, b, module.padding)
    output = requantize_int8(acc, m)
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "input.bin": x.astype(np.int8),
        "weight.bin": w.astype(np.int8),
        "bias.bin": b.astype("<i4"),
        "acc_golden.bin": acc.astype("<i4"),
        "output_golden.bin": output.astype(np.int8),
        "input_full.bin": module.captured_full_input.astype(np.int8),
        "output_full_golden.bin": module.captured_full_output.astype(np.int8),
    }
    hashes = {}
    for filename, array in files.items():
        path = directory / filename
        np.ascontiguousarray(array).tofile(path)
        hashes[filename] = _sha256(path)
    config = {
        **layer, "full_input_shape": layer["input_shape"],
        "full_output_shape": layer["output_shape"],
        "input_shape": list(x.shape), "output_shape": list(output.shape),
        "full_input_file": "input_full.bin", "full_output_file": "output_full_golden.bin",
        "layout": "NCHW", "weight_layout": "OIHW", "byte_order": "little",
        "input_dtype": "int8", "weight_dtype": "int8", "bias_dtype": "int32",
        "acc_dtype": "int32", "output_dtype": "int8", "zero_point": 0,
        "q_range": [QMIN, QMAX], "q_fractional_bits": QFRAC,
        "rounding": "nearest ties to even", "saturation": [QMIN, QMAX],
        "padding_value": 0, "files_sha256": hashes,
    }
    _write_json(directory / "config.json", config)
    return {"name": layer["name"], "directory": layer["directory"],
            "input_shape": config["input_shape"], "output_shape": config["output_shape"]}


# ------------------------------------------------------- 冻结 2309 训练校准
OVERALL_CALIBRATION_MODE: str = "overall_train_frozen_2309"
OVERALL_CALIBRATION_SPLIT: str = "train_overall_v1_frozen_2309"
#: 旧 ``--split`` 导出路径的校准模式标记。
LEGACY_CALIBRATION_MODE: str = "split_train"
#: 旧导出路径的坐标参数；与 ``--overall-train`` 同时给出时直接报错。
LEGACY_EXPORT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("split", "--split"), ("image_dir", "--image-dir"), ("mask_dir", "--mask-dir"),
)


def _overall_module() -> Any:
    """延迟导入 ``train_overall_v1``（它只装配冻结划分，从不读取 2018 留出集）。"""
    try:
        from software.role_a import train_overall_v1 as overall
    except ImportError:  # 直接把 software/role_a 加入 sys.path 的场景
        import train_overall_v1 as overall  # type: ignore[no-redef]
    return overall


def load_overall_assembly() -> Any:
    """按冻结规格装配三来源 2309 训练集与 638 开发集（不含 2018 留出集）。"""
    overall = _overall_module()
    return overall.assemble(overall.build_default_spec())


def allocate_calibration_counts(total: int, capacities: Sequence[int]) -> list[int]:
    """把 ``total`` 张校准图按轮转方式均匀分到各来源（不超过来源容量）。

    ``total=96`` 且三个来源容量充足时得到 ``[32, 32, 32]``；余数按来源顺序逐个
    分发，因此结果只依赖 ``(total, capacities)``，与集合/字典遍历顺序无关。容量
    不足的来源会被跳过，份额让给其余来源；总量超过全部容量时直接报错，绝不重复取 ID。
    """
    total = int(total)
    caps = [int(value) for value in capacities]
    if not caps:
        raise ValueError("至少需要一个训练来源")
    if any(value < 0 for value in caps):
        raise ValueError("来源容量不能为负")
    if total < 1:
        raise ValueError("calibration-images must be >= 1")
    if total > sum(caps):
        raise ValueError(
            f"calibration-images={total} 超过全部训练图 {sum(caps)}，无法不重复取样")
    counts = [0] * len(caps)
    active = [index for index, cap in enumerate(caps) if cap > 0]
    cursor = 0
    remaining = total
    while remaining > 0 and active:
        position = cursor % len(active)
        index = active[position]
        counts[index] += 1
        remaining -= 1
        if counts[index] >= caps[index]:
            active.pop(position)
        else:
            cursor += 1
    if remaining:  # pragma: no cover - 前面的容量检查已排除
        raise ValueError(f"仍有 {remaining} 张校准图无法分配")
    return counts


def evenly_spaced_indices(count: int, take: int) -> list[int]:
    """在 ``0..count-1`` 上取 ``take`` 个确定性、互不重复且均匀分布的下标。

    ``take > 1`` 时首尾固定为 ``0`` 与 ``count-1``；间隔用整数除法算，不依赖浮点，
    因此 ``take <= count`` 时严格递增（``take == count`` 即全取）。
    """
    count = int(count)
    take = int(take)
    if count < 0 or take < 0:
        raise ValueError("count/take 不能为负")
    if take == 0:
        return []
    if take > count:
        raise ValueError(f"取样数 {take} 超过可用样本 {count}")
    if take == 1:
        return [0]
    return [(index * (count - 1)) // (take - 1) for index in range(take)]


def group_train_items(assembly: Any) -> dict[str, tuple[IsicPaths, ...]]:
    """把 ``Assembly.train_items`` 按训练来源切开，并核对每项确实来自该来源目录。"""
    items = tuple(assembly.train_items)
    grouped: dict[str, tuple[IsicPaths, ...]] = {}
    offset = 0
    for source in assembly.spec.train_sources:
        count = int(assembly.counts[f"train:{source.name}"])
        chunk = items[offset:offset + count]
        if len(chunk) != count:
            raise ValueError(f"{source.name}: 训练项数量与装配计数不符")
        expected_dir = Path(source.image_dir).resolve()
        for item in chunk:
            if Path(item.image).resolve().parent != expected_dir:
                raise ValueError(f"{item.stem} 不属于训练来源 {source.name}")
        grouped[source.name] = chunk
        offset += count
    if offset != len(items):
        raise ValueError("训练项数量与各来源计数之和不符")
    return grouped


def select_calibration_items(grouped: Mapping[str, Sequence[IsicPaths]],
                             total: int) -> tuple[list[IsicPaths], list[dict[str, Any]]]:
    """按来源均匀取样；返回扁平样本列表与逐来源记录（确定顺序、ID 不重复）。"""
    names = list(grouped)
    counts = allocate_calibration_counts(total, [len(grouped[name]) for name in names])
    selected: list[IsicPaths] = []
    records: list[dict[str, Any]] = []
    for name, take in zip(names, counts):
        source_items = list(grouped[name])
        picked = [source_items[index]
                  for index in evenly_spaced_indices(len(source_items), take)]
        selected.extend(picked)
        records.append({"name": name, "train_count": len(source_items),
                        "selected": take, "stems": [item.stem for item in picked]})
    stems = [item.stem for item in selected]
    if len(set(stems)) != len(stems):
        raise ValueError("校准取样出现重复 ID")
    return selected, records


def _overall_calibration(limit: int, size: int) -> tuple[ISICDataset, list[IsicPaths],
                                                         list[dict[str, Any]], dict[str, Any]]:
    """冻结 2309 训练集上的跨来源校准取样，并复核 ID 只属于训练集。"""
    assembly = load_overall_assembly()
    grouped = group_train_items(assembly)
    items, records = select_calibration_items(grouped, limit)
    train_ids = set(assembly.train_ids)
    dev_ids = {stem for ids in assembly.dev_ids.values() for stem in ids}
    outside = [item.stem for item in items if item.stem not in train_ids]
    leaked = [item.stem for item in items if item.stem in dev_ids]
    if outside or leaked:
        raise ValueError(f"校准取样越界：非训练 ID {outside[:5]}，开发集 ID {leaked[:5]}")
    summary = {
        "train_total": int(assembly.counts["train_total"]),
        "development_total": int(assembly.counts["development_total"]),
        "train_counts_by_source": {name: int(assembly.counts[f"train:{name}"])
                                   for name in grouped},
        "development_counts_by_source": {
            source.name: int(assembly.counts[f"development:{source.name}"])
            for source in assembly.spec.dev_sources},
        "train_ids_sha256": str(assembly.split_digest["train_ids"]),
        "development_all_ids_sha256": str(assembly.split_digest["development_all"]),
        "frozen_exposed_split_sha256": str(assembly.split_digest["reference_split_file"]),
        "calibration_ids_are_train_only": True,
        "final_holdout_2018": "not read by this export path",
    }
    dataset = ISICDataset(items, root=None, size=size)
    return dataset, items, records, summary


def command_export(args: argparse.Namespace) -> int:
    if args.tile_size < 1:
        raise ValueError("tile-size must be >= 1")
    if args.calibration_images < 1:
        raise ValueError("calibration-images must be >= 1")
    conflicts = [flag for name, flag in LEGACY_EXPORT_OPTIONS
                 if getattr(args, name, None) is not None]
    if args.overall_train and conflicts:
        raise ValueError(
            "--overall-train 只使用冻结划分中的训练来源，不能与 " + " ".join(conflicts)
            + " 组合使用（这些参数会被静默忽略，故直接拒绝）")
    if not args.overall_train:
        missing = [flag for name, flag in LEGACY_EXPORT_OPTIONS
                   if getattr(args, name, None) is None]
        if missing:
            raise ValueError("export 需要 " + " ".join(missing) + "，或改用 --overall-train")
    checkpoint = Path(args.checkpoint)
    model, config = _load_fp32(checkpoint)
    calibration_extra: dict[str, Any] = {}
    if args.overall_train:
        dataset, calibration_items, source_records, split_summary = _overall_calibration(
            args.calibration_images, int(config["size"]))
        limit = len(calibration_items)
        calibration_extra = {
            "calibration_mode": OVERALL_CALIBRATION_MODE,
            "calibration_split": OVERALL_CALIBRATION_SPLIT,
            "calibration_sources": source_records,
            "training_split": split_summary,
        }
    else:
        split = load_split(args.split)
        pairs = enumerate_isic_pairs(args.image_dir, args.mask_dir, require_mask=True)
        train_items = _items_for(split, "train", pairs, allow_missing_mask=False)
        limit = min(args.calibration_images, len(train_items))
        if limit < 1:
            raise ValueError("calibration-images must be >= 1")
        # Cover the full sorted training list instead of biasing calibration to
        # its first IDs; the selection is deterministic and recorded in the spec.
        indices = np.linspace(0, len(train_items) - 1, num=limit, dtype=int)
        calibration_items = [train_items[int(index)] for index in indices]
        dataset = ISICDataset(calibration_items, root=args.image_dir,
                              size=int(config["size"]))
    stats = calibrate(model, dataset, limit)
    quant_model, layers = _quantized_model(model, stats)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        module = quant_model.get_submodule(layer["name"])
        assert isinstance(module, QuantConv2d)
        module.capture_tile = args.tile_size
        module.capture_full = True
    with torch.no_grad():
        quant_model(dataset[0].image.unsqueeze(0))
    golden = []
    for layer in layers:
        module = quant_model.get_submodule(layer["name"])
        assert isinstance(module, QuantConv2d)
        golden.append(_save_layer_golden(output_dir / layer["directory"], module, layer))
    weight_bytes = bytearray()
    bias_bytes = bytearray()
    for layer in layers:
        directory = output_dir / layer["directory"]
        weight = (directory / "weight.bin").read_bytes()
        bias = (directory / "bias.bin").read_bytes()
        layer["weight_offset_bytes"] = len(weight_bytes)
        layer["weight_nbytes"] = len(weight)
        layer["bias_offset_bytes"] = len(bias_bytes)
        layer["bias_nbytes"] = len(bias)
        weight_bytes.extend(weight)
        bias_bytes.extend(bias)
    (output_dir / "weights_int8.bin").write_bytes(weight_bytes)
    (output_dir / "bias.bin").write_bytes(bias_bytes)
    spec = {
        "schema_version": 1, "model": "TinyUNet", "size": int(config["size"]),
        "base_channels": int(config["base_channels"]), "num_pools": int(config["num_pools"]),
        "preprocessing": "RGB float32=image_uint8/255; aspect-preserving letterbox zero pad",
        "quantization": "symmetric signed INT8 per-tensor activations, per-output-channel weights; zero_point=0; INT32 bias/acc; Q24 ties-even requant",
        "zero_point": 0, "q_range": [QMIN, QMAX],
        "weights_int8_file": "weights_int8.bin", "weights_int8_sha256": _sha256(output_dir / "weights_int8.bin"),
        "bias_int32_file": "bias.bin", "bias_int32_sha256": _sha256(output_dir / "bias.bin"),
        "checkpoint_sha256": _sha256(checkpoint),
        "calibration_split": "train", "calibration_mode": LEGACY_CALIBRATION_MODE,
        "calibration_images": limit,
        "calibration_stems": [item.stem for item in calibration_items],
        "golden_source_stem": calibration_items[0].stem, "golden_tile_size": args.tile_size,
        "layers": layers, "golden_layers": golden,
        "limitations": ["small tile vectors are isolated with zero padding at tile edges; full vectors follow an end-to-end quantized forward pass",
                        "FPGA resource use and timing have not been measured"],
    }
    # 冻结 2309 模式额外记录逐来源数量、校准 ID 与划分摘要；旧模式字段保持原状。
    spec.update(calibration_extra)
    _write_json(output_dir / "model_spec.json", spec)
    print(json.dumps({"model_spec": str(output_dir / "model_spec.json"),
                      "layers": len(layers), "calibration_images": limit,
                      "calibration_mode": spec["calibration_mode"],
                      "calibration_sources": [
                          {"name": row["name"], "selected": row["selected"]}
                          for row in spec.get("calibration_sources", [])],
                      "golden_source_stem": calibration_items[0].stem}, ensure_ascii=False))
    return 0


def _load_quantized(spec_path: Path) -> tuple[TinyUNet, dict[str, Any]]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported model_spec schema")
    for name_key, hash_key in (("weights_int8_file", "weights_int8_sha256"),
                               ("bias_int32_file", "bias_int32_sha256")):
        if _sha256(spec_path.parent / spec[name_key]) != spec[hash_key]:
            raise ValueError(f"hash mismatch: {spec[name_key]}")
    size = int(spec["size"])
    model = build_model(base_channels=int(spec["base_channels"]),
                        num_pools=int(spec["num_pools"]), input_shape=(3, size, size))
    for layer in spec["layers"]:
        directory = spec_path.parent / layer["directory"]
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        for filename in (layer["weight_file"], layer["bias_file"]):
            if _sha256(directory / filename) != config["files_sha256"][filename]:
                raise ValueError(f"hash mismatch: {directory / filename}")
        shape = tuple(int(v) for v in layer["weight_shape"])
        w = np.fromfile(directory / layer["weight_file"], dtype=np.int8)
        b = np.fromfile(directory / layer["bias_file"], dtype="<i4")
        if w.size != np.prod(shape) or b.size != shape[0]:
            raise ValueError(f"invalid weight/bias file length for {layer['name']}")
        qconv = QuantConv2d(w.reshape(shape), b,
                            np.asarray(layer["multiplier_q24"], dtype=np.int32),
                            float(layer["input_scale"]), float(layer["output_scale"]),
                            int(layer["padding"]))
        _replace(model, layer["name"], qconv)
    return model.eval(), spec


def command_evaluate(args: argparse.Namespace) -> int:
    fp32, config = _load_fp32(Path(args.checkpoint))
    int8_model, spec = _load_quantized(Path(args.spec))
    if int(config["size"]) != int(spec["size"]):
        raise ValueError("FP32 checkpoint and INT8 spec use different input sizes")
    if _sha256(Path(args.checkpoint)) != spec["checkpoint_sha256"]:
        raise ValueError("INT8 spec was exported from a different FP32 checkpoint")
    split = load_split(args.split)
    pairs = enumerate_isic_pairs(args.image_dir, args.mask_dir, require_mask=True)
    items = _items_for(split, args.role, pairs, allow_missing_mask=False)
    loader = make_loader(items, Path(args.image_dir), int(config["size"]),
                         args.batch_size, shuffle=False, seed=42, num_workers=0)
    with torch.no_grad():
        fp_metrics, _ = evaluate(fp32, loader, torch.device("cpu"), threshold=args.threshold)
        q_metrics, _ = evaluate(int8_model, loader, torch.device("cpu"), threshold=args.threshold)
    report = {
        "role": args.role, "images": len(items), "threshold": args.threshold,
        "fp32_dice_mean": fp_metrics["dice_mean"], "int8_dice_mean": q_metrics["dice_mean"],
        "dice_delta_fp32_minus_int8": fp_metrics["dice_mean"] - q_metrics["dice_mean"],
        "dice_abs_difference": abs(fp_metrics["dice_mean"] - q_metrics["dice_mean"]),
        "fp32_iou_mean": fp_metrics["iou_mean"], "int8_iou_mean": q_metrics["iou_mean"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "evaluation_space": "letterbox valid region at model resolution",
    }
    if args.output_json:
        _write_json(Path(args.output_json), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="calibrate on train and export INT8 layers/golden vectors")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--split", type=Path,
                        help="旧路径：冻结划分 JSON；与 --overall-train 互斥")
    export.add_argument("--image-dir", type=Path,
                        help="旧路径：训练图像目录；与 --overall-train 互斥")
    export.add_argument("--mask-dir", type=Path,
                        help="旧路径：训练掩膜目录；与 --overall-train 互斥")
    export.add_argument("--overall-train", action="store_true",
                        help="只用冻结的 2309 张三来源训练图做跨来源 INT8 校准")
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--calibration-images", type=int, default=32)
    export.add_argument("--tile-size", type=int, default=8)
    export.set_defaults(handler=command_export)
    compare = sub.add_parser("compare", help="compare FP32 and exported INT8 on val/test")
    compare.add_argument("--checkpoint", type=Path, required=True)
    compare.add_argument("--spec", type=Path, required=True)
    compare.add_argument("--split", type=Path, required=True)
    compare.add_argument("--role", choices=("val", "test"), required=True)
    compare.add_argument("--image-dir", type=Path, required=True)
    compare.add_argument("--mask-dir", type=Path, required=True)
    compare.add_argument("--threshold", type=float, default=0.5)
    compare.add_argument("--batch-size", type=int, default=1)
    compare.add_argument("--output-json", type=Path)
    compare.set_defaults(handler=command_evaluate)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
