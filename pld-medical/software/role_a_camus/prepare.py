"""CAMUS 数据准备：官方名单 → 端点枚举 → QC → 原生/letterbox 缓存 + 受审 combined manifest。

命令（绝对路径示例见 ``README.md``）::

    python -m software.role_a_camus.prepare --split training --mode cache
    python -m software.role_a_camus.prepare --mode combined

规则：

* 只有**完整**枚举（train 1600 帧 / validation 200 帧）才可 ``qc_passed=true``；
  任意 ``patient_limit`` 的部分准备会显式写下 ``qc_passed=false`` 与 ``partial=true``。
* 缓存写入 **新目录**（已存在且非空即拒绝，不覆盖）。
* 每帧记录原生单位（spacing mm）、图像/GT 文件 SHA、缓存 SHA、官方 quality 与类别直方图。
* **测试组只清点路径与哈希，不解码 GT**。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from . import data as camus_data

CAMUS_ROOT_DEFAULT = Path(r"E:/嵌赛/pld-medical/datasets/camus_public/official/CAMUS_public")
OUT_DIR_DEFAULT = Path(r"E:/嵌赛/pld-medical/runs/role_a_camus/20261012/data/cache")
COMBINED_SCHEMA = "camus-combined-cache-manifest-v1"

EXPECTED_FRAMES = {camus_data.TRAIN_SPLIT: 1600, camus_data.DEV_SPLIT: 200}


def _new_cache_dir(path: Path) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise camus_data.SplitError(f"缓存目录已存在且非空，拒绝覆盖: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_split(camus_root: Path, out_dir: Path, split: str, *, size: int = 256,
                  patient_limit: int | None = None) -> dict[str, Any]:
    """Stream native frames to fresh cache; only complete official splits pass."""
    camus_root=Path(camus_root);out_dir=Path(out_dir)
    if split not in (camus_data.TRAIN_SPLIT,camus_data.DEV_SPLIT):
        raise camus_data.TestSealedError("prepare supports only training and validation; test GT remains sealed")
    patient_split=camus_data.load_split(camus_root/"database_split")
    enumeration=camus_data.enumerate_split(camus_root,split,patient_split,verify_endpoints=True)
    members=patient_split.training if split==camus_data.TRAIN_SPLIT else patient_split.validation
    if patient_limit is not None:
        if patient_limit<=0:raise camus_data.SplitError("patient_limit must be positive")
        members=members[:patient_limit]
    partial=patient_limit is not None
    cache_dir=_new_cache_dir(out_dir/f"{split}_{'full' if not partial else f'partial{patient_limit}'}")
    audit=[]
    def frames():
        for patient in members:
            for view,phase,image,gt in camus_data.expected_paths(camus_root/"database_nifti"/patient,patient):
                frame=camus_data.load_frame(image,gt,split=split,patient=patient,
                    split_object=patient_split,camus_root=camus_root,size=size)
                if tuple(frame.native_image.shape)!=tuple(frame.native_shape):
                    raise camus_data.SplitError("native shape mismatch")
                record=frame.as_record()
                record.update(image_path=image.as_posix(),gt_path=gt.as_posix(),units="mm",
                              native_pixels=int(frame.native_image.size))
                audit.append(record)
                yield frame
    cached=camus_data.cache_frames(frames(),cache_dir,config={"size":size,"split":split})
    complete=(not partial and enumeration["patients"]==camus_data.EXPECTED_COUNTS[split]
              and cached["frames"]==EXPECTED_FRAMES[split] and len(audit)==EXPECTED_FRAMES[split])
    manifest={"schema":"camus-prepare-manifest-v3","split":split,"size":size,
        "camus_root":camus_root.as_posix(),"split_source_sha256":dict(patient_split.source_sha256),
        "split_digest":patient_split.digest(),"enumeration":{"patients":enumeration["patients"],
        "frames":enumeration["frames"],"half_sequence_files":enumeration["half_sequence_files"]},
        "audit_frames":len(audit),"expected_frames":EXPECTED_FRAMES[split],"partial":partial,
        "qc_passed":complete,"test_gt_decoded":False,"audit":audit,
        "cache_manifest":(cache_dir/"CACHE_MANIFEST.json").as_posix(),
        "cache_frames":cached["frames"],"cache_dir":cache_dir.as_posix(),
        "memory_policy":"one native frame at a time"}
    output=out_dir/f"prepare_{split}{'' if not partial else f'_partial{patient_limit}'}.json"
    output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return {"path":output.as_posix(),"split":split,"patients":len(members),"frames":cached["frames"],
        "qc_passed":complete,"partial":partial,"cache_dir":cache_dir.as_posix(),"test_gt_decoded":False}



def build_combined_manifest(cache_root: Path, *, out_path: Path | None = None) -> dict[str, Any]:
    """把受审的 training(1600) + validation(200) 缓存合并为一个真实 manifest 供 train/dev 使用。"""
    cache_root = Path(cache_root)
    parts: dict[str, dict[str, Any]] = {}
    for split, expected in ((camus_data.TRAIN_SPLIT, 1600), (camus_data.DEV_SPLIT, 200)):
        path = cache_root / f"prepare_{split}.json"
        if not path.is_file():
            raise camus_data.SplitError(f"缺少 {split} 准备清单: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("partial"):
            raise camus_data.SplitError(f"{split} 为部分准备（partial=true），不得进入 combined manifest")
        if not payload.get("qc_passed"):
            raise camus_data.SplitError(f"{split} 未通过 QC，不得进入 combined manifest")
        if payload.get("audit_frames") != expected:
            raise camus_data.SplitError(f"{split} 帧数 {payload.get('audit_frames')} != {expected}")
        records = payload.get("records")
        if not records:
            cache_manifest_path = payload.get("cache_manifest")
            if not cache_manifest_path:
                raise camus_data.SplitError(f"{split} 准备清单缺少 cache_manifest")
            records = json.loads(Path(cache_manifest_path).read_text(encoding="utf-8"))["records"]
        parts[split] = {"manifest": path.as_posix(), "split_digest": payload.get("split_digest"),
                        "split_source_sha256": payload.get("split_source_sha256"),
                        "records": records}
    combined = {
        "schema": COMBINED_SCHEMA, "camus_root": None,
        "splits": {split: {"manifest": part["manifest"], "split_digest": part["split_digest"],
                           "split_source_sha256": part["split_source_sha256"],
                           "frames": len(part["records"])} for split, part in parts.items()},
        "records": parts[camus_data.TRAIN_SPLIT]["records"] + parts[camus_data.DEV_SPLIT]["records"],
        "counts": {camus_data.TRAIN_SPLIT: len(parts[camus_data.TRAIN_SPLIT]["records"]),
                   camus_data.DEV_SPLIT: len(parts[camus_data.DEV_SPLIT]["records"])},
        "qc_passed": True, "test_gt_decoded": False, "split": "combined",
        "note": "只含 training 1600 + validation 200；测试组不在其中"}
    target = Path(out_path) if out_path else cache_root / "COMBINED_MANIFEST.json"
    target.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    combined["path"] = target.as_posix()
    return combined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAMUS 端点枚举、QC、原生缓存与 combined manifest")
    parser.add_argument("--camus-root", default=str(CAMUS_ROOT_DEFAULT))
    parser.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    parser.add_argument("--split", default=camus_data.TRAIN_SPLIT,
                        choices=[camus_data.TRAIN_SPLIT, camus_data.DEV_SPLIT, camus_data.TEST_SPLIT])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--patient-limit", type=int, default=None,
                        help="仅用于冒烟；设置后 qc_passed 必为 false")
    parser.add_argument("--mode", default="cache", choices=["cache", "combined"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "combined":
        result = build_combined_manifest(Path(args.out_dir))
        print(json.dumps({"path": result["path"], "counts": result["counts"],
                          "qc_passed": result["qc_passed"]}, ensure_ascii=False), flush=True)
        return 0
    result = prepare_split(Path(args.camus_root), Path(args.out_dir), args.split, size=args.size,
                           patient_limit=args.patient_limit)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if (result["qc_passed"] or result["split"] == camus_data.TEST_SPLIT) else 1


if __name__ == "__main__":
    raise SystemExit(main())
