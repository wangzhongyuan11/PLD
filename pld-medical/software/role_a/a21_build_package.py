"""生成 A 组最终交付包 deliverables/role_a_final/（不覆盖旧 deliverables/role_a）。

原则：交付的最终模型是**已验证的 soup_v5_a050**（SHA256 f488719f…67eb77），
其 INT8 产物直接复制自 A14-S 已验收的导出目录（不做任何重新量化/重新生成），
本脚本只做复制、放置证据、计算 SHA256 与字节数、生成 MANIFEST.json。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SOUP = PROJECT / "runs" / "role_a_a14_soup" / "candidates" / "soup_v5_a050.pt"
SOUP_SHA256 = "f488719fe6d4d09c9b86be84d06b19d34241d6c14060b87da7bab25d1367eb77"
INT8_SRC = PROJECT / "runs" / "role_a_a14_soup" / "int8_soup_v5_a050"
DEV_SRC = PROJECT / "runs" / "role_a_a14_soup" / "int8_soup_v5_a050_development"
A21_DIR = PROJECT / "runs" / "role_a_final_qat8"
CONSISTENCY_638 = PROJECT / "runs" / "role_a_final_stage" / "a21_consistency_638.json"
OUT = PROJECT / "deliverables" / "role_a_final"
OLD = PROJECT / "deliverables" / "role_a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, origin: str, role: str) -> dict:
    return {"path": path.relative_to(OUT).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path), "origin": origin, "role": role}


def copy_file(source: Path, target: Path, origin: str, role: str) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return record(target, origin, role)


def _is_link_or_reparse(path: Path) -> bool:
    """不跟随链接地判断路径是否为符号链接/重解析点。

    顺序很关键：**先** ``is_symlink()``（对悬空链接也返回 True，不会 stat），
    **再** ``stat(follow_symlinks=False)`` 读属性位。
    缺失路径返回 False（不存在即不可能是链接）；只有真正的 OSError（权限等）才保守视为不安全。
    """
    try:
        if path.is_symlink():
            return True
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return bool(attributes & 0x400)          # FILE_ATTRIBUTE_REPARSE_POINT


def _is_reparse_point(path: Path) -> bool:
    """路径自身（或其已存在的最近祖先）是否为符号链接/重解析点。

    末级路径**不存在**是正常情况（首次生成包），此时只检查已存在的祖先；
    悬空符号链接也会被 ``is_symlink()`` 直接识别为不安全。
    """
    current = Path(path)
    while True:
        if _is_link_or_reparse(current):
            return True
        if not current.exists() and not current.is_symlink():
            # 该级不存在：继续向上检查祖先
            if current.parent == current:
                return False
            current = current.parent
            continue
        if current.parent == current:
            return False
        current = current.parent


def _scan_for_links(directory: Path) -> list[Path]:
    """递归预检将被清理的整棵目录树，返回其中的符号链接/重解析点（不跟随、不删除）。"""
    found: list[Path] = []
    stack = [directory]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            found.append(current)               # 无法读取：保守视为不安全
            continue
        for entry in entries:
            path = Path(entry.path)
            if os.path.islink(entry.path) or _is_link_or_reparse(path):
                found.append(path)
                continue                        # 绝不进入链接指向的目标
            try:
                if entry.is_dir(follow_symlinks=False):
                    key = str(path)
                    if key not in seen:
                        seen.add(key)
                        stack.append(path)
            except OSError:
                found.append(path)
    return found


def _clean_contents(directory: Path) -> int:
    """清空目录内容：**先递归预检**整棵树无符号链接/重解析点，再删除。"""
    links = _scan_for_links(directory)
    if links:
        detail = ", ".join(str(path) for path in links[:3])
        raise SystemExit(f"[package] 目标目录内存在符号链接/重解析点，拒绝清理（共 {len(links)} 个）: {detail}")
    removed = 0
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed += 1
    return removed


def resolve_target(raw: Path) -> Path:
    """硬性限制输出目录：只允许项目内 deliverables/role_a_final，且拒绝符号链接/重解析点。

    目标目录**不存在**时（首次生成）不做 stat，只校验其已存在的祖先；末级目录在写入前创建。
    """
    allowed = (PROJECT / "deliverables" / "role_a_final").resolve()
    root = PROJECT.resolve()
    target = Path(raw)
    if not target.is_absolute():
        target = PROJECT / target
    # 只规范化，不要求目标已存在；用 os.path.normpath 避免 resolve() 对缺失路径的行为差异
    target = Path(os.path.normpath(str(target)))
    if _is_reparse_point(target):
        raise SystemExit(f"[package] 目标路径或其祖先含符号链接/重解析点，拒绝写入: {target}")
    if target.resolve() != allowed:
        raise SystemExit(f"[package] 目标路径被限制为 {allowed}，收到 {target.resolve()}")
    if root not in target.resolve().parents and target.resolve() != root:
        raise SystemExit(f"[package] 目标路径必须位于项目内: {target.resolve()}")
    return target.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="只做路径校验与清单计算，不删除/不写入任何文件")
    args = parser.parse_args()
    out = resolve_target(args.out)
    if args.dry_run:
        print(json.dumps({"target": out.as_posix(), "dry_run": True}, ensure_ascii=False))
        return
    if out.exists():
        # 只清理本包（已由 resolve_target 限定为 deliverables/role_a_final）的内容；
        # 目录内的符号链接/重解析点子项会直接拒绝，不递归进入。
        _clean_contents(out)
    out.mkdir(parents=True, exist_ok=True)

    files: list[dict] = []

    # 1. 最终模型（已验证 soup）
    if sha256(SOUP) != SOUP_SHA256:
        raise SystemExit("soup 权重哈希不符，拒绝打包")
    files.append(copy_file(SOUP, out / "best.pt",
                           "runs/role_a_a14_soup/candidates/soup_v5_a050.pt",
                           "最终交付模型（FP32 权重，soup_v5_a050）"))

    # 2. INT8 15 层产物（复制自已验收导出，不重新量化）
    for path in sorted(INT8_SRC.rglob("*")):
        if path.is_file():
            relative = path.relative_to(INT8_SRC)
            files.append(copy_file(path, out / "int8" / relative,
                                   f"runs/role_a_a14_soup/int8_soup_v5_a050/{relative.as_posix()}",
                                   "INT8 导出（15 层权重/偏置/黄金向量 + model_spec + verification）"))

    # 3. dev638 INT8/FP32 证据
    for name in ("comparison.json", "fp32_isic2016_test.json", "fp32_isic2017_validation.json",
                 "fp32_isic2017_exposed_validation.json", "int8_isic2016_test.json",
                 "int8_isic2017_validation.json", "int8_isic2017_exposed_validation.json"):
        source = DEV_SRC / name
        if source.is_file():
            files.append(copy_file(source, out / "metrics" / name,
                                   f"runs/role_a_a14_soup/int8_soup_v5_a050_development/{name}",
                                   "开发集 638 逐图与汇总证据"))

    # 4. A21 QAT 负结果证据（仅作实验记录，不是最终模型）
    for name in ("REPORT.md", "history.json", "calibration_recompute.json", "consistency_check.json"):
        source = A21_DIR / name
        if source.is_file():
            files.append(copy_file(source, out / "experiments" / f"a21_qat_{name}",
                                   f"runs/role_a_final_qat8/{name}",
                                   "A21 联合 QAT 负结果实验记录（不作为最终模型）"))
    # 5. 638 逐图代理一致性
    if CONSISTENCY_638.is_file():
        files.append(copy_file(CONSISTENCY_638, out / "metrics" / "a21_consistency_638.json",
                               CONSISTENCY_638.relative_to(PROJECT).as_posix(),
                               "假量化代理 vs 真实 INT8 的 638 逐图一致性"))

    # 6. 包内文档：从可复现源目录复制（内容不在此脚本内改写）
    docs_source = Path(__file__).resolve().parent / "final_docs"
    for name, role in (("README.md", "交付包导读（最终模型/B/C 应先读哪些文件/验证命令/依赖与未上板声明）"),
                       ("REPORT.md", "交付包报告"),
                       ("B_C_接口说明.md", "B/C 接口说明"),
                       ("DATA_PROVENANCE.md", "数据来源与授权说明（依据仓库既有记录）")):
        source = docs_source / name
        if source.is_file():
            files.append(copy_file(source, out / name,
                                   f"software/role_a/final_docs/{name}", role))

    manifest = {
        "schema": "role-a-final-package-v1",
        "final_model": {"file": "best.pt", "sha256": SOUP_SHA256,
                        "source": "runs/role_a_a14_soup/candidates/soup_v5_a050.pt",
                        "note": "已验证 INT8 候选 soup_v5_a050；A21 QAT 8 轮为负结果，未采用"},
        "int8_export": {"dir": "int8", "layers": 15,
                        "weights_int8_bytes": 121760, "bias_int32_bytes": 1412,
                        "source": "runs/role_a_a14_soup/int8_soup_v5_a050（已验收，未重新量化）",
                        "verification": "int8/verification.json"},
        "development_dev638": {"dir": "metrics", "images": 638,
                               "fp32_mean_per_image_dice": 0.8440543675818143,
                               "int8_mean_per_image_dice": 0.8420508068333175,
                               "selected_threshold": 0.45,
                               "thresholds": [0.45, 0.5, 0.53, 0.56, 0.6]},
        "frozen_split": {"train": 2309, "development": 638,
                         "train_ids_sha256": "0b4f6ade53314b206fb8bcccf7d7169c90d2a35fc987dbc70404c9fd4fba2bb0",
                         "development_all_ids_sha256": "4ead5256747a87003cb341ce647e2863ec60e5d2d82a737a9572d2963805133d"},
        "holdout": "ISIC 2018 Task 1 测试掩膜未打开、未评分、未用于选模",
        "old_package_untouched": {"path": "deliverables/role_a",
                                  "files": sum(1 for _ in OLD.rglob("*") if _.is_file())
                                  if OLD.exists() else 0},
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(entry["bytes"] for entry in files),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps({"out": out.as_posix(), "files": manifest["file_count"],
                      "total_bytes": manifest["total_bytes"]}, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
