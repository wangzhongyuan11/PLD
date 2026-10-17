"""把已验收的 deliverables/role_a_final/ 打包为 deliverables/role_a_final.zip（只收录该目录）。

安全边界：
* 只读取 deliverables/role_a_final/，输出固定为 deliverables/role_a_final.zip（同一父目录，不接受任意路径）；
* 不收录 data/、.venv/、runs/ 或任何其他目录；跳过符号链接/重解析点（不跟随、不入包）；
* 创建后用 zipfile 只读核查：条目数、每条的路径、未压缩字节、CRC 与目录内文件一一对应。

不修改包内任何文件（含 MANIFEST.json）与模型证据。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]      # software/role_a/final_docs/<file> -> 仓库根
PACKAGE = PROJECT / "deliverables" / "role_a_final"
ARCHIVE = PROJECT / "deliverables" / "role_a_final.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return bool(attributes & 0x400)


def collect(root: Path) -> list[Path]:
    """逐级遍历收集要入包的文件（确定性顺序）：**进入任何目录前**先判链接/重解析点。

    刻意不使用 ``root.rglob('*')``：在 Windows 上 rglob 可能先递归进入 junction
    指向的外部目录，之后才做检查。这里用 ``os.scandir`` 显式下行，遇到符号链接/
    junction **直接拒绝打包**（不是跳过），绝不进入其目标。
    """
    files: list[Path] = []
    directories = [root]
    while directories:
        current = directories.pop()
        entries: list[os.DirEntry] = []
        try:
            with os.scandir(current) as iterator:
                entries = list(iterator)
        except OSError as error:
            raise SystemExit(f"[zip] 无法读取目录，拒绝打包: {current} ({error})")
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink() or is_link_or_reparse(path):
                raise SystemExit(f"[zip] 包内存在符号链接/重解析点，拒绝打包: {path}")
            try:
                if entry.is_dir(follow_symlinks=False):
                    directories.append(path)
                elif entry.is_file(follow_symlinks=False):
                    files.append(path)
            except OSError as error:
                raise SystemExit(f"[zip] 无法判定条目类型，拒绝打包: {path} ({error})")
    return sorted(files)


def verify(root: Path, archive: Path, expected: list[Path]) -> None:
    with zipfile.ZipFile(archive) as handle:
        bad_crc = handle.testzip()
        if bad_crc is not None:
            raise SystemExit(f"[zip] CRC 校验失败: {bad_crc}")
        names = handle.namelist()
        if len(names) != len(expected):
            raise SystemExit(f"[zip] 条目数不符: 包内 {len(names)} vs 目录 {len(expected)}")
        for path in expected:
            relative = path.relative_to(root.parent).as_posix()
            info = handle.getinfo(relative)
            if info.file_size != path.stat().st_size:
                raise SystemExit(f"[zip] 字节数不符: {relative} {info.file_size} != {path.stat().st_size}")
            with handle.open(info) as stream, path.open("rb") as raw:
                packed = hashlib.sha256(stream.read()).hexdigest()
                on_disk = hashlib.sha256(raw.read()).hexdigest()
            if packed != on_disk:
                raise SystemExit(f"[zip] 内容哈希不符: {relative}")
        print(f"[zip] verified entries={len(names)} crc=ok")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    root = args.package.resolve()
    archive = args.archive.resolve()
    allowed_root = (PROJECT / "deliverables").resolve()
    if root != (allowed_root / "role_a_final") or archive != (allowed_root / "role_a_final.zip"):
        raise SystemExit("[zip] 路径被限制为 deliverables/role_a_final → deliverables/role_a_final.zip")
    if not root.is_dir():
        raise SystemExit(f"[zip] 包目录不存在: {root}")
    for forbidden in ("data", ".venv", "runs"):
        if (root / forbidden).exists():
            raise SystemExit(f"[zip] 包目录内不应包含 {forbidden}/")

    files = collect(root)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in files:
            handle.write(path, path.relative_to(root.parent).as_posix())
    os.replace(temporary, archive)

    verify(root, archive, files)
    print(f"[zip] archive={archive}")
    print(f"[zip] entries={len(files)} bytes={archive.stat().st_size} sha256={sha256(archive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
