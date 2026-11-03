"""交付包最终核验：MANIFEST 逐条哈希 + 文档修订检查（只读）。"""
from __future__ import annotations

import hashlib
import json
import pathlib

ROOT = pathlib.Path(r"E:\嵌赛\pld-medical")
PKG = ROOT / "deliverables" / "role_a_final"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((PKG / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = [entry["path"] for entry in manifest["files"]
           if not (PKG / entry["path"]).is_file()
           or (PKG / entry["path"]).stat().st_size != entry["bytes"]
           or sha256(PKG / entry["path"]) != entry["sha256"]]
    print("manifest entries", len(manifest["files"]), "| mismatches", bad)
    files = [p for p in PKG.rglob("*") if p.is_file()]
    print("disk files", len(files), "bytes", sum(p.stat().st_size for p in files))

    report = (PKG / "REPORT.md").read_text(encoding="utf-8")
    dollar_py = "$py = " in report
    bare = [line for line in report.splitlines() if line.strip().startswith("python ")]
    print("report uses $py:", dollar_py, "| bare 'python ' lines:", len(bare))
    print("report doc == source doc:",
          (PKG / "REPORT.md").read_bytes() ==
          (ROOT / "software" / "role_a" / "final_docs" / "REPORT.md").read_bytes())
    print("iface doc == source doc:",
          (PKG / "B_C_接口说明.md").read_bytes() ==
          (ROOT / "software" / "role_a" / "final_docs" / "B_C_接口说明.md").read_bytes())

    for rel in ("REPORT.md", "B_C_接口说明.md", "MANIFEST.json", "best.pt",
                "int8/verification.json", "metrics/comparison.json",
                "metrics/a21_consistency_638.json"):
        path = PKG / rel
        print(rel, path.stat().st_size, sha256(path)[:16])
    for rel in ("software/role_a/a21_build_package.py",
                "software/role_a/a21_consistency_638.py"):
        path = ROOT / rel
        print(rel, path.stat().st_size, sha256(path)[:16])


if __name__ == "__main__":
    main()
