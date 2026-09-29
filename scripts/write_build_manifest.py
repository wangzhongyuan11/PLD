"""Bind a successful CNN build to packaged RTL, sources, and paired artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from check_build import check

ROOT=Path(__file__).resolve().parents[1]
MODULES=("conv_mac_array","requant_q24","conv_tile_engine","tensor_tile_engine",
         "axi_lite_endpoint","axis_byte_pack","axis_byte_unpack","axis_tensor_accel")


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory):
    root=Path(directory)
    manifest=json.loads((root/"build_manifest.json").read_text())
    for name,sha in manifest["sources"].items():
        if digest(ROOT/name)!=sha:
            raise ValueError(f"source changed since CNN build: {name}")
    for name,sha in manifest["artifacts"].items():
        if digest(root/name)!=sha:
            raise ValueError(f"artifact changed since CNN build: {name}")
    check(root)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    p.add_argument("--verify",action="store_true")
    args=p.parse_args()
    if args.verify:
        verify(args.directory); print("BUILD_MANIFEST_PASS"); return
    check(args.directory)
    sources={}
    for name in MODULES:
        original=ROOT/"rtl"/f"{name}.sv"
        packed=ROOT/"ip_repo/axis_tensor_accel/src"/f"{name}.sv"
        if original.read_bytes()!=packed.read_bytes():
            raise ValueError(f"RTL changed after packaging: {name}")
        sources[original.relative_to(ROOT).as_posix()]=digest(original)
    for name in ("vivado/package_axis_tensor_accel.tcl","vivado/build_dma_bd.tcl","vivado/ps7_config.tcl",
                 "scripts/build.sh","scripts/check_build.py","scripts/extract_xsa.py","scripts/write_build_manifest.py"):
        sources[name]=digest(ROOT/name)
    manifest={"sources":sources,"artifacts":{name:digest(args.directory/name) for name in
              ("dma_top.bit","dma_top.hwh","dma_top.xsa","timing_summary.rpt","utilization.rpt","drc.rpt")},
              "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
    (args.directory/"build_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print("BUILD_MANIFEST_WRITTEN")


if __name__=="__main__": main()
