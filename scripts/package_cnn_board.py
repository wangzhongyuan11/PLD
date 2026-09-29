"""Package a checked CNN bit/hwh, frozen A data, and the matching PS runtime."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reference.tinyunet import Model,DEFAULT_MODEL_ROOT,FINAL_SPEC_SHA256
from write_build_manifest import verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build",type=Path,default=ROOT/"out/vivado_cnn")
    p.add_argument("--output",type=Path,default=ROOT/"out/stage34/cnn_bundle.tar.gz")
    p.add_argument("--model",type=Path,default=DEFAULT_MODEL_ROOT)
    args=p.parse_args()
    verify(args.build)
    model=Model(args.model)
    manifest=json.loads(model.manifest_path.read_text(encoding="utf-8"))
    model_directory="role_a_final" if model.spec_hash==FINAL_SPEC_SHA256 else "role_a_handoff/role_a"
    files={}
    for name in ("simple_dma","tensor_accel","pl_graph","verify_cnn","profile_cnn","evaluate_dataset","infer_images","cpu_baseline","native_copy","cache_sync","native_control","native_resize"):
        files[f"board/{name}.py"]=ROOT/"board"/f"{name}.py"
    for name in ("__init__.py","tinyunet.py","tiling.py","preprocess.py","parallel_resize.py","native_resize.py","pillow_resize.py","evaluation.py","datasets.py","dataset_contracts.json","baseline_manifest.json","final_manifest.json"):
        files[f"reference/{name}"]=ROOT/"reference"/name
    files["scripts/prepare_cnn_input.py"]=ROOT/"scripts/prepare_cnn_input.py"
    for name in ("board/native/tensor_copy.c", "scripts/build_native_copy.py", "scripts/verify_native_copy.py",
                 "board/native/tensor_control.c", "scripts/build_native_control.py", "scripts/verify_native_control.py",
                 "board/native/rgb_resize.c", "scripts/build_native_resize.py", "scripts/check_native_resize.py"):
        files[name]=ROOT/name
    for name in manifest["files"]:
        path=model.root.parent/name
        path.resolve().relative_to(model.root.parent.resolve())
        files[f"{model_directory}/{name}"]=path
    for ext in ("bit","hwh"):
        files[f"overlay/dma_top.{ext}"]=args.build/f"dma_top.{ext}"
    files["overlay/build_manifest.json"]=args.build/"build_manifest.json"
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
    provenance={"model_spec_sha256":model.spec_hash,"files":hashes,
                "model_directory":model_directory,"model_manifest_sha256":model.manifest_hash,
                "model_name":model.model_name,"threshold":model.threshold,
                "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                "git_status":subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True).splitlines()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(args.output,"w:gz") as archive:
        for name,path in sorted(files.items()): archive.add(path,arcname=name,recursive=False)
        data=(json.dumps(provenance,indent=2)+"\n").encode()
        info=tarfile.TarInfo("deployment_manifest.json"); info.size=len(data)
        archive.addfile(info,io.BytesIO(data))
    print(f"CNN_PACKAGE_PASS files={len(files)} bytes={args.output.stat().st_size} path={args.output}")


if __name__=="__main__": main()
