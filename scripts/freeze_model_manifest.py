"""Explicit A-version migration: verify the entire graph, then pin every file.

Never overwrites an existing manifest. FPGA shape/capacity and accuracy gates
must still pass before the new bundle can replace the deployed model.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference.tinyunet import Model, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",required=True)
    p.add_argument("--spec-sha256",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    target=Path(args.output)
    if target.exists():
        raise FileExistsError("preserve the existing version; choose a new manifest name")
    model=Model(args.model,args.spec_sha256,manifest=None)
    model.run(model.golden(0,"input"),verify=True)
    root=Path(args.model).parent
    paths=[root/"best.pt",model.root/"model_spec.json",model.root/"weights_int8.bin",model.root/"bias.bin"]
    for layer,cfg in zip(model.layers,model.configs):
        folder=model.root/layer["directory"]
        paths.append(folder/"config.json")
        paths.extend(folder/name for name in cfg["files_sha256"])
    frozen={"schema_version":1,"spec_sha256":model.spec_hash,
            "files":{path.relative_to(root).as_posix():digest(path) for path in paths}}
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x",encoding="utf-8") as f:
        f.write(json.dumps(frozen,indent=2)+"\n")
    print(f"MODEL_FROZEN files={len(frozen['files'])} path={target}")


if __name__=="__main__":
    main()
