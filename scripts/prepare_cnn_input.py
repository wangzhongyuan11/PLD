"""Produce quantized input + letterbox metadata in the verified JPEG runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image,features

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reference.tinyunet import DEFAULT_MODEL_ROOT,Model,digest,require_equal
from reference.preprocess import preprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("image",type=Path)
    p.add_argument("--output",type=Path,default=ROOT/"out/stage34/input_int8.bin")
    p.add_argument("--compare-golden",action="store_true")
    p.add_argument("--model",type=Path,default=DEFAULT_MODEL_ROOT)
    args=p.parse_args()
    model=Model(args.model)
    data,valid,meta=preprocess(args.image,model.layers[0]["input_scale"])
    if args.compare_golden:
        require_equal(data,model.golden(0,"input"),"raw JPEG preprocessing")
    meta.update({"model_spec_sha256":model.spec_hash,"input_sha256":hashlib.sha256(data.tobytes()).hexdigest(),
                 "image_sha256":digest(args.image),"shape":list(data.shape),"dtype":"int8","layout":"NCHW",
                 "valid_rect":[meta["left"],meta["top"],meta["resized_width"],meta["resized_height"]],
                 "valid_pixels":int(valid.sum()),"pillow":Image.__version__,"jpeg":features.version("jpg"),
                 "golden_compared":args.compare_golden})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(data.tobytes())
    args.output.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(f"INPUT_PREPARE_PASS bytes={data.nbytes} valid_pixels={valid.sum()} golden_compared={args.compare_golden}")


if __name__=="__main__": main()
