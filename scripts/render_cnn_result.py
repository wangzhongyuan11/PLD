"""Plot the downloaded PL logits; this is an offline evidence view, not HDMI."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reference.tinyunet import DEFAULT_MODEL_ROOT,Model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=ROOT/"out/stage34/input_int8.bin")
    p.add_argument("--head",type=Path,default=ROOT/"out/stage34/head_int8.bin")
    p.add_argument("--metadata",type=Path,default=ROOT/"out/stage34/head_int8.json")
    p.add_argument("--output",type=Path,default=ROOT/"out/stage34/cnn_result.png")
    p.add_argument("--model",type=Path,default=DEFAULT_MODEL_ROOT)
    args=p.parse_args()
    model=Model(args.model)
    meta=json.loads(args.metadata.read_text())
    input_bytes,head_bytes=args.input.read_bytes(),args.head.read_bytes()
    if (meta["model_spec_sha256"]!=model.spec_hash or meta["status"]!="success" or
        hashlib.sha256(input_bytes).hexdigest()!=meta["input_metadata"]["input_sha256"] or
        hashlib.sha256(head_bytes).hexdigest()!=meta["output_sha256"]):
        raise ValueError("result/input/model identity mismatch")
    image=np.frombuffer(input_bytes,np.int8).reshape(3,256,256).transpose(1,2,0)
    image=np.clip(image.astype(np.float32)*model.layers[0]["input_scale"],0,1)
    head=np.frombuffer(head_bytes,np.int8).reshape(256,256)
    left,top,width,height=meta["valid_rect"]
    valid=np.zeros((256,256),bool); valid[top:top+height,left:left+width]=True
    mask=(head>=meta["threshold_q"])&valid
    overlay=image.copy(); overlay[mask]=0.6*overlay[mask]+0.4*np.array([1,0.15,0.05])
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout="constrained")
    axes[0].imshow(image); axes[0].set_title("PS input (dequantized)")
    im=axes[1].imshow(head,cmap="coolwarm",vmin=-127,vmax=127)
    axes[1].set_title("PL head: signed INT8")
    fig.colorbar(im,ax=axes[1],shrink=0.8,label="quantized logit")
    axes[2].imshow(overlay); axes[2].set_title(f"q >= {meta['threshold_q']}; valid area {100*mask.sum()/valid.sum():.2f}%")
    for ax in axes: ax.axis("off")
    fig.suptitle(f"PYNQ-Z2 | frame {meta['frame_id']} | downloaded hardware result",fontsize=15)
    fig.supxlabel("Offline visualization of raw PL output; no morphology. Engineering demonstration only.",fontsize=10)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=150); plt.close(fig)
    print(f"RESULT_VIEW_PASS path={args.output} lesion_pixels={mask.sum()} valid_pixels={valid.sum()}")


if __name__=="__main__": main()
