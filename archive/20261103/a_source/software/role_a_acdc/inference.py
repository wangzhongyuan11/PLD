"""Final inference uses full FP32 weights/arithmetic, or exact INT8 emulation.

Training-time selection uses AMP for speed; it is reported separately. Final
FP32/INT8 comparisons never label AMP inference as FP32 arithmetic.
"""
import numpy as np
import torch
from .geometry import AXES,stack,undo_letterbox,put_plane

@torch.inference_mode()
def predict_working(model,x,direction,device,baseline=False,batch_size=16):
    model.eval();axis=AXES[direction]
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    probs=np.empty((4,*x.shape),dtype=np.float32)
    for start in range(0,x.shape[axis],batch_size):
        prepared=[stack(x,direction,i,baseline=baseline) for i in range(start,min(start+batch_size,x.shape[axis]))]
        inputs=torch.from_numpy(np.stack([a[0] for a in prepared])).to(device=device,dtype=torch.float32)
        # No autocast: the FP32 reference must remain FP32. IntegerModel itself
        # uses a bounded exact FP64 accumulator emulator and int8 node outputs.
        logits=model(inputs).float()
        if not torch.isfinite(logits).all():raise ValueError('non-finite model logits')
        predictions=logits.softmax(1).cpu().numpy()
        for j,(prediction,(_,transform)) in enumerate(zip(predictions,prepared)):
            full=undo_letterbox(prediction,transform)
            for c in range(4):put_plane(probs[c],axis,start+j,full[c])
    return probs
