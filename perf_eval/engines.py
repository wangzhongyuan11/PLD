"""Existing model adapters; defaults to CPU and never starts/stops board services."""
from pathlib import Path
import subprocess
import sys
import numpy as np
from .core import digest

SPECS={'skin':'fe137ca5b128fc1983fabad9bee39b3208cda4256e0d04c4e27cef5c07666aef','camus':'68b4f425cc3612ad72d5e83955c2018981edee912fbc0937e9ac8c1718e7b5ae'}
CHECKPOINTS={'skin':'f488719fe6d4d09c9b86be84d06b19d34241d6c14060b87da7bab25d1367eb77','camus':'3d8f268910783ea524513b3506f4425aec991a54dcb23bf25cff5598fb942378'}


def source_path(path):
    root=Path(path).resolve()
    if not (root/'reference/tinyunet.py').is_file() or not (root/'camus/model.py').is_file():raise ValueError('--source must be the complete PLD source root')
    sys.path.insert(0,str(root/'board'));sys.path.insert(0,str(root))
    return root


class Lease:
    def __init__(self):
        self.file=None
        if sys.platform!='linux':raise RuntimeError('PL requires the Linux PYNQ board')
        for unit in ('pld-hub.service','pld-web.service','pld-camus.service'):
            p=subprocess.run(['systemctl','show',unit,'-p','ActiveState','-p','MainPID'],capture_output=True,text=True,timeout=10)
            state=dict(line.split('=',1) for line in p.stdout.splitlines() if '=' in line)
            if state.get('ActiveState')!='inactive' or state.get('MainPID')!='0':raise RuntimeError(unit+' must be normally stopped and have MainPID=0 before PL benchmark')
        import fcntl
        self.file=open('/run/pld-unified-fpga.lock','a+b')
        try:fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:self.file.close();self.file=None;raise
    def close(self):
        if self.file:self.file.close();self.file=None


class Engine:
    def __init__(self,source,model,model_root,backend,checkpoint=None,runtime=None,bit=None,native_copy=None,native_control=None,fuse=False):
        self.source=source_path(source);self.kind=model;self.backend=backend;self.device=None;self.lease=None
        self.model_root=Path(model_root).resolve();self.spec=digest(self.model_root/'model_spec.json')
        if self.spec!=SPECS[model]:raise ValueError('model is not the currently accepted skin/CAMUS candidate')
        self.input_shape=(1,3 if model=='skin' else 1,256,256)
        self.head_shape=(1,1 if model=='skin' else 4,256,256)
        self.provenance={'spec_sha256':self.spec,'numpy':np.__version__,'backend':backend}
        self.provenance['source_files']={str(p.relative_to(self.source)):digest(p) for p in [self.source/'reference/tinyunet.py',self.source/'camus/reference.py',self.source/'camus/graph.py',self.source/'board/pl_graph.py']}
        try:
            if model=='skin':
                from reference.tinyunet import Model
                self.model=Model(self.model_root);self.scale=self.model.layers[-1]['output_scale']
            else:
                from camus.model import load_model
                self.model=load_model(self.model_root);self.scale=self.model.head_scale
            if backend=='cpu-fp32':self._float(checkpoint)
            elif backend=='cpu-int8':
                if model=='camus':
                    from camus.reference import ReferenceNetwork
                    self.network=ReferenceNetwork(self.model)
            elif backend=='pl':
                self.lease=Lease()
                if model=='skin':
                    if bit is None:raise ValueError('skin PL requires --bit')
                    from tensor_accel import TileAccelerator
                    from pl_graph import PLGraph
                    self.device=TileAccelerator(str(bit),native_copy=native_copy,native_control=native_control,tx_range_sync=False)
                    self.network=PLGraph(self.model,self.device,fuse_maps=fuse,fuse_decoder=fuse)
                    self.provenance['bit_sha256']=digest(bit);self.provenance['hwh_sha256']=digest(Path(bit).with_suffix('.hwh'))
                else:
                    if runtime is None:raise ValueError('CAMUS PL requires --runtime')
                    from camus.accelerator import CamusTileAccelerator
                    from camus.graph import CamusGraph
                    self.device=CamusTileAccelerator(runtime,self.model_root);self.network=CamusGraph(self.model,self.device)
                    self.provenance['bit_sha256']=digest(Path(runtime)/'overlay/dma_top.bit')
                    self.provenance['hwh_sha256']=digest(Path(runtime)/'overlay/dma_top.hwh')
            else:raise ValueError('unknown backend')
        except BaseException:self.close();raise

    def _float(self,checkpoint):
        if checkpoint is None:raise ValueError('cpu-fp32 requires --checkpoint')
        if digest(checkpoint)!=CHECKPOINTS[self.kind]:raise ValueError('unrecognized FP32 checkpoint')
        import torch
        self.torch=torch;torch.set_num_threads(2)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.float_state=payload['model_state']
        self.float_names=[f'encoders.{i}.{j}' for i in range(4) for j in (0,2)]
        decoder='decoder_blocks' if self.kind=='skin' else 'decoders'
        self.float_names += [f'{decoder}.{i}.{j}' for i in range(3) for j in (0,2)]+['head']
        expected={name+suffix for name in self.float_names for suffix in ('.weight','.bias')}
        if set(self.float_state)!=expected:raise ValueError('FP32 state topology differs')
        self.provenance.update(checkpoint_sha256=digest(checkpoint),torch=torch.__version__,torch_threads=torch.get_num_threads())

    def run(self,image,frame):
        expected=np.float32 if self.backend=='cpu-fp32' else np.int8
        if image.shape!=self.input_shape or image.dtype!=expected or not np.isfinite(image).all():raise ValueError('input dtype/shape/range differs')
        if expected==np.int8 and (image==-128).any():raise ValueError('-128 excluded from quantization contract')
        if self.backend=='cpu-fp32':
            F=self.torch.nn.functional
            with self.torch.inference_mode():
                x=self.torch.from_numpy(image);outputs=[]
                for i,name in enumerate(self.float_names):
                    if i:
                        x=F.relu(outputs[-1])
                        if i in (2,4,6):x=F.max_pool2d(x,2)
                        if i in (8,10,12):x=self.torch.cat((F.relu(outputs[{8:5,10:3,12:1}[i]]),F.interpolate(x,scale_factor=2,mode='nearest')),dim=1)
                    x=F.conv2d(x,self.float_state[name+'.weight'],self.float_state[name+'.bias'],padding=0 if i==14 else 1);outputs.append(x)
                return x.cpu().numpy().copy(),[]
        if self.backend=='cpu-int8':
            if self.kind=='skin':return self.model.run(image)[-1],[]
            return self.network.forward(image)['head'],[]
        if self.kind=='skin':
            self.device.events=[];head,_=self.network.run(image,frame=frame)
            return head,list(self.device.events)
        row=self.network.forward(image);return row['head'],row['events']

    def labels(self,head,valid):
        if head.shape!=self.head_shape:raise ValueError('unexpected output shape')
        if self.kind=='skin':
            # Compare in the frozen signed-head contract, avoiding float drift.
            result=(head[0,0]>=-1) if head.dtype==np.int8 else (head[0,0]>=np.float32(np.log(.45/.55)))
            result=result.astype(np.uint8)
        else:result=head[0].argmax(axis=0).astype(np.uint8)
        result[~valid]=0
        return result

    def close(self):
        try:
            if self.device is not None:self.device.close();self.device=None
        finally:
            if self.lease is not None:self.lease.close();self.lease=None
