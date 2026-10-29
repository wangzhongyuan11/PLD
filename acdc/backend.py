"""Explicit execution modes. No automatic CPU fallback from PL/remote."""
from __future__ import annotations

import http.client
import time
from urllib.parse import urlsplit
import numpy as np

from .contract import ContractError, INPUT_SHAPE, HEAD_SHAPE, HEAD_BYTES, checked_int8, digest,file_sha,is_sha
from .graph import Graph,emulator
from .protocol import encode,decode,validate_request,response_meta


class GraphBackend:
    def __init__(self,model,kind='reference',runtime=None):
        begin=time.perf_counter()
        self.model=model;self.kind=kind;self.synthetic=model.synthetic
        self.model_sha=model.model_sha256;self.head_scale=model.head_scale
        self.binding=model.implementation_binding
        self.runtime_sha=None;self.hardware_info=None
        if kind=='reference':device=None
        elif kind=='emulator':device=emulator()
        elif kind=='pl':
            from .hardware import TileAccelerator
            import json
            from pathlib import Path
            if runtime is None:raise ContractError('PL requires a certified runtime')
            self.runtime_sha=file_sha(Path(runtime)/'runtime_manifest.json')
            record=json.loads((Path(runtime)/'overlay/build_manifest.json').read_text('utf-8'))
            self.hardware_info={k:record.get(k) for k in ('hardware_id','abi','compute_lanes','clock_mhz','resources','routed_timing')}
            device=TileAccelerator(runtime,model)
        else:raise ContractError('unknown graph backend')
        self.graph=Graph(model,device)
        self.initialization_seconds=time.perf_counter()-begin

    def infer(self,x,request,cancel=None):
        validate_request(request)
        if request['model_sha256']!=self.model_sha:
            raise ContractError('requested model differs from loaded model')
        if request.get('runtime_sha256')!=self.runtime_sha:
            raise ContractError('requested runtime differs from loaded runtime')
        started=time.perf_counter()
        result=self.graph.forward(x,cancel=cancel)
        meta=response_meta(request,self.head_scale,self.kind,self.synthetic)
        meta.update(nodes=result['nodes'],events=result['events'],slice_seconds=time.perf_counter()-started,
                    runtime_sha256=self.runtime_sha,hardware=self.hardware_info)
        return result['head'],meta

    def close(self):self.graph.close()


class DemoBackend:
    """Analytic intensity illustration. Does NOT execute the 15-layer neural network."""
    kind='demo'
    synthetic=True
    model_sha=digest({'backend':'analytic-demo-v1','synthetic':True})
    head_scale=.125

    def infer(self,x,request,cancel=None):
        validate_request(request)
        if request['model_sha256']!=self.model_sha:
            raise ContractError('demo identity mismatch')
        x=checked_int8(x,INPUT_SHAPE)
        center=x[0,1].astype(np.float32)
        means=np.array([-125,-52,37,125],np.float32)
        q=np.clip(np.rint(-np.abs(center[None]-means[:,None,None])/2),-127,127).astype(np.int8)[None]
        meta=response_meta(request,self.head_scale,self.kind,True)
        meta.update(nodes=[],slice_seconds=0.,notice='Synthetic analytic illustration; no neural inference or diagnostic accuracy claim')
        return q,meta

    def close(self):pass


class SpatialBackend:
    kind='pl'
    def __init__(self,model,runtime,*,verify=True):
        from .optimization.device import SpatialDevice
        import json
        from pathlib import Path
        begin=time.perf_counter()
        self.model=model;self.model_sha=model.model_sha256;self.head_scale=model.head_scale
        self.synthetic=model.synthetic;self.binding=model.implementation_binding
        self.verify=verify
        self.runtime_sha=(file_sha(Path(runtime)/'runtime_manifest.json') if verify else
                          json.loads((Path(runtime)/'v2_application.json').read_text())['core_runtime_sha256'])
        self.hardware_info=json.loads((Path(runtime)/'overlay/build_manifest.json').read_text('utf-8'))
        self.device=SpatialDevice(runtime,model,verify=verify)
        self.initialization_seconds=time.perf_counter()-begin

    def infer(self,x,request,cancel=None):
        if self.verify:validate_request(request)
        if self.verify and (request['model_sha256']!=self.model_sha or request.get('runtime_sha256')!=self.runtime_sha):
            raise ContractError('spatial request model/runtime mismatch')
        head,timing=self.device.forward(x,cancel=cancel)
        meta=response_meta(request,self.head_scale,self.kind,self.synthetic)
        meta.update(timing,nodes=[],events=[],runtime_sha256=self.runtime_sha,
                    hardware={k:self.hardware_info[k] for k in ('hardware_id','abi','spatial_p','clock_mhz')})
        return head,meta

    def close(self):self.device.close()


class TorchReferenceBackend:
    """Explicit PC validation backend, exact integer arithmetic via FP64 conv.

    Never selected as a PL fallback. The independent INT64 oracle remains in
    integer.py; verify-model compares all nodes of both implementations.
    """
    kind='reference_torch'
    def __init__(self,model):
        import torch
        begin=time.perf_counter();torch.set_num_threads(4)
        self.model=model;self.model_sha=model.model_sha256;self.head_scale=model.head_scale
        self.synthetic=model.synthetic;self.binding=model.implementation_binding
        self.runtime_sha=None;self.hardware_info=None
        self.initialization_seconds=time.perf_counter()-begin

    def infer(self,x,request,cancel=None):
        from .witness import forward
        validate_request(request)
        if request['model_sha256']!=self.model_sha or request.get('runtime_sha256') is not None:
            raise ContractError('reference request model/runtime mismatch')
        if cancel is not None and cancel.is_set():raise InterruptedError('reference cancelled')
        begin=time.perf_counter();captures=forward(self.model,x)
        meta=response_meta(request,self.head_scale,self.kind,self.synthetic)
        meta.update(nodes=[],events=[],slice_seconds=time.perf_counter()-begin,
                    arithmetic='exact integer witness using FP64 convolution under proven INT32 bounds',runtime_sha256=None)
        return captures['C15'],meta

    def close(self):pass


class RemoteBackend:
    kind='remote'
    def __init__(self,url,*,model_sha,head_scale,synthetic=False,timeout=30,binding=None):
        address=urlsplit(url)
        if address.scheme!='http' or not address.hostname or address.path not in ('','/'):
            raise ContractError('board URL must be http://host:port; use SSH tunnel for remote access')
        self.connection=http.client.HTTPConnection(address.hostname,address.port or 8093,timeout=timeout)
        self.model_sha=model_sha;self.head_scale=head_scale;self.synthetic=synthetic
        self.binding=binding
        import json
        self.connection.request('GET','/health')
        response=self.connection.getresponse();raw=response.read(65537)
        if response.status!=200 or len(raw)>65536:raise ContractError('worker health handshake failed')
        health=json.loads(raw)
        if (health.get('model_sha256')!=model_sha or health.get('head_scale')!=head_scale
                or health.get('synthetic') is not synthetic or (not synthetic and health.get('backend')!='pl')):
            self.connection.close();raise ContractError('worker/model provenance handshake mismatch')
        self.runtime_sha=health.get('runtime_sha256');self.hardware_info=health.get('hardware')
        supported={(0x41434433,0x00030000),(0x41434434,0x00030000),(0x41434438,0x00020000)}
        if not synthetic and (not is_sha(self.runtime_sha) or not isinstance(self.hardware_info,dict)
                or (self.hardware_info.get('hardware_id'),self.hardware_info.get('abi')) not in supported):
            self.connection.close();raise ContractError('production worker lacks certified ACDC hardware/runtime provenance')
        # Preparing/uploading a volume may exceed the worker's 30 s idle limit.
        # Do not retain the health socket until the first actual slice.
        self.connection.close()
        self._last_response=None

    def infer(self,x,request,cancel=None):
        x=checked_int8(x,INPUT_SHAPE);validate_request(request)
        if cancel is not None and cancel.is_set():raise InterruptedError('remote cancelled')
        if self._last_response is not None and time.monotonic()-self._last_response>=20:
            self.connection.close()
        try:
            self.connection.request('POST','/infer',encode(request,x.tobytes()),{'Content-Type':'application/octet-stream'})
            response=self.connection.getresponse()
            raw=response.read(HEAD_BYTES+65536+5)
        except (OSError,http.client.HTTPException):
            # Delivery is uncertain: report failure, never silently replay PL.
            # A later user job must be able to establish a clean connection.
            self.connection.close();self._last_response=None
            raise
        self._last_response=time.monotonic()
        if response.status!=200:
            raise RuntimeError(f'board refused inference: {response.status} {raw[:200]!r}')
        meta,payload=decode(raw,HEAD_BYTES)
        for key in ('architecture_id','execution_profile','preprocessing_id','run_id','phase','orientation',
                    'center_index','neighbor_indices','total_slices','geometry_sha256','model_sha256'):
            if meta.get(key)!=request.get(key):raise ContractError('board response identity mismatch: '+key)
        if (meta.get('head_scale')!=self.head_scale or meta.get('status')!='completed' or meta.get('synthetic') is not self.synthetic
                or meta.get('shape')!=list(HEAD_SHAPE) or meta.get('dtype')!='int8' or meta.get('layout')!='NCHW'):
            raise ContractError('board response provenance mismatch')
        if meta.get('runtime_sha256')!=self.runtime_sha:raise ContractError('worker runtime changed during job')
        if not self.synthetic and meta.get('backend')!='pl':
            raise ContractError('production remote worker did not execute PL')
        return checked_int8(np.frombuffer(payload,np.int8).reshape(HEAD_SHAPE)).copy(),meta

    def close(self):self.connection.close()
