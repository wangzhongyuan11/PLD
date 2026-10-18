"""Content-addressed internal bundle. A's final layout can be imported without weakening CAMUS."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .contract import (ARCHITECTURE, PREPROCESSING, SHAPES, LABELS, Concat, Layer, Model,
                       ContractError, file_sha, is_sha, safe_member, positive, atomic_json, INPUT_SHAPE,HEAD_SHAPE)


def load_bundle(root, *, expected_sha=None, allow_synthetic=False):
    root = Path(root).resolve()
    spec_path = safe_member(root, 'model_spec.json')
    sha = file_sha(spec_path)
    if expected_sha is not None and (not is_sha(expected_sha) or sha != expected_sha):
        raise ContractError('model spec SHA mismatch')
    spec = json.loads(spec_path.read_text('utf-8'))
    if (spec.get('schema') != 'pld-acdc-model-v1' or spec.get('architecture_id') != ARCHITECTURE
            or spec.get('model_id') != 'acdc_axial_v1' or spec.get('orientation') != 'axial'
            or spec.get('preprocessing_id') != PREPROCESSING or spec.get('labels') != list(LABELS)):
        raise ContractError('unsupported model architecture/profile/labels/preprocessing')
    for key,wanted in [('input_shape',list(INPUT_SHAPE)),('head_shape',list(HEAD_SHAPE)),
                       ('input_scale',1/127),('zero_point',0),('layout','NCHW')]:
        if spec.get(key,wanted)!=wanted:raise ContractError('contradictory model '+key)
    synthetic = spec.get('synthetic')
    if type(synthetic) is not bool:
        raise ContractError('explicit synthetic boolean required')
    if synthetic and not allow_synthetic:
        raise ContractError('synthetic model is not production')
    if not synthetic and expected_sha is None:
        raise ContractError('production requires an externally pinned spec SHA')
    members = set()

    def read(item, dtype=None, shape=None):
        if not isinstance(item, dict) or not is_sha(item.get('sha256')):
            raise ContractError('invalid member record')
        name = item['path']
        if name in members:
            raise ContractError('duplicate bundle member')
        members.add(name)
        path = safe_member(root, name)
        if path.stat().st_size != item['bytes'] or file_sha(path) != item['sha256']:
            raise ContractError('member size/hash mismatch: '+name)
        if dtype is None:
            return path
        if item.get('dtype') != dtype or item.get('shape') != list(shape):
            raise ContractError('member dtype/shape mismatch')
        dt = np.dtype(dtype)
        if path.stat().st_size != int(np.prod(shape))*dt.itemsize:
            raise ContractError('raw tensor length mismatch')
        a = np.frombuffer(path.read_bytes(), dt).reshape(shape).copy()
        if not np.all(np.isfinite(a)) or (dt == np.int8 and np.any(a == -128)):
            raise ContractError('invalid model values')
        a.flags.writeable = False
        return a

    rows, cats = spec.get('layers', []), spec.get('concats', [])
    if len(rows) != 15 or len(cats) != 3:
        raise ContractError('expected 15 convolution / 3 CAT definitions')
    layers = []
    input_scales = [1/127]
    for i, (row, (ci, co, h, k)) in enumerate(zip(rows, SHAPES)):
        if row.get('name') != f'C{i+1:02}' or row.get('relu') is not (i < 14):
            raise ContractError('layer order/ReLU mismatch')
        for key,wanted in [('kernel',k),('padding',k//2),('stride',1),('dilation',1),('groups',1)]:
            if row.get(key,wanted)!=wanted:raise ContractError('unsupported convolution '+key)
        sx, sy = positive(row['input_scale']), positive(row['output_scale'])
        w = read(row['weight'], '|i1', (co, ci, k, k))
        b = read(row['bias'], '<i4', (co,))
        m = read(row['multiplier'], '<i4', (co,))
        sw = read(row['weight_scales'], '<f8', (co,))
        bound = np.abs(b.astype(np.int64)) + 127*np.abs(w.astype(np.int64)).sum(axis=(1,2,3))
        if np.any(bound > (1 << 31)-1) or np.any(m < 1) or np.any(sw <= 0):
            raise ContractError('accumulator/multiplier/weight-scale range error')
        derived_m = np.rint((sx*sw/sy)*(1 << 24))
        if not np.array_equal(derived_m, m.astype(np.float64)):
            raise ContractError('Q24 multiplier does not match scale derivation')
        layers.append(Layer(i, row['name'], k, k//2, i < 14, sx, sy, w,b,m,sw,
                            (1,ci,h,h), (1,co,h,h)))
    concats = []
    previous = 1/127
    for i, l in enumerate(layers):
        if i in (8,10,12):
            j = (i-8)//2
            row = cats[j]
            level = 2-j
            sources = (layers[(1,3,5)[level]].output_scale, layers[i-1].output_scale)
            common = max(sources)
            mult = tuple(int(np.rint(v/common*(1 << 24))) for v in sources)
            if (row.get('level') != level or row.get('source_scales') != list(sources)
                    or row.get('common_scale') != common or row.get('multiplier_q24') != list(mult)
                    or min(mult) <= 0):
                raise ContractError('CAT scale/order mismatch')
            concats.append(Concat(level, sources, common, mult))
            previous = common
        if l.input_scale != previous:
            raise ContractError('disconnected scale at '+l.name)
        previous = l.output_scale
    if positive(spec.get('head_scale')) != layers[-1].output_scale:
        raise ContractError('head scale mismatch')
    binding=None
    if not synthetic:
        report = json.loads(read(spec['acceptance']).read_text('utf-8'))
        validate_acceptance(report)
        read(spec['fp32_checkpoint'])
        binding=json.loads(read(spec['implementation_binding']).read_text('utf-8'))
        versions=binding.get('versions',{})
        if (not all(isinstance(versions.get(k),str) for k in ('numpy','scipy','nibabel','torch'))
                or versions.get('nibabel')!='5.4.2' or versions.get('torch')!='2.11.0'
                or binding.get('preprocessing_id')!=PREPROCESSING):
            raise ContractError('complete frozen preprocessing implementation binding required')
        golden = spec.get('golden', {})
        if set(golden) != {'input', 'head', 'nodes'} or len(golden['nodes']) != 21:
            raise ContractError('complete 21-node golden required')
        read(golden['input'], '|i1', (1,3,256,256))
        read(golden['head'], '|i1', (1,4,256,256))
        from .contract import NODE_NAMES
        if [r.get('node') for r in golden['nodes']] != list(NODE_NAMES):
            raise ContractError('golden node order mismatch')
        shapes={layer.name:layer.full_output_shape for layer in layers}
        shapes.update(P0=(1,16,128,128),P1=(1,32,64,64),P2=(1,64,32,32),
                      CAT1=(1,192,64,64),CAT2=(1,96,128,128),CAT3=(1,48,256,256))
        for row in golden['nodes']:
            read(row['file'],'|i1',shapes[row['node']])
    return Model(root, spec, tuple(layers), tuple(concats), 1/127,
                 layers[-1].output_scale, sha, synthetic,binding)


def validate_acceptance(report):
    if (report.get('status') != 'passed' or report.get('orientation') != 'axial'
            or report.get('integer_oracle_mismatches') != 0 or report.get('verified_conv_layers') != 15):
        raise ContractError('Axial integer qualification has not passed')
    rows = report.get('volumes', [])
    identities = {(r.get('patient'), r.get('phase')) for r in rows}
    patients = {r[0] for r in identities}
    if len(patients) != 20 or len(rows) != 40 or identities != {(p,t) for p in patients for t in ('ED','ES')}:
        raise ContractError('qualification requires 20 complete paired development patients')
    drops = []
    for r in rows:
        fp = np.asarray(r['fp32_dice'], dtype=float)
        qi = np.asarray(r['int8_dice'], dtype=float)
        if fp.shape != (3,) or qi.shape != (3,) or not np.all(np.isfinite([fp,qi])):
            raise ContractError('invalid Dice vectors')
        if np.any(fp < 0) or np.any(fp > 1) or np.any(qi < 0) or np.any(qi > 1):
            raise ContractError('Dice outside [0,1]')
        if float((fp-qi).mean()) > .05+1e-12 or r.get('new_empty_nonempty_gt') != 0:
            raise ContractError('per-volume quantization gate failed')
        drops.append(fp-qi)
    d = np.mean(drops, axis=0)
    if float(d.mean()) > .01+1e-12 or np.any(d > .02+1e-12):
        raise ContractError('global/class quantization gate failed')


def create_synthetic_bundle(root, *, seed=7):
    """Small deterministic sparse parameters; explicitly never production-qualified."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)

    def save(name, a):
        path = root/name
        path.write_bytes(a.tobytes(order='C'))
        return dict(path=name, bytes=a.nbytes, sha256=file_sha(path), dtype=a.dtype.str, shape=list(a.shape))

    layers=[]
    for i,(ci,co,h,k) in enumerate(SHAPES):
        w=np.zeros((co,ci,k,k),np.int8)
        for o in range(co):
            w[o,o%ci,k//2,k//2]=1 if i<14 else (1 if o%2==0 else -1)
        b=rng.integers(-2,3,size=co,dtype=np.int32)
        if i==14: b=np.array([-4,3,-2,5],np.int32)
        layers.append(dict(name=f'C{i+1:02}',relu=i<14,input_scale=1/127,output_scale=1/127,
            weight=save(f'C{i+1:02}.w.bin',w), bias=save(f'C{i+1:02}.b.bin',b.astype('<i4')),
            multiplier=save(f'C{i+1:02}.m.bin',np.full(co,1<<24,dtype='<i4')),
            weight_scales=save(f'C{i+1:02}.sw.bin',np.ones(co,dtype='<f8'))))
    spec=dict(schema='pld-acdc-model-v1',architecture_id=ARCHITECTURE,model_id='acdc_axial_v1',
              orientation='axial',preprocessing_id=PREPROCESSING,labels=list(LABELS),synthetic=True,
              layers=layers,head_scale=1/127,
              concats=[dict(level=l,source_scales=[1/127,1/127],common_scale=1/127,
                            multiplier_q24=[1<<24,1<<24]) for l in (2,1,0)])
    atomic_json(root/'model_spec.json',spec)
    return root
