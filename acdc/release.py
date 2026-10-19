"""Pinned A research release adapter; retains original parameter/spec identity.

Only the required, manifest-authenticated subset is needed on the board. The
source manifest stays byte-for-byte identical when importing that subset.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import numpy as np

from .contract import (ARCHITECTURE, PREPROCESSING, SHAPES, Layer, Concat, Model,
                       ContractError, file_sha, safe_member, positive)

MANIFEST_SHA = 'f4dadde385b12d1d5639ae1085ccc92d22a61682240ccfc6a4db5a3c020fca11'
AXIAL_SHA = 'fef60178abaf8f81cd3206b2c37e3ca1aef56299ee01c2f8cb3dac3c28bf33be'


def release_root(path):
    path = Path(path).resolve()
    if (path/'MANIFEST.json').is_file():
        return path
    if ((path/'model_spec.json').is_file() and path.name == 'int8'
            and path.parent.name == 'axial' and path.parents[1].name == 'models'):
        candidate = path.parents[2]
        if (candidate/'MANIFEST.json').is_file():
            return candidate
    return None


def load_release(root, *, expected_sha):
    """Load only the frozen Axial profile. Hash pinning includes all evidence."""
    from .model import validate_acceptance
    root = Path(root).resolve()
    if expected_sha != AXIAL_SHA:
        raise ContractError('A research release requires its externally pinned Axial spec SHA')
    manifest_path = safe_member(root, 'MANIFEST.json')
    if file_sha(manifest_path) != MANIFEST_SHA:
        raise ContractError('unrecognized release manifest SHA')
    manifest = json.loads(manifest_path.read_text('utf-8'))
    records = {r['path']: r for r in manifest['files']}
    if len(records) != len(manifest['files']):
        raise ContractError('duplicate release members')
    used = {'MANIFEST.json'}

    def member(name):
        row = records.get(name)
        if row is None:
            raise ContractError('unmanifested release member: '+name)
        p = safe_member(root, name)
        if p.stat().st_size != row['bytes'] or file_sha(p) != row['sha256']:
            raise ContractError('release member size/hash mismatch: '+name)
        used.add(name)
        return p

    def document(name):
        return json.loads(member(name).read_text('utf-8'))

    prefix = 'models/axial/int8/'
    source = document(prefix+'model_spec.json')
    if (file_sha(member(prefix+'model_spec.json')) != expected_sha
            or source.get('schema') != 'acdc-int8-model-v1'
            or source.get('architecture') != ARCHITECTURE
            or source.get('profile_id') != 'acdc_axial_v1'
            or source.get('direction') != 'axial' or source.get('source_kind') != 'trained'
            or source.get('input_shape') != [1,3,256,256]
            or source.get('output_shape') != [1,4,256,256]
            or source.get('input_scale') != 1/127 or source.get('head_activation') != 'none'
            or source.get('labels') != {'0':'background','1':'RV_cavity','2':'LV_myocardium','3':'LV_cavity'}
            or len(source.get('layers', [])) != 15):
        raise ContractError('A release Axial topology/identity mismatch')
    layers = []
    previous = 1/127
    cats = []
    if len(source.get('concat', [])) != 3:
        raise ContractError('three CAT definitions required')
    for i, (row, (ci,co,h,k)) in enumerate(zip(source['layers'], SHAPES)):
        if i in (8,10,12):
            j = (i-8)//2; level = 2-j; c = source['concat'][j]
            scales = (layers[2*level+1].output_scale, previous)
            common = max(scales)
            multipliers = tuple(int(np.rint(v/common*2**24)) for v in scales)
            if (c.get('level') != level or c.get('common_scale') != common
                    or (c.get('skip_multiplier_q24'),c.get('deep_multiplier_q24')) != multipliers
                    or c.get('skip_first') is not True):
                raise ContractError('A release CAT scale/order mismatch')
            cats.append(Concat(level, scales, common, multipliers))
            previous = common
        if (row.get('id') != f'C{i+1:02}' or row.get('weight_shape') != [co,ci,k,k]
                or row.get('kernel') != k or row.get('padding') != k//2
                or row.get('relu') is not (i<14) or row.get('input_scale') != previous):
            raise ContractError('A release disconnected convolution: '+str(row.get('id')))
        arrays = {}
        for name,dtype,count in [('weight.bin','|i1',co*ci*k*k),('bias.bin','<i4',co),('multipliers.bin','<i4',co)]:
            desc = row['files'][name]; p = member(prefix+desc['path'])
            if p.stat().st_size != count*np.dtype(dtype).itemsize or file_sha(p) != desc['sha256']:
                raise ContractError('A parameter length/hash mismatch')
            arrays[name] = np.frombuffer(p.read_bytes(),dtype).copy()
        w = arrays['weight.bin'].reshape(co,ci,k,k); b = arrays['bias.bin']; m = arrays['multipliers.bin']
        sw = np.asarray(row['weight_scales'],np.float64)
        sx,sy = positive(row['input_scale']),positive(row['output_scale'])
        bounds = np.abs(b.astype(np.int64))+127*np.abs(w.astype(np.int64)).sum((1,2,3))
        if (sw.shape != (co,) or not np.all(np.isfinite(sw)) or np.any(sw<=0)
                or np.any(w==-128) or np.any(m<1) or np.any(bounds>2**31-1)
                or np.any(bounds*m.astype(np.int64)>2**63-1)
                or not np.array_equal(bounds,row['conservative_bound'])
                or not np.array_equal(m,np.rint(sx*sw/sy*2**24))):
            raise ContractError('A release integer bounds/Q24 mismatch')
        for a in (w,b,m,sw):a.flags.writeable=False
        layers.append(Layer(i,f'C{i+1:02}',k,k//2,i<14,sx,sy,w,b,m,sw,(1,ci,h,h),(1,co,h,h)))
        previous = sy
    if source['head_scale'] != previous:
        raise ContractError('A head scale mismatch')

    acceptance = document('validation/profiles/axial/ACCEPTANCE.json')
    full = document('validation/profiles/axial/INTEGER_FULL_GRAPH_ACCEPTANCE.json')
    fp = document('validation/profiles/axial/fp32.json')
    qi = document('validation/profiles/axial/int8.json')
    if (acceptance.get('state') != 'ACCEPTED_NATIVE_DEV' or acceptance.get('spec_sha256') != expected_sha
            or full.get('spec_sha256') != expected_sha or full.get('different_elements') != 0
            or full.get('compared_elements') != 262144 or full.get('independent_layer_oracles') != 15
            or full.get('all_layer_differences_zero') is not True
            or qi.get('model_identity') != expected_sha
            or fp.get('model_identity') != source['source_checkpoint_sha256']):
        raise ContractError('A software acceptance identity mismatch')
    split = document('docs/acdc/PATIENT_SPLIT.json')
    paired = {}
    for label,report in [('fp32',fp),('int8',qi)]:
        identities = set()
        for row in report['rows']:
            key = (row['patient'],row['phase'])
            if key in identities:raise ContractError('duplicate qualification volume')
            identities.add(key);paired.setdefault(key,{})[label]=row['metrics']
        if identities != {(p,t) for p in split['dev'] for t in ('ED','ES')}:
            raise ContractError('qualification split mismatch')
    rows=[]
    for (patient,phase),reports in paired.items():
        f,q = reports['fp32'],reports['int8']
        new_empty = sum(not f[str(c)]['pred_empty'] and q[str(c)]['pred_empty']
                        and not q[str(c)]['reference_empty'] for c in (1,2,3))
        rows.append(dict(patient=patient,phase=phase,fp32_dice=[f[str(c)]['dice'] for c in (1,2,3)],
                         int8_dice=[q[str(c)]['dice'] for c in (1,2,3)],new_empty_nonempty_gt=new_empty))
    validate_acceptance(dict(status='passed',orientation='axial',integer_oracle_mismatches=0,
                             verified_conv_layers=15,volumes=rows))
    checkpoint = member('models/axial/model_state_fp32.pt')
    if file_sha(checkpoint) != source['source_checkpoint_sha256']:
        raise ContractError('FP32 source checkpoint mismatch')
    prep = document('docs/acdc/PREPROCESSING_CONTRACT.json')
    for name,sha in prep['implementation_binding']['files_sha256'].items():
        if file_sha(member(name)) != sha:
            raise ContractError('A preprocessing implementation binding mismatch')
    for name in ('licenses/LICENSE_TERMS.md','licenses/MANDATORY_CITATION.md'):
        member(name)
    binding = dict(preprocessing_id=PREPROCESSING,versions=prep['implementation_binding']['libraries'],
                   reference_python=prep['implementation_binding']['python'])
    # ACDC origin is authenticated by this pinned manifest, never by a boolean.
    binding['recognized_acdc_images'] = {r['sha256']: r['path'] for r in manifest['files']
        if r['path'].startswith('datasets/acdc/training/') and r['path'].endswith('.nii.gz')
        and not r['path'].endswith('_gt.nii.gz')}
    golden = document(prefix+'golden.json')
    descs={}
    for key,name,shape in [('input','golden_input.bin',[1,3,256,256]),('head','golden_head.bin',[1,4,256,256])]:
        p=member(prefix+name)
        if file_sha(p) != golden[key+'_sha256'] or p.stat().st_size != int(np.prod(shape)):
            raise ContractError('A golden identity mismatch')
        descs[key]=dict(path=prefix+name,sha256=file_sha(p),bytes=p.stat().st_size,dtype='|i1',shape=shape)
    if golden['spec_sha256'] != expected_sha or golden['head_scale'] != previous:
        raise ContractError('A golden model mismatch')
    spec=dict(source,source_model_spec_sha256=expected_sha,source_manifest_sha256=MANIFEST_SHA,
              golden=dict(descs,nodes=[]),golden_evidence_kind='A-delivered-input-head-only',
              qualification=dict(software_development_accepted=True,physical_board_accepted=False,production_ready=False))
    # Required files are complete even when the original dataset is not copied.
    spec['release_members']=sorted(used)
    return Model(root,spec,tuple(layers),tuple(cats),1/127,previous,expected_sha,False,binding)


def import_release(source, output, *, expected_sha=AXIAL_SHA):
    root = release_root(source)
    if root is None:raise ContractError('not an A release root or Axial int8 folder')
    model = load_release(root,expected_sha=expected_sha)
    output=Path(output).resolve()
    if output.exists() or output.is_relative_to(root):
        raise ContractError('choose a new import directory outside the source release')
    output.mkdir(parents=True)
    for name in model.spec['release_members']:
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(safe_member(root,name),target)
    # No modification of A spec, parameters or manifest; revalidate the subset.
    return load_release(output,expected_sha=expected_sha)
