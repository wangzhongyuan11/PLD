"""Full 256 network and every-node comparisons; synthetic and board scopes explicit."""
from __future__ import annotations

import numpy as np

from .contract import NODE_NAMES,ContractError,safe_member
from .graph import Graph,emulator


def reference_case(model,first=None):
    if first is not None:
        from .contract import checked_int8
        first=checked_int8(first,model.input_shape)
        check_a=False
    else:check_a=not model.synthetic
    if first is None and model.synthetic:
        first=np.random.default_rng(1234).integers(-127,128,size=model.input_shape,dtype=np.int8)
    elif first is None:
        first=np.fromfile(safe_member(model.root,model.spec['golden']['input']['path']),np.int8).reshape(model.input_shape)
    oracle=Graph(model).forward(first,capture=True)
    if check_a:
        for row in model.spec['golden']['nodes']:
            name=row['node'];a=oracle['captures'][name]
            raw=np.fromfile(safe_member(model.root,row['file']['path']),np.int8)
            if raw.size!=a.size or not np.array_equal(raw.reshape(a.shape),a):
                raise ContractError('independent oracle does not match A golden at '+name)
        expected=np.fromfile(safe_member(model.root,model.spec['golden']['head']['path']),np.int8).reshape(model.head_shape)
        if not np.array_equal(expected,oracle['head']):raise ContractError('head does not match A golden')
    return first,oracle


def verify_model(model,device=None):
    first,oracle=reference_case(model)
    actual=Graph(model,device or emulator()).forward(first,capture=True)
    for node in NODE_NAMES:
        a,b=oracle['captures'][node],actual['captures'][node]
        if not np.array_equal(a,b):
            idx=tuple(np.argwhere(a!=b)[0]);raise ContractError(f'{node}{idx}: {a[idx]} != {b[idx]}')
    return dict(synthetic=model.synthetic,shape=list(model.input_shape),nodes_compared=21,
                mismatch_count=0,max_abs_error=0,scope='independent INT64 full tensor vs tiled device; A goldens also checked when production')
