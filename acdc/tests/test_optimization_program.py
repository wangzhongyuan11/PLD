from dataclasses import replace
import math

import numpy as np
import pytest

from acdc.contract import SHAPES
from acdc.optimization.program import Job, Arena, compile_program, BARRIER, CAT, POOL
from acdc.optimization.image import prepare_segments, decode_head
from acdc.optimization.layout import pack_feature
from acdc.model import load_bundle, create_synthetic_bundle


@pytest.mark.parametrize('profile,count', [('P256',1040),('P384',1024)])
def test_original_grid_program_covers_all_layers_and_stays_in_arena(profile,count):
    result=compile_program(profile,base=0x10000000,epoch=17)
    arena=result['arena']
    assert len(result['jobs'])==count
    assert len(result['program'])==64*count
    assert arena.used < 6*1024*1024
    allocations=sorted(arena.buffers.values(),key=lambda x:x['address'])
    assert all(a['address']+a['bytes']<=b['address'] for a,b in zip(allocations,allocations[1:]))
    expected_producers=[0,1,3,4,6,7,9,10,8,13,5,16,2,19,20]
    for i,((ci,co,h,k),row) in enumerate(zip(SHAPES,result['layers'])):
        coverage=np.zeros((math.ceil(co/8),h,h),dtype=np.uint8)
        jobs=result['jobs'][row['first_job']:row['first_job']+row['jobs']]
        assert jobs[0].flags & BARRIER
        assert all(not j.flags & BARRIER for j in jobs[1:])
        assert len({j.producer0 for j in jobs})==1 and jobs[0].producer0==expected_producers[i]
        for j in jobs:
            assert j==Job.unpack(j.pack())
            assert j.validate(profile) is j
            assert j.ci==ci and j.epoch==17
            assert bool(j.flags & CAT)==(i in (8,10,12))
            assert bool(j.flags & POOL)==(i in (1,3,5))
            group=(j.parameters-result['addresses']['parameters']-row['parameter_offset'])//row['parameter_group_bytes']
            coverage[group,j.y:j.y+j.height,j.x:j.x+j.width]+=1
            for a in (j.source0,j.parameters,j.output):
                assert arena.base<=a<arena.base+arena.used
        assert np.all(coverage==1)
    assert sum(math.ceil(co/8)*(64+math.ceil(ci/8)*k*k*64) for ci,co,h,k in SHAPES)==493120


def test_job_rejects_unsafe_ranges_geometry_and_layout():
    j=compile_program()['jobs'][0]
    invalid=[dict(source0=j.source0+1),dict(parameters=0xfffffff8),dict(flags=128),
             dict(image_w=17),dict(x=255),dict(ci=193),dict(co=9),dict(split=8),
             dict(output_stride=8),dict(source0=1<<32),dict(epoch=True),
             dict(flags=j.flags|CAT,split=1),dict(flags=j.flags|POOL,height=3),
             dict(output=0xfffffff8),dict(height=256,width=256)]
    for fields in invalid:
        with pytest.raises(ValueError):replace(j,**fields).validate('P256')
    with pytest.raises(ValueError):Job.unpack(b'\0'*63)
    with pytest.raises(ValueError):Arena(3)
    with pytest.raises(ValueError):Arena(0xfffff000,8192)
    a=Arena(0x10000000,4096);a.allocate('x',4096)
    with pytest.raises(ValueError):a.allocate('y',1)
    with pytest.raises(ValueError):compile_program(epoch=-1)


def test_materialized_ddr_has_no_intermediate_injection_and_exact_cat_tables(tmp_path):
    model=load_bundle(create_synthetic_bundle(tmp_path/'model'),allow_synthetic=True)
    program=compile_program(epoch=7)
    x=np.random.default_rng(716).integers(-127,128,(1,3,256,256),dtype=np.int16).astype(np.int8)
    segments=prepare_segments(model,program,x)
    assert [s.name for s in segments]==['input','parameters','luts','program']
    for s in segments:
        region=program['arena'].buffers[s.name]
        assert s.address==region['address'] and len(s.data)==region['bytes']
        assert len(s.sha256)==64
    packed=np.frombuffer(segments[0].data,dtype=np.int8).reshape(256,256,8)
    assert np.array_equal(packed[:,:,:3].transpose(2,0,1),x[0]) and not np.any(packed[:,:,3:])
    tables=np.frombuffer(segments[2].data,dtype=np.int8).reshape(3,2,256)
    for i,cat in enumerate(model.concats):
        for branch,m in enumerate(cat.multiplier_q24):
            for value in range(-127,128):
                product=value*int(m)
                base,rem=divmod(abs(product),1<<24)
                q=base+int(rem>(1<<23) or (rem==(1<<23) and base%2==1))
                q=q if product>=0 else -q
                assert tables[i,branch,value&255]==max(-127,min(127,q))
    head=x[:,:1].repeat(4,axis=1)
    physical=pack_feature(head[0]).tobytes()
    assert np.array_equal(decode_head(physical),head)
    corrupted=bytearray(physical);corrupted[4]=1
    with pytest.raises(ValueError,match='padding'):decode_head(corrupted)
    with pytest.raises(ValueError):decode_head(physical[:262144])
