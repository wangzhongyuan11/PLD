"""CPU cache retains verified bytes, never old physical allocations."""
import hashlib
import json
import struct
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pytest
from board_image import PreparedArtifacts, build_image
from frame_driver import FrameSession
from test_frame_driver import Buffer, MMIO


REGIONS={'input':(0x1000000,0x100000),'arena':(0x2000000,0x1000000),'weights':(0x4000000,0x100000),
         'luts':(0x5000000,0x10000),'descriptors':(0x6000000,0x100000),'head':(0x7000000,0x100000)}


def model_fixture(root):
    artifacts={}; layers=[]; descriptors=[]
    def artifact(name,data):
        (root/name).write_bytes(data)
        artifacts[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    artifact('first_input_nchwc8.bin',bytes(128)); artifact('golden_head_nchwc8.bin',bytes(128))
    for li in range(15):
        artifact(f'{li:02d}_weights.bin',bytes([li+1])*64)
        artifact(f'{li:02d}_luts.bin',bytes([li+2])*512)
        artifact(f'{li:02d}_params.bin',bytes([li+3])*64)
        kernel=3 if li%2==0 else 1
        layers.append(dict(input_groups=1,kernel=kernel,input_shape=[1,2,8],output_shape=[1,2,8],weight_block_bytes=64))
        for y in range(2):
            descriptors.append(dict(layer=li,groups=1,kernel=kernel,width=8,height=2,y=y,rows=1,
                branches=[dict(groups=1,source_offset=li*128,mode=0,table=0)],
                destination_offset=(li+1)*128+y*64,weights_file=f'{li:02d}_weights.bin',weights_offset=0,
                params_file=f'{li:02d}_params.bin',params_offset=0,relu=li!=14,
                stripe_epoch_within_frame=y,reuse_features=False,load_luts=True))
    (root/'manifest.json').write_text(json.dumps(dict(artifacts=artifacts,layers=layers)))
    (root/'descriptors.json').write_text(json.dumps(dict(arena_bytes=4096,descriptors=descriptors)))
    return root


def test_prepared_image_bytes_relocate_without_rereading_files(tmp_path):
    root=model_fixture(tmp_path); prepared=PreparedArtifacts(root)
    counts={}; original=Path.read_bytes
    def read(path):
        counts[path.name]=counts.get(path.name,0)+1
        return original(path)
    with patch.object(Path,'read_bytes',read):
        image=build_image(root,REGIONS,prepared)
        again=build_image(root,REGIONS,prepared)
        assert image==again
        assert all(count==1 for count in counts.values())
        assert counts['00_params.bin']==1  # used by two stripe descriptors
    assert image['weights']==b''.join(bytes([li+1])*64 for li in range(15))
    assert image['luts']==b''.join(bytes([li+2])*512 for li in range(15))
    first=image['descriptors'][:128]; last=image['descriptors'][-128:]
    assert struct.unpack_from('<4I',first)==(REGIONS['input'][0],0,REGIONS['weights'][0],REGIONS['arena'][0]+128)
    assert struct.unpack_from('<I',last,12)[0]==REGIONS['head'][0]+64
    assert first[48:112]==bytes([3])*64 and last[48:112]==bytes([17])*64
    moved={key:(base+0x8000000,size) for key,(base,size) in REGIONS.items()}
    relocated=build_image(root,moved,prepared)
    assert struct.unpack_from('<I',relocated['descriptors'])[0]==moved['input'][0]
    assert relocated['weights']==image['weights']


@pytest.mark.parametrize('name',['manifest.json','descriptors.json','00_weights.bin','00_params.bin'])
def test_changed_artifacts_rejected_before_new_allocations(tmp_path,name):
    root=model_fixture(tmp_path); prepared=PreparedArtifacts(root); build_image(root,REGIONS,prepared)
    path=root/name; data=path.read_bytes();path.write_bytes(data+b'x')
    with pytest.raises(ValueError,match='frozen artifact changed'):
        FrameSession(root,MMIO(),lambda **kw:pytest.fail('allocation before integrity check'),prepared=prepared)


def test_invalid_hash_is_not_cached(tmp_path):
    root=model_fixture(tmp_path);(root/'00_params.bin').write_bytes(bytes([99])*64)
    prepared=PreparedArtifacts(root)
    with pytest.raises(ValueError,match='integrity failure'):build_image(root,REGIONS,prepared)
    assert '00_params.bin' not in prepared.files


def test_two_sessions_share_cpu_bytes_but_not_dma_and_timeout_retains_buffers(tmp_path):
    root=model_fixture(tmp_path); allocated=[]; address=0x1000000
    def allocate(shape,dtype):
        nonlocal address
        buf=Buffer(shape[0],address);address+=(shape[0]+4095)//4096*4096
        allocated.append(buf);return buf
    first=FrameSession(root,MMIO(),allocate); old=list(allocated); first.close()
    assert all(buf.freed for buf in old)
    mmio=MMIO();second=FrameSession(root,mmio,allocate,prepared=first.prepared)
    assert second.prepared is first.prepared
    assert all(second.buffers[name] is not first.buffers[name] for name in second.buffers)
    second.buffers['head'][:]=7
    result,_=second.run(np.zeros((1,2,8),np.int8)); second.buffers['head'][:]=9
    assert np.all(result==7)
    mmio.stuck=True
    with pytest.raises(TimeoutError):second.run(np.zeros((1,2,8),np.int8),timeout=.001)
    with pytest.raises(RuntimeError):second.close()
    assert not any(buf.freed for buf in allocated[len(old):])
    mmio.status=0;second.release_after_verified_reset()
    assert all(buf.freed for buf in allocated)


def test_skin_worker_preserves_only_software_cache_across_display_release(monkeypatch,tmp_path):
    import web_entry
    calls=[]
    class Core:
        def __init__(self,kind,prepared=None):
            calls.append(prepared)
            self.session=SimpleNamespace(prepared=prepared if prepared is not None else object())
            self.startup_timing=dict(stages_seconds=dict(frame_session=.1))
        def close(self):pass
    monkeypatch.setattr(web_entry,'Core',Core)
    worker=web_entry.SkinWorker(tmp_path)
    first=worker._run(['--warmup-only'],tmp_path/'unused.log')
    cached=worker.prepared
    worker.release_hardware()
    assert worker.core is None and worker.prepared is cached
    resumed=worker._run(['--warmup-only'],tmp_path/'unused.log')
    assert not first['artifacts_reused'] and resumed['artifacts_reused']
    assert not resumed['hardware_reused'] and resumed['core_startup_stages']=={'frame_session':.1}
    hot=worker._run(['--warmup-only'],tmp_path/'unused.log')
    assert hot['hardware_reused'] and hot['core_startup_stages']=={}
    assert calls==[None,cached]
    worker.close(); assert worker.core is None and worker.prepared is None
