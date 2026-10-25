import pytest
from acdc.contract import ContractError
from acdc.appliance.storage import ResultUsage


def test_storage_updates_only_finished_writers_and_timing(tmp_path,monkeypatch):
    for sub in ('inputs','jobs'):(tmp_path/sub).mkdir()
    a,b,c,d=(x*32 for x in 'abcd')
    state=b'{"status":"assets_ready"}'
    for job,size in ((a,13),(b,17)):
        folder=tmp_path/'jobs'/job;folder.mkdir();(folder/'asset').write_bytes(b'x'*size)
        (folder/'status.json').write_bytes(state)
    base=30+2*len(state)
    usage=ResultUsage(tmp_path,limit=base+170)
    measured=[];real=usage.measure
    def measure(job):measured.append(job);return real(job)
    monkeypatch.setattr(usage,'measure',measure)
    assert usage.admit(c,100)==base and measured==[]
    folder=tmp_path/'inputs'/c;folder.mkdir();(folder/'partial').write_bytes(b'x'*100)
    assert usage.admit(d,50)==base+100 and measured==[c]
    (tmp_path/'jobs'/a/'timing').write_bytes(b'y'*10);usage.refresh(a)
    assert sum(usage.sizes.values())==base+110
    with pytest.raises(ContractError):usage.admit('e'*32,61)
    # A Web restart rebuilds usage from disk, including an interrupted upload.
    restarted=ResultUsage(tmp_path)
    assert sum(restarted.sizes.values())==base+110
    # A restarted Web may inventory a still-running writer before its output.
    (folder/'later').write_bytes(b'z'*20)
    assert restarted.admit('f'*32,1)==base+130
