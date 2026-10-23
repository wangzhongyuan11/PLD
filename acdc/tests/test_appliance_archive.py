import hashlib
import zipfile
import pytest
from acdc.appliance.archive import write_zip


def test_streaming_archive_roundtrip_digest_and_cancel(tmp_path):
    values={'标签.nii.gz':bytes(range(256))*1000,'mesh.vtp':b'<VTKFile>'+b'abc123'*300000,'empty':b''}
    for name,data in values.items():(tmp_path/name).write_bytes(data)
    destination=tmp_path/'result.zip'
    digest=write_zip(tmp_path,values,destination,lambda:None)
    assert digest==hashlib.sha256(destination.read_bytes()).hexdigest()
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert archive.namelist()==list(values)
        assert all(member.compress_type==zipfile.ZIP_STORED for member in archive.infolist())
        for name,data in values.items():
            assert hashlib.sha256(archive.read(name)).digest()==hashlib.sha256(data).digest()
    def cancelled():raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):write_zip(tmp_path,values,tmp_path/'cancelled.zip',cancelled)
