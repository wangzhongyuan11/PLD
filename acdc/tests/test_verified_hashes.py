from unittest.mock import patch,Mock
from acdc import package
from acdc.appliance.verified_hashes import install


def test_reuse_unchanged_hash_but_revalidate_rewrite_and_replacement(tmp_path):
    path=tmp_path/'member';path.write_bytes(b'old')
    original=package.file_sha;calls=Mock(wraps=original)
    with patch.object(package,'file_sha',calls):
        clear=install();first=package.file_sha(path)
        assert package.file_sha(path)==first and calls.call_count==1
        path.write_bytes(b'new-longer')
        assert package.file_sha(path)!=first and calls.call_count==2
        replacement=tmp_path/'new';replacement.write_bytes(b'old');replacement.replace(path)
        assert package.file_sha(path)==first and calls.call_count==3
        clear();package.file_sha(path);assert calls.call_count==4
