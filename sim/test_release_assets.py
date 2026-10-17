"""Regression gates for safe offline preparation of the formal delivery."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import release_assets


class ReleaseAssetsTests(unittest.TestCase):
    def archive(self, root, members):
        archive = root/'assets.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            for name, data in members.items(): z.writestr(name, data)
        manifest = {'release': 'test', 'archive_sha256': release_assets.digest(archive),
                    'files': {n: hashlib.sha256(v).hexdigest() for n, v in members.items()}}
        (root/'MANIFEST.json').write_text(json.dumps(manifest))
        return archive

    def test_conflicting_model_is_preserved_and_no_partial_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self.archive(root, {'models/role_a_final/first': b'first',
                                          'models/role_a_final/conflict': b'expected'})
            target = root/'checkout/role_a_final'; target.mkdir(parents=True)
            (target/'conflict').write_bytes(b'user work')
            with patch.object(release_assets, 'RELEASE', root):
                with self.assertRaisesRegex(ValueError, 'Existing file differs'):
                    release_assets.prepare(archive, root/'checkout')
            self.assertFalse((target/'first').exists())
            self.assertEqual((target/'conflict').read_bytes(), b'user work')

    def test_even_hash_matching_archive_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self.archive(root, {'models/../../escape': b'payload'})
            with patch.object(release_assets, 'RELEASE', root):
                with self.assertRaisesRegex(ValueError, 'Unsafe asset member'):
                    release_assets.prepare(archive, root/'checkout')
            self.assertFalse((root/'escape').exists())

    def test_archive_corruption_fails_before_extracting_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self.archive(root, {'models/role_a_final/model': b'frozen'})
            with archive.open('ab') as f: f.write(b'changed')
            with patch.object(release_assets, 'RELEASE', root):
                with self.assertRaisesRegex(ValueError, 'Archive SHA256 mismatch'):
                    release_assets.prepare(archive, root/'checkout')
            self.assertFalse((root/'checkout').exists())
