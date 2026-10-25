"""Repository integration checks; not shipped in the standalone board ZIP."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'display_web/unified'
spec = importlib.util.spec_from_file_location('package_dual_model_web', ROOT / 'scripts/package_dual_model_web.py')
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


class RepositoryTests(unittest.TestCase):
    def test_checked_in_backend_interfaces_are_accepted(self):
        known = json.loads((SOURCE / 'accepted_backends.json').read_text())
        for kind, base in [('skin', ROOT / 'display_web'), ('camus', ROOT)]:
            for name, hashes in known[kind].items():
                data = (base / name).read_bytes().replace(b'\r\n', b'\n')
                for variant in (data, data.replace(b'\n', b'\r\n')):
                    with self.subTest(kind=kind, file=name):
                        self.assertIn(hashlib.sha256(variant).hexdigest(), hashes)

    def test_package_excludes_state_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            for name in packager.FILES:
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(packager.member_path(SOURCE,name).read_bytes())
            for name in ['config.json', 'installation.json', '.access_token', 'state/last_model']:
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('must-not-be-packaged')
            first, second = root / 'one.zip', root / 'two.zip'
            packager.package(first, source)
            packager.package(second, source)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                self.assertIsNone(archive.testzip())
                names = set(archive.namelist())
                self.assertEqual(names, {'PLD_DualModel/' + name for name in packager.FILES} | {'PLD_DualModel/SHA256.json'})
                hashes = json.loads(archive.read('PLD_DualModel/SHA256.json'))
                for name, digest in hashes.items():
                    self.assertEqual(hashlib.sha256(archive.read('PLD_DualModel/' + name)).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
