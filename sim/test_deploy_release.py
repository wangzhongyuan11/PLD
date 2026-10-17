"""Deployment recovery must preserve unexpected configuration edits."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from board import deploy_release as deploy


class DeploymentRecoveryTests(unittest.TestCase):
    def fixture(self, root):
        system = root/'etc/systemd/system'; system.mkdir(parents=True)
        backup = root/'rollback/pre_pld_dual_20261017'; backup.mkdir(parents=True)
        expected = {}
        for name in deploy.UNITS:
            p = system/name; p.write_text('current '+name)
            expected[str(p)] = deploy.sha(p)
        receipt = {'release_root': str(root/'release'), 'installed_units': expected,
                   'original_files': {}, 'enabled': {}}
        (backup/'receipt.json').write_text(json.dumps(receipt))
        return system, backup, receipt

    def test_unknown_dropin_blocks_rollback_before_stopping_or_deleting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); system, backup, _ = self.fixture(root)
            p = system/'pld-web.service.d/user.conf'; p.parent.mkdir(); p.write_text('user edit')
            with patch.object(deploy, 'BASE', root), patch.object(deploy, 'SYSTEM', system), patch.object(deploy, 'stop') as stop:
                with self.assertRaisesRegex(ValueError, 'Unexpected unit/drop-in'):
                    deploy.rollback(argparse.Namespace(backup=backup, action='rollback'))
                stop.assert_not_called()
            self.assertEqual(p.read_text(), 'user edit')

    def test_corrupt_backup_blocks_rollback_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); system, backup, receipt = self.fixture(root)
            original = backup/'units/pld-hub.service'; original.parent.mkdir(); original.write_text('corrupt')
            receipt['original_files'][str(system/'pld-hub.service')] = hashlib.sha256(b'original').hexdigest()
            (backup/'receipt.json').write_text(json.dumps(receipt))
            with patch.object(deploy, 'BASE', root), patch.object(deploy, 'SYSTEM', system), patch.object(deploy, 'stop') as stop:
                with self.assertRaisesRegex(ValueError, 'Rollback file hash drift'):
                    deploy.rollback(argparse.Namespace(backup=backup, action='rollback'))
                stop.assert_not_called()
            self.assertEqual((system/'pld-hub.service').read_text(), 'current pld-hub.service')

    def test_edited_release_template_cannot_destroy_current_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); system, _, _ = self.fixture(root)
            release = root/'release'; (release/'units').mkdir(parents=True)
            hashes = {}
            for name in deploy.UNITS:
                p = release/'units'/name; p.write_text('release '+name)
                hashes['units/'+name] = deploy.sha(p)
            (release/'RELEASE_MANIFEST.json').write_text(json.dumps({'files': hashes}))
            (release/'units/pld-hub.service').write_text('unexpected')
            with patch.object(deploy, 'SYSTEM', system):
                with self.assertRaisesRegex(ValueError, 'Release unit template drift'):
                    deploy.apply_units(release, {})
            self.assertEqual((system/'pld-hub.service').read_text(), 'current pld-hub.service')
