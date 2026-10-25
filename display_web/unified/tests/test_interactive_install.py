import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import numpy as np
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('interactive_installer',SOURCE/'install_interactive.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallTests(unittest.TestCase):
    def test_failed_shutdown_prevents_replacement(self):
        with patch.object(installer,'call',return_value='ActiveState=active\nMainPID=12\nResult=success\n'):
            with self.assertRaises(RuntimeError):installer.stop_safely()

    def test_apply_and_rollback_preserve_originals_and_remove_only_new_files(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);hub=base/'hub';skin=base/'skin';package=base/'package'
            hub.mkdir();skin.mkdir();package.mkdir();(skin/'server.py').write_text('fixture')
            (hub/'installation.json').write_text(json.dumps({'backends':{'skin':{'source':str(skin)}}}))
            targets=installer.destinations(hub);manifest={};old_paths=[]
            for i,(key,target) in enumerate(targets.items()):
                target.parent.mkdir(parents=True,exist_ok=True)
                old=b'old' if i%2 or key=='skin/server.py' else None
                if old is not None:target.write_bytes(old);old_paths.append(key)
                else:target.unlink(missing_ok=True)
                payload=package/'payload'/key;payload.parent.mkdir(parents=True,exist_ok=True);payload.write_bytes(b'new')
                manifest[key]={'old':[hashlib.sha256(old).hexdigest() if old is not None else None],
                               'new':hashlib.sha256(b'new').hexdigest()}
            (package/'patch_manifest.json').write_text(json.dumps(manifest))
            preflight=types.ModuleType('prompt_seg.refinement')
            preflight.seeded_cut=lambda *args:None
            with patch.dict(sys.modules,{'prompt_seg':types.ModuleType('prompt_seg'),'prompt_seg.refinement':preflight}),patch.object(installer,'__file__',str(package/'install_interactive.py')),patch.object(installer.sys,'platform','linux'),patch.object(installer.os,'geteuid',return_value=0,create=True),patch.object(installer,'stop_safely'),patch.object(installer,'call'):
                with patch.object(sys,'argv',['install_interactive.py','--hub',str(hub),'--apply']):installer.main()
                self.assertTrue(all(p.read_bytes()==b'new' for p in targets.values()))
                backup=next(hub.glob('backup_interactive_*'))
                with patch.object(sys,'argv',['install_interactive.py','--hub',str(hub),'--rollback',str(backup)]):installer.main()
            for key,target in targets.items():
                if key in old_paths:self.assertEqual(target.read_bytes(),b'old')
                else:self.assertFalse(target.exists())


if __name__ == '__main__':unittest.main()
