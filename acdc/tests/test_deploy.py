import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from acdc.contract import ContractError
from acdc.deploy.manage import Manager


class DeployTest(unittest.TestCase):
    def test_managed_unit_only_and_explicit_restore(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);calls=[];active={'pld-hub.service','pld-web.service'}
            def run(args):
                calls.append(args)
                if args[0]=='systemd-analyze':return SimpleNamespace(returncode=0,stdout='',stderr='')
                action=args[1]
                if action=='show':
                    unit=args[2];on=unit in active
                    text=f'ActiveState={"active" if on else "inactive"}\nMainPID={1 if on else 0}\nControlGroup=\nResult=success\nExecMainStatus=0\n'
                    return SimpleNamespace(returncode=0,stdout=text,stderr='')
                if action=='stop':active.discard(args[2])
                if action=='start':active.add(args[2])
                return SimpleNamespace(returncode=0,stdout='',stderr='')
            m=Manager(root/'state',root/'pld-acdc.service',run)
            m.install('/home/xilinx/jupyter_notebooks/pld/releases/acdc_test','a'*64)
            self.assertIn('pld-hub.service',m.unit_path.read_text())
            m.activate();self.assertEqual(active,{'pld-acdc.service'})
            m.stop();self.assertEqual(active,{'pld-hub.service'})
            m.unit_path.write_text('foreign edit')
            with self.assertRaises(ContractError):m.activate()
            self.assertFalse(any('/pld-web.service' in ' '.join(c) for c in calls))
