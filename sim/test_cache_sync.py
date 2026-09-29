"""Verify exact TX sync bounds and allocation retention without hardware."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'board'))
from cache_sync import XrtTXPrefix


class Tests(unittest.TestCase):
    def test_prefix_rounds_outward_and_retains_owner(self):
        calls=[]
        root=SimpleNamespace(offset=0,coherent=False,physical_address=0x100000,nbytes=131072,
                             bo=SimpleNamespace(sync=lambda *args:calls.append(args)))
        for length, expected in ((1,64),(64,64),(65,128),(280,320),(131072,131072)):
            prefix=XrtTXPrefix(root,length,7)
            self.assertIs(prefix.buffer,root)
            self.assertEqual(prefix.nbytes,length)
            prefix.flush()
            self.assertEqual(calls[-1],(7,expected,0))
            with self.assertRaises(RuntimeError): prefix.invalidate()
        for bad in (0,-1,131073,1.5):
            with self.assertRaises((ValueError,TypeError)): XrtTXPrefix(root,bad,7)
        for key,value in (('offset',64),('coherent',True),('physical_address',0x100008),('nbytes',131071)):
            invalid=SimpleNamespace(**vars(root)); setattr(invalid,key,value)
            with self.assertRaises(ValueError): XrtTXPrefix(invalid,64,7)


if __name__=='__main__': unittest.main()
