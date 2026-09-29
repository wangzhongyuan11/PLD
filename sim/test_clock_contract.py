import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"board"))
from tensor_accel import expected_fclk_mhz


class ClockContractTests(unittest.TestCase):
    def test_clock_identity_must_be_unique_supported_and_finite(self):
        with tempfile.TemporaryDirectory() as folder:
            hwh=Path(folder)/"overlay.hwh"
            for value,accepted in (("100",True),("125",True),("142.857143",False),("200",False),("nan",False)):
                element=f'<PARAMETER NAME="PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ" VALUE="{value}"/>'
                hwh.write_text('<ROOT>'+element+'</ROOT>')
                if accepted: self.assertAlmostEqual(expected_fclk_mhz(hwh),float(value))
                else:
                    with self.assertRaises(ValueError): expected_fclk_mhz(hwh)
            for body in ('',element+element):
                hwh.write_text('<ROOT>'+body+'</ROOT>')
                with self.assertRaises(ValueError): expected_fclk_mhz(hwh)


if __name__=="__main__": unittest.main()
