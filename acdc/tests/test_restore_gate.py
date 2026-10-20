import unittest
import numpy as np
from acdc.contract import ContractError
from acdc.validate_restore import compare_probabilities


class RestoreGateTest(unittest.TestCase):
    def test_small_float_error_cannot_hide_changed_label(self):
        reference=np.zeros((4,1,1,1),np.float32)
        reference[0]=.5;reference[1]=.5
        actual=reference.copy();actual[1]+=np.float32(1e-7)
        with self.assertRaisesRegex(ContractError,'label_differences=1'):
            compare_probabilities(actual,reference)

    def test_large_probability_error_rejected_even_with_same_labels(self):
        reference=np.zeros((4,1,1,1),np.float32);reference[0]=1.
        actual=reference.copy();actual[0]-=.01;actual[1]+=.01
        with self.assertRaisesRegex(ContractError,'label_differences=0'):
            compare_probabilities(actual,reference)
