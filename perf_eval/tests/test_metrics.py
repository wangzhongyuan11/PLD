import unittest
import numpy as np
from perf_eval.metrics import confusion,scores,array_difference

class MetricsTests(unittest.TestCase):
    def test_absent_not_perfect(self):
        r=scores(confusion(np.zeros((2,2),np.uint8),np.zeros((2,2),np.uint8),4))
        self.assertIsNone(r['foreground_macro_dice']);self.assertIsNone(r['per_class'][1]['dice'])
    def test_dice_and_valid(self):
        p=np.array([[1,1],[0,0]],np.uint8);g=np.array([[1,0],[0,0]],np.uint8)
        self.assertAlmostEqual(scores(confusion(p,g,2))['foreground_macro_dice'],2/3)
        keep=np.array([[True,False],[True,True]])
        self.assertEqual(scores(confusion(p,g,2,keep))['foreground_macro_dice'],1)
    def test_class_shape_errors(self):
        for pred,gt in [(np.zeros((3,3),np.uint8),np.zeros((2,2),np.uint8)),(np.full((2,2),85,np.uint8),np.zeros((2,2),np.uint8))]:
            with self.assertRaises(ValueError):confusion(pred,gt,4)
    def test_unsigned_abs_difference_not_wrapped(self):
        r=array_difference(np.array([0],np.uint8),np.array([255],np.uint8));self.assertEqual(r['max_abs_error'],255)

if __name__=='__main__':unittest.main()
