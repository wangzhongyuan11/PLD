import unittest
from perf_eval.core import summarize,dma_summary
from perf_eval.evidence import integrate_power

class CoreTests(unittest.TestCase):
    def test_fps_is_count_over_total(self):
        r=summarize([1.,3.],5.)
        self.assertEqual(r['active_fps'],.5);self.assertEqual(r['wall_throughput_fps'],.4);self.assertAlmostEqual(r['p95_seconds'],2.9)
    def test_bad_times_rejected(self):
        for value in (0,-1,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):summarize([value])
        with self.assertRaises(ValueError):summarize([2.],1.)
    def test_dma_packet_accounting(self):
        r=dma_summary([dict(command=1,bytes=100,dma_seconds=.1),dict(command=2,bytes=50),dict(command=4,bytes=30)],2.)
        self.assertEqual((r['tx_bytes'],r['rx_bytes'],r['total_packet_bytes']),(150,30,180));self.assertEqual(r['estimated_dma_MB_s'],.00009)
        with self.assertRaises(ValueError):dma_summary([dict(command=17,bytes=50)],1.)
    def test_trapezoid_energy(self):
        r=integrate_power(dict(scope='board_input',frames=2,samples=[dict(time_s=0,power_w=2),dict(time_s=1,power_w=4),dict(time_s=2,power_w=2)],idle_power_w=1))
        self.assertEqual(r['energy_joules'],6);self.assertEqual(r['joules_per_frame'],3);self.assertEqual(r['incremental_joules_per_frame'],2)
    def test_power_bad_samples(self):
        with self.assertRaises(ValueError):integrate_power(dict(scope='board_input',frames=1,samples=[dict(time_s=1,power_w=1),dict(time_s=0,power_w=1)]))

if __name__=='__main__':unittest.main()
