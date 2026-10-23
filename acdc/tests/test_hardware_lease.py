import unittest
from unittest.mock import Mock, mock_open, patch

from acdc.hardware import Lease


class HardwareLeaseTest(unittest.TestCase):
    def test_formal_owner_restart_race_refuses_acdc_and_releases_handle(self):
        # Services can restart immediately after is-active. The formal lock
        # must still prevent downloading an ACDC overlay over their DMA owner.
        locks = Mock(LOCK_EX=2, LOCK_NB=4)
        locks.flock.side_effect = BlockingIOError('formal backend owns FPGA')
        opened = mock_open()
        inactive = Mock(returncode=3, stdout='inactive\n')
        with patch('acdc.hardware.sys.platform', 'linux'), \
                patch.dict('sys.modules', {'fcntl': locks}), \
                patch('acdc.hardware.subprocess.run', return_value=inactive), \
                patch('builtins.open', opened):
            with self.assertRaises(BlockingIOError):
                Lease()
        self.assertEqual(opened.call_args.args[0].as_posix(), '/run/pld-unified-fpga.lock')
        self.assertEqual(opened.call_count, 1)
        opened().close.assert_called_once()
