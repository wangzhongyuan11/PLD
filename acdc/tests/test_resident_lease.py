from unittest.mock import patch
import pytest
from acdc.appliance import resident_lease as lease


def test_running_service_requires_matching_pid_and_actual_pause():
    props=dict(ActiveState='active',MainPID='123',Result='success')
    state=dict(owner_pid=123,lifecycle='pause-resume-v1',suspended=True,hardware_released=True,
               hdmi='off',active=False,fault=None)
    with patch.object(lease,'properties',return_value=props),patch.object(lease,'status',return_value=state):
        lease.released('skin',8766,'/api/ownership')
        for invalid in [dict(owner_pid=124),dict(suspended=False),dict(hardware_released=False),
                        dict(hdmi='showing'),dict(active=True),dict(fault='DMA fault')]:
            with patch.object(lease,'status',return_value={**state,**invalid}),pytest.raises(RuntimeError):
                lease.released('skin',8766,'/api/ownership')


def test_hub_must_be_paused_and_no_evaluation_or_interactive_work():
    state=dict(owner_pid=123,phase='paused',hardware_released=True)
    with patch.object(lease,'properties',return_value=dict(ActiveState='active',MainPID='123')):
        with patch.object(lease,'status',return_value=state):lease.released('hub',8765,'/hub/ownership',hub=True)
        for invalid in [dict(phase='ready'),dict(interactive_busy=True),dict(evaluation_busy=True)]:
            with patch.object(lease,'status',return_value={**state,**invalid}),pytest.raises(RuntimeError):
                lease.released('hub',8765,'/hub/ownership',hub=True)


def test_failed_stop_cannot_be_treated_as_release():
    with patch.object(lease,'properties',return_value=dict(ActiveState='inactive',MainPID='0',Result='timeout')):
        with pytest.raises(RuntimeError):lease.released('skin',8766,'/api/ownership')
