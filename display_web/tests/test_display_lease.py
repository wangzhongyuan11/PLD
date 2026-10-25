import json
from types import SimpleNamespace
from unittest.mock import Mock,patch
import pytest
from display_web import display_lease as lease


def test_active_skin_is_allowed_only_after_matching_release_ack():
    state=dict(owner_pid=123,lifecycle='pause-resume-v1',suspended=True,hardware_released=True,hdmi='off')
    response=Mock(status=200);connection=Mock();connection.getresponse.return_value=response
    props=SimpleNamespace(stdout='ActiveState=active\nMainPID=123\nResult=success\n')
    with patch.object(lease.subprocess,'run',return_value=props),patch.object(lease.http.client,'HTTPConnection',return_value=connection):
        response.read.return_value=json.dumps(state).encode();lease.skin_released()
        for invalid in [dict(owner_pid=456),dict(hardware_released=False),dict(active=True),dict(fault='DMA fault')]:
            response.read.return_value=json.dumps({**state,**invalid}).encode()
            with pytest.raises(RuntimeError):lease.skin_released()
    connection.close.assert_called()
