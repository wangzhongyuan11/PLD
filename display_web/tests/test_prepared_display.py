from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock,patch
import server


def test_console_uses_prepared_child_without_spawning_another():
    agent=Mock();unexpected=Mock(side_effect=AssertionError('second display process'))
    with tempfile.TemporaryDirectory() as directory,patch.object(server.platform,'system',return_value='Linux'),patch.object(server.platform,'machine',return_value='armv7l'),patch.dict('os.environ',{'PLD_WARM_HDMI':'1'}),patch.dict('sys.modules',{'warm_hdmi':SimpleNamespace(WarmHDMI=unexpected)}):
        app=server.Console(Path(directory),token='test',boot_id='boot',display_agent=agent)
        assert app.display_agent is agent;unexpected.assert_not_called()
        app.close();agent.close.assert_called_once()
