"""Display intent must survive settings changes without stale completion races."""
import tempfile
from pathlib import Path
import unittest

from test_entry import entry, Board


class DisplayIntentTests(unittest.TestCase):
    def controller(self, directory=None):
        c = entry.Controller(Board(), directory)
        c.model = 'skin'; c.phase = 'ready'
        return c

    def test_stop_invalidates_pending_and_late_completion(self):
        c = self.controller(); d = c.display
        token = d.task(dict(action='begin', model='skin', job='old'))['token']
        d.task(dict(action='display', display_action='stop'))
        d.task(dict(action='done', model='skin', job='old', token=token))
        self.assertIsNone(d.pending)
        self.assertFalse(c.settings['auto_hdmi'])

    def test_new_job_replaces_old_identity_and_roi_maps_source_job(self):
        c = self.controller(); d = c.display
        old = d.task(dict(action='begin', model='skin', job='old'))['token']
        new = d.task(dict(action='begin', model='skin', job='roi', view='interactive'))['token']
        d.task(dict(action='done', model='skin', job='old', token=old))
        self.assertIsNone(d.pending)
        d.task(dict(action='done', model='skin', job='roi', token=new, display_job='source'))
        self.assertEqual((d.pending['job'], d.pending['view']), ('source', 'interactive'))

    def test_switch_cancels_old_model_and_failure_never_queues(self):
        c = self.controller(); d = c.display
        token = d.task(dict(action='begin', model='skin', job='a'))['token']
        d.task(dict(action='failed', model='skin', job='a', token=token))
        self.assertIsNone(d.pending)
        c.model = 'camus'
        d.task(dict(action='done', model='skin', job='a', token=token))
        self.assertIsNone(d.pending)

    def test_preferences_persist_without_persisting_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.controller(directory)
            c.display.settings(dict(auto_hdmi=False, auto_segment=False))
            restored = self.controller(Path(directory))
            self.assertFalse(restored.settings['auto_hdmi'])
            self.assertFalse(restored.settings['auto_segment'])
            self.assertIsNone(restored.display.pending)

    def test_last_result_persists_but_is_not_restored_into_another_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            c=self.controller(directory);d=c.display
            d.settings(dict(auto_hdmi=False))
            job='a'*32
            token=d.task(dict(action='begin',model='skin',job=job,view='outlined',identity='version1'))['token']
            d.task(dict(action='done',model='skin',job=job,token=token))
            self.assertIsNone(d.pending)
            restored=self.controller(Path(directory));d=restored.display
            self.assertEqual(d.last_results['skin']['job'],job)
            restored.board.request=lambda *a,**kw:dict(runtime_identity='version1',state='done')
            d.settings(dict(auto_hdmi=True));d.restore('skin')
            self.assertEqual(d.pending['job'],job)
            d.cancel();restored.board.request=lambda *a,**kw:dict(runtime_identity='version2',state='done')
            d.restore('skin')
            self.assertIsNone(d.pending)
            self.assertNotIn('skin',d.last_results)


if __name__ == '__main__': unittest.main()
