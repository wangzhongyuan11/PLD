import unittest
from unittest.mock import Mock
from types import SimpleNamespace
from test_entry import entry, Board

class BootPreparation(unittest.TestCase):
    def make(self):
        b=Board();b.health=lambda:dict(lifecycle='pause-resume-v1',resident_handoff='resident-hub-v1')
        b.hub=lambda:dict(lifecycle='resident-hub-v1',backend=dict(lifecycle='pause-resume-v1'))
        b.can_park_heart=lambda:True;b.running=lambda unit:True
        c=entry.Controller(b);c.display.restore=Mock();c.boot['state']='running'
        return c
    def test_order_default_and_ready_gate(self):
        for default in entry.MODELS:
            c=self.make();seen=[];c.settings['last_model']=default
            def switch(target):
                seen.append(target);c.model=target;c.phase='ready'
                self.assertEqual(c.snapshot()['phase'],'preparing')
                with self.assertRaises(RuntimeError):c.switch('skin')
            c._switch=switch;c._prepare_boot(default)
            self.assertEqual(seen[0],'acdc');self.assertEqual(len(seen),4 if default=='acdc' else 3);self.assertEqual(set(seen),set(entry.MODELS))
            self.assertEqual(seen[-1],default);self.assertEqual(c.boot['state'],'ready')
            self.assertEqual(c.settings['last_model'],default);c.display.restore.assert_called_once_with(default)
    def test_failure_stops_sequence(self):
        c=self.make();seen=[]
        def fail(target):seen.append(target);c.phase='error';c.error='release failed'
        c._switch=fail;c._prepare_boot('acdc')
        self.assertEqual(seen,['acdc']);self.assertEqual(c.boot['state'],'error');c.display.restore.assert_not_called()
    def test_old_lifecycle_rejected(self):
        c=self.make();c.board.health=lambda:{}
        c._switch=lambda target:setattr(c,'phase','ready');c._prepare_boot('acdc')
        self.assertEqual(c.boot['state'],'error');c.display.restore.assert_not_called()
    def test_stopped_service_rejected(self):
        c=self.make();c.board.running=lambda unit:False
        c._switch=lambda target:setattr(c,'phase','ready');c._prepare_boot('acdc')
        self.assertEqual(c.boot['state'],'error');c.display.restore.assert_not_called()
    def test_shutdown_prevents_next_model(self):
        c=self.make();c.stopping=True;c._switch=Mock();c._prepare_boot('acdc');c._switch.assert_not_called()
    def test_intermediate_switch_does_not_save_or_display(self):
        c=entry.Controller(Board());c.boot['state']='running';c.settings['last_model']='skin'
        c.display.restore=Mock();c.save_settings=Mock();c._switch('acdc')
        self.assertEqual(c.settings['last_model'],'skin');c.save_settings.assert_not_called();c.display.restore.assert_not_called()


class ColdResidentBoard(Board):
    """Cold services with one simulated hardware owner; uses real _switch."""
    def __init__(self):
        super().__init__()
        self.states={unit:False for unit in self.states}
        self.owner=None;self.heart_paused=False;self.hub_paused=False
        self.generation=0

    def start(self, unit):
        self.events.append(('start',unit))
        if unit==entry.WEB and not self.states[entry.ENGINE]:
            assert self.owner is None
            self.states[entry.ENGINE]=True;self.owner='acdc';self.heart_paused=False
        elif unit==entry.HUB:
            assert self.owner is None
            self.states['pld-web.service' if self.model=='skin' else 'pld-camus.service']=True
            self.owner=self.model;self.hub_paused=False;self.generation+=1
        self.states[unit]=True

    def stop(self, unit):
        self.events.append(('stop',unit));self.states[unit]=False
        if unit==entry.HUB:
            self.states['pld-web.service']=self.states['pld-camus.service']=False
            if self.owner in ('skin','camus'):self.owner=None
        if unit==entry.ENGINE and self.owner=='acdc':self.owner=None

    def health(self):
        return dict(ready=not self.heart_paused,backend='pl',active_job=None,
                    suspended=self.heart_paused,hardware_released=self.heart_paused,hdmi='off',
                    lifecycle='pause-resume-v1',resident_handoff='resident-hub-v1')

    def hub(self):
        return dict(phase='paused' if self.hub_paused else 'ready',model=self.model,
                    generation=self.generation,backend=dict(lifecycle='pause-resume-v1',active=None),
                    lifecycle='resident-hub-v1',hardware_released=self.hub_paused)

    def can_park_heart(self):return True

    def request(self, port, path, body=None):
        self.events.append(('request',port,path,body))
        if port==8092:
            if body['action']=='pause':
                assert self.owner=='acdc';self.owner=None;self.heart_paused=True
            else:
                assert self.owner is None;self.owner='acdc';self.heart_paused=False
        elif path=='/hub/lifecycle' and body['action']=='pause':
            assert self.owner in ('skin','camus');self.owner=None;self.hub_paused=True
        else:
            assert self.owner is None or self.owner in ('skin','camus')
            self.model=body['model'];self.owner=self.model;self.hub_paused=False;self.generation+=1
            self.states['pld-web.service' if self.model=='skin' else 'pld-camus.service']=True


class BootIntegration(unittest.TestCase):
    def test_cold_start_prepares_all_models_without_destroying_earlier_services(self):
        for default in entry.MODELS:
            with self.subTest(default=default):
                board=ColdResidentBoard();c=entry.Controller(board)
                c.settings['last_model']=default;c.display.restore=Mock();c.save_settings=Mock()
                c.start_boot_preparation();c.thread.join(3)
                self.assertFalse(c.thread.is_alive());self.assertEqual(c.boot['state'],'ready',c.error)
                self.assertEqual(set(c.boot['completed']),set(entry.MODELS))
                self.assertEqual(c.model,default);self.assertEqual(board.owner,default)
                self.assertEqual(board.events[0],('start',entry.WEB))
                self.assertTrue(all(board.running(u) for u in (entry.ENGINE,entry.WEB,entry.HUB,'pld-web.service','pld-camus.service')))
                self.assertFalse(any(event[0]=='stop' for event in board.events))
                self.assertEqual(c.settings['last_model'],default);c.save_settings.assert_not_called()
                c.display.restore.assert_called_once_with(default)
                self.assertGreaterEqual(c.snapshot()['boot']['elapsed_seconds'],0)

    def test_preparation_gates_tasks_restore_dispatch_and_proxy_during_ready_step(self):
        c=entry.Controller(Board());c.phase='ready';c.model='skin';c.boot['state']='running'
        c.board.hub=Mock(side_effect=AssertionError('must not contact display backend'))
        self.assertEqual(c.display.task(dict(action='begin',model='skin',job='a')), {})
        c.display.last_results['skin']=dict(model='skin',job='a'*32,view='outlined',identity='model')
        c.display.restore('skin')
        c.display.pending=dict(model='skin',job='a'*32,view='outlined')
        c.display.dispatch();c.board.hub.assert_not_called()
        for method in ('GET','POST'):
            handler=entry.Handler.__new__(entry.Handler)
            handler.server=SimpleNamespace(controller=c);handler.command=method
            handler.path='/skin/g/7/api/status';handler.send=Mock()
            handler.proxy();self.assertEqual(handler.send.call_args.args[0],409)

    def test_start_cancels_previous_display_and_cannot_be_repeated(self):
        c=entry.Controller(Board());c.display.pending=dict(model='skin',job='old')
        c._prepare_boot=Mock();c.start_boot_preparation();c.thread.join(1)
        self.assertIsNone(c.display.pending)
        with self.assertRaises(RuntimeError):c.start_boot_preparation()
        c._prepare_boot.assert_called_once_with('acdc')

    def test_shutdown_after_last_target_cannot_report_ready(self):
        c=entry.Controller(ColdResidentBoard());c.boot['state']='running'
        original=c.board.running
        def running(unit):
            if len(c.boot['completed'])==3:c.stopping=True
            return original(unit)
        c.board.running=running;c.display.restore=Mock();c._prepare_boot('camus')
        self.assertEqual(c.boot['state'],'error');c.display.restore.assert_not_called()
