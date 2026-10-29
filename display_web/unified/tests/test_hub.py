import http.client
from http.server import ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path,PurePosixPath
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch,Mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import hub
import install

class Board:
    def __init__(self):
        self.states={'skin':'inactive','camus':'inactive'};self.events=[]
        self.active=False;self.hdmi='off';self.fail_stop=False;self.fail_start=False
    def props(self,model):return {'ActiveState':self.states[model]}
    def status(self,model):return dict(mode='board',active=self.active,hdmi=self.hdmi,ready=True,backend='pl',inference_available=True,warmup='ready')
    def stop(self,model):
        self.events.append(('stop',model))
        if self.fail_stop:raise RuntimeError('stop failed')
        self.states[model]='inactive'
    def start(self,model):
        if any(s!='inactive' for s in self.states.values()):raise AssertionError('two live backends')
        self.events.append(('start',model))
        if self.fail_start:raise RuntimeError('init failed')
        self.states[model]='active'
    def request(self,model,method,path,body=None,kind=None):
        self.events.append(('request',model,method,path))
        if path=='/api/hdmi':self.hdmi='off';return 202,'application/json',b'{"hdmi":"off"}'
        return 200,'application/json',b'{"ok":true}'

class SwitchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.board=Board();self.c=hub.Controller(self.board,self.tmp.name)
    def tearDown(self):
        self.board.active=False
        if self.c.thread:self.c.thread.join(3)
        self.tmp.cleanup()
    def switch(self,model):
        self.c.switch(model);self.c.thread.join(3);self.assertFalse(self.c.thread.is_alive())
    def test_serial_switch_back_and_restore_selection(self):
        for model in ['skin','camus','skin']:
            self.switch(model);self.assertEqual(self.c.phase,'ready');self.assertEqual(self.c.model,model)
            self.assertEqual((Path(self.tmp.name)/'last_model').read_text(),model)
        self.assertEqual([e for e in self.board.events if e[0]=='start'],[('start','skin'),('start','camus'),('start','skin')])
    def test_busy_export_waits_and_duplicate_switch_rejected(self):
        self.switch('skin');self.board.active=True;self.c.switch('camus')
        time.sleep(.05);self.assertEqual(self.c.phase,'waiting')
        with self.assertRaises(RuntimeError):self.c.switch('skin')
        self.assertNotIn(('start','camus'),self.board.events)
        self.board.active=False;self.c.thread.join(3);self.assertEqual(self.c.model,'camus')
    def test_failed_release_never_starts_new_model(self):
        self.switch('skin');self.board.fail_stop=True;self.switch('camus')
        self.assertEqual(self.c.phase,'error');self.assertNotIn(('start','camus'),self.board.events)
        with self.assertRaises(RuntimeError):self.c.switch('camus')
    def test_failed_initialization_is_not_ready(self):
        self.board.fail_start=True;self.switch('camus');self.assertEqual(self.c.phase,'error')
        with self.assertRaises(RuntimeError):self.c.proxy('camus',self.c.generation,'POST','/api/jobs',b'{}','application/json')
    def test_hdmi_stopped_before_exit_accepts_camus_202(self):
        self.switch('camus');self.board.events=[];self.board.hdmi='showing';self.switch('skin')
        self.assertEqual(self.board.events[0],('request','camus','POST','/api/hdmi'))
        self.assertEqual(self.c.phase,'ready')
    def test_stale_generation_and_other_model_rejected(self):
        self.switch('skin');generation=self.c.generation;self.switch('camus');self.switch('skin')
        for model,epoch in [('skin',generation),('camus',self.c.generation)]:
            with self.assertRaises(RuntimeError):self.c.proxy(model,epoch,'POST','/api/jobs',b'{}','application/json')
    def test_unexpected_second_owner_blocks_requests(self):
        self.switch('skin');self.board.states['camus']='active'
        with self.assertRaises(RuntimeError):self.c.proxy('skin',self.c.generation,'POST','/api/hdmi',b'{}','application/json')

    def test_result_reads_do_not_query_other_hardware_owner(self):
        self.switch('skin')
        self.board.props=Mock(side_effect=AssertionError('service query for a result read'))
        self.board.status=Mock(side_effect=AssertionError('owner query for a result read'))
        self.c.proxy('skin',self.c.generation,'GET','/files/result/mask.png',None,'application/json')
        self.c.proxy('skin',self.c.generation,'POST','/api/jobs',b'{}','application/json')
        with self.assertRaises(RuntimeError):
            self.c.proxy('skin',self.c.generation-1,'GET','/api/status',None,'application/json')
    def test_close_releases_both(self):
        self.switch('skin');self.c.close();self.assertTrue(all(v=='inactive' for v in self.board.states.values()))
        with self.assertRaises(RuntimeError):self.c.switch('camus')

class FormatTests(unittest.TestCase):
    def test_rewrite_all_endpoints_and_preserve_data(self):
        text=b'''<a href="/"><script src="/app.js"></script>fetch('/api/jobs');const root='/files/'+id;const x="/style.css";'''
        out=hub.rewrite(text,'text/html','/skin/g/3/').decode()
        for p in ['app.js','api/jobs','files/','style.css']:self.assertIn('/skin/g/3/'+p,out)
        self.assertIn('href="/skin/g/3/"',out)
        self.assertEqual(hub.rewrite(b'\x00\xff','image/png','/skin/g/3/'),b'\x00\xff')
    def test_shell_command_read_without_execution(self):
        text="[Service]\nExecStart=/old\n[Service]\nExecStart=\nExecStart=/bin/bash -lc 'exec "+install.PYTHON+" -u server.py --runtime /runtime --data /data'\n"
        self.assertEqual(install.option(install.command_from_unit(text),'--runtime'),'/runtime')
        with self.assertRaises(ValueError):install.command_from_unit('[Service]\nExecStart=/bin/sh -c evil')
    def test_paths_and_units(self):
        for p in ['/tmp/a b','/tmp/../etc','/tmp/a%','/tmp/a;exit','/tmp/$(x)']:
            with self.assertRaises(ValueError):install.safe_path(p)
        backend={'source':'/web','extra':['--runtime','/runtime','--data','/data']}
        for model,signal in [('skin','SIGINT'),('camus','SIGTERM')]:
            unit=install.dropin(model,backend,PurePosixPath('/hub'))
            self.assertIn('SendSIGKILL=no',unit);self.assertIn('KillSignal='+signal,unit)
            self.assertIn('--host 127.0.0.1',unit);self.assertIn('Restart=no',unit)
    def test_failed_preflight_never_writes(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(install,'ROOT',Path(tmp)),patch.object(install,'RECEIPT',Path(tmp)/'receipt'),patch.object(install,'safe_path',side_effect=ValueError('unsupported')):
            with self.assertRaises(ValueError):install.install()
            self.assertEqual(list(Path(tmp).iterdir()),[])

class ResidentTests(unittest.TestCase):
    def test_bootstrap_adopts_verified_preloaded_target_without_reinitializing_it(self):
        board=Board();board.states['skin']='active'
        status=board.status;board.status=lambda model:dict(status(model),hardware_ready=model=='skin')
        with tempfile.TemporaryDirectory() as tmp:
            controller=hub.Controller(board,tmp);controller.switch('skin');controller.thread.join(3)
            self.assertEqual(controller.phase,'ready');self.assertTrue(controller.metrics['adopted_preloaded_target'])
            self.assertNotIn(('stop','skin'),board.events);self.assertNotIn(('start','skin'),board.events)

    def test_preloaded_process_without_http_status_is_observed_until_ready(self):
        board=Board();board.states['skin']='active';status=board.status;attempts=[]
        def preparing(model):
            attempts.append(model)
            if len(attempts)==1:raise ConnectionRefusedError()
            return dict(status(model),hardware_ready=model=='skin')
        board.status=preparing
        with tempfile.TemporaryDirectory() as tmp,patch.object(hub.time,'sleep'):
            controller=hub.Controller(board,tmp);controller.switch('skin');controller.thread.join(3)
            self.assertEqual(controller.phase,'ready');self.assertGreater(len(attempts),1)

    def test_pause_invalidates_requests_and_retains_services_until_resume(self):
        board=Board()
        board.park=lambda model:board.events.append(('park',model))
        board.check_external_owner=Mock()
        with tempfile.TemporaryDirectory() as tmp:
            controller=hub.Controller(board,tmp)
            controller.switch('skin');controller.thread.join(3)
            generation=controller.generation
            state=controller.lifecycle('pause')
            self.assertEqual(state['phase'],'paused');self.assertTrue(state['hardware_released'])
            self.assertEqual(board.states['skin'],'active')
            with self.assertRaises(RuntimeError):controller.switch('skin')
            with self.assertRaises(RuntimeError):controller.proxy('skin',generation,'POST','/api/jobs',b'{}','application/json')
            # Starting a paused application resumes its hardware, not its PID.
            board.start=lambda model:board.events.append(('resume',model))
            controller.lifecycle('resume','skin');controller.thread.join(3)
            self.assertEqual(controller.phase,'ready');self.assertFalse(controller.snapshot()['hardware_released'])
            self.assertEqual(board.check_external_owner.call_count,2)

    def test_external_owner_failure_prevents_backend_start(self):
        board=Board();board.check_external_owner=Mock(side_effect=RuntimeError('ACDC still owns hardware'))
        with tempfile.TemporaryDirectory() as tmp:
            controller=hub.Controller(board,tmp);controller.switch('camus');controller.thread.join(3)
            self.assertEqual(controller.phase,'error');self.assertEqual(board.events,[])

class HttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.c=hub.Controller(Board(),self.tmp.name)
        self.c.switch('skin');self.c.thread.join(3)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),hub.Handler)
        self.server.controller=self.c;self.server.token='test-access-code-0123456789';self.server.session='test-session'
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
    def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join();self.tmp.cleanup()
    def request(self,path,body=None,cookie=False,origin=True):
        host='127.0.0.1:'+str(self.server.server_port);c=http.client.HTTPConnection(host)
        headers={'Content-Type':'application/json'}
        if origin:headers['Origin']='http://'+host
        if cookie:headers['Cookie']='pld_hub=test-session'
        c.request('POST' if body is not None else 'GET',path,json.dumps(body) if body is not None else None,headers)
        r=c.getresponse();result=r.status,r.getheader('Set-Cookie'),r.read();c.close();return result
    def test_auth_and_cookie(self):
        self.assertEqual(self.request('/hub/status')[0],401)
        self.assertEqual(self.request('/skin/g/1/files/anything/result.zip')[0],401)
        self.assertEqual(self.request('/hub/login',{'token':'wrong'})[0],401)
        code,cookie,_=self.request('/hub/login',{'token':self.server.token});self.assertEqual(code,200);self.assertIn('HttpOnly',cookie)
        self.assertEqual(self.request('/hub/status',cookie=True)[0],200)
    def test_csrf_and_payload(self):
        self.assertEqual(self.request('/hub/switch',{'model':'camus'},True,False)[0],403)
        self.assertEqual(self.request('/hub/switch',{'model':'invalid'},True)[0],400)
        self.assertEqual(self.request('/hub/switch',[],True)[0],400)
    def test_proxy_stale_epoch_and_valid_route(self):
        self.assertEqual(self.request('/skin/g/0/api/jobs',{},True)[0],409)
        self.assertEqual(self.request('/camus/g/1/api/jobs',{},True)[0],409)
        self.assertEqual(self.request('/skin/g/1/api/status',cookie=True)[0],200)
    def test_preview_wait_query_reaches_backend(self):
        self.assertEqual(self.request('/skin/g/1/api/jobs/result?wait=preview',cookie=True)[0],200)
        self.assertIn(('request','skin','GET','/api/jobs/result?wait=preview'),self.c.board.events)
    def test_public_shell_has_switch_buttons(self):
        code,_,raw=self.request('/');self.assertEqual(code,200)
        self.assertIn(b'id="camus"',raw);self.assertIn(b'id="skin"',raw)

class ShutdownTests(unittest.TestCase):
    def test_systemd_stop_failures_prevent_reuse(self):
        board=hub.Board('token');before=dict(ControlGroup='')
        valid=dict(ActiveState='inactive',Result='success',MainPID='0')
        for values in [dict(ActiveState='failed'),dict(Result='timeout'),dict(MainPID='123')]:
            with patch.object(board,'props',side_effect=[before,{**valid,**values}]),patch.object(hub.subprocess,'run',return_value=types.SimpleNamespace(returncode=0)):
                with self.assertRaises(RuntimeError):board.stop('skin')
        with patch.object(board,'props',side_effect=[before,valid]),patch.object(hub.subprocess,'run',return_value=types.SimpleNamespace(returncode=0)):
            board.stop('skin')

    def test_skin_eof_before_launcher_exit(self):
        # fcntl is Linux-only; only the portable shutdown adapter is executed here.
        with patch.dict(sys.modules,{'fcntl':types.SimpleNamespace()}):
            spec=importlib.util.spec_from_file_location('launcher_test',ROOT/'backend_launcher.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        events=[]
        class Worker:
            def _stop(self,reason):events.append('original')
            def _shutdown_launcher(self):events.append('launcher_exit')
        process=Mock();process.poll.return_value=None;process.returncode=0
        process.stdin.close.side_effect=lambda:events.append('eof')
        process.wait.side_effect=lambda **kw:events.append('wait_exit')
        module.install_skin_shutdown(types.SimpleNamespace(PersistentBWorker=Worker))
        worker=Worker();worker.process=process;worker.reader=None
        worker._stop('closed_by_service')
        self.assertEqual(events,['eof','wait_exit','launcher_exit']);process.kill.assert_not_called();process.terminate.assert_not_called()
        process.returncode=1
        with self.assertRaises(RuntimeError):worker._stop('closed_by_service')

if __name__=='__main__':unittest.main(verbosity=2)
