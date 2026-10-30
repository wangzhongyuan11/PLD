"""Small LAN entry around the existing ACDC and dual-model applications."""
import argparse
from contextlib import nullcontext
import http.client
import json
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit,parse_qs

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
from display_control import DisplayControl
from recovery_demo import RecoveryDemo
from demo_metrics import catalog_status
ENGINE = 'pld-acdc-v2-engine.service'
WEB = 'pld-acdc-v2-web.service'
HUB = 'pld-hub.service'
MODELS = ('acdc', 'skin', 'camus')


class UpstreamError(RuntimeError):
    def __init__(self,path,status):
        super().__init__(f'{path}: HTTP {status}')
        self.status=status


class Board:
    def __init__(self, hub_root):
        self.hub_root = Path(hub_root)
        config = json.loads((self.hub_root / 'config.json').read_text())
        self.token = Path(config['token_file']).read_text().strip()
        self.cookie = None

    def request(self, port, path, body=None):
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
        headers = {'Content-Type': 'application/json', 'Origin': f'http://127.0.0.1:{port}'}
        if self.cookie and port == 8765:
            headers['Cookie'] = self.cookie.split(';', 1)[0]
        try:
            conn.request('POST' if body is not None else 'GET', path,
                         json.dumps(body).encode() if body is not None else None, headers)
            response = conn.getresponse()
            raw = response.read()
            if response.status != 200 and response.status != 202:
                raise UpstreamError(path,response.status)
            if response.getheader('Set-Cookie'):
                self.cookie = response.getheader('Set-Cookie')
            return json.loads(raw)
        finally:
            conn.close()

    def props(self, unit):
        result = subprocess.run(['systemctl', 'show', unit, '-p', 'ActiveState', '-p', 'MainPID',
                                 '-p', 'Result', '-p', 'ControlGroup', '-p', 'Conflicts'],
                                check=True, capture_output=True, text=True, timeout=10)
        return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)

    def running(self, unit):
        state = self.props(unit)['ActiveState']
        if state not in ('active', 'inactive'):
            raise RuntimeError(f'{unit} 状态为 {state}，请检查服务日志')
        return state == 'active'

    def stop(self, unit):
        group = self.props(unit).get('ControlGroup', '')
        subprocess.run(['systemctl', 'stop', unit], check=True, capture_output=True, timeout=1200)
        state = self.props(unit)
        if state['ActiveState'] != 'inactive' or state['MainPID'] != '0' or state.get('Result') != 'success':
            raise RuntimeError(f'{unit} 未正常退出，禁止加载另一模型')
        if group:
            if not group.startswith('/') or '..' in group.split('/'):
                raise RuntimeError('无效 cgroup')
            for base, name in (('/sys/fs/cgroup', 'cgroup.procs'), ('/sys/fs/cgroup/systemd', 'tasks')):
                path = Path(base + group) / name
                if path.exists() and path.read_text().strip():
                    raise RuntimeError(f'{unit} 仍有子进程')

    def start(self, unit):
        subprocess.run(['systemctl', 'start', unit], check=True, capture_output=True, timeout=200)

    def hub(self):
        # The existing token stays on the board. Only the original HttpOnly session cookie is used.
        if not self.cookie:self.request(8765, '/hub/login', {'token': self.token})
        try:state=self.request(8765, '/hub/status')
        except UpstreamError as exc:
            if exc.status!=401:raise
            self.request(8765, '/hub/login', {'token': self.token})
            state=self.request(8765, '/hub/status')
        if state.get('phase')=='ready':
            state['backend']=self.request(8765,f"/{state['model']}/g/{state['generation']}/api/status")
        return state

    def health(self):
        return self.request(8092, '/api/v2/health')

    def can_park_heart(self):
        conflicts=set(self.props(ENGINE).get('Conflicts','').split())
        return not conflicts.intersection((HUB,'pld-web.service','pld-camus.service'))

    def can_start_resident_heart(self):
        # A stopped engine has no health endpoint. Check its effective service
        # configuration before preserving the already capable, paused hub.
        if not self.can_park_heart():return False
        result=subprocess.run(['systemctl','show',ENGINE,'-p','Environment','--value'],
                              check=True,capture_output=True,text=True,timeout=10)
        return 'PLD_RESIDENT_MODELS=1' in shlex.split(result.stdout)

    def prestart_dual(self,model):
        units={'skin':'pld-web.service','camus':'pld-camus.service'}
        states={key:self.props(unit) for key,unit in units.items()}
        if any(state.get('ActiveState')!='inactive' for state in states.values()):return
        if any(set(state.get('Conflicts','').split()).intersection((ENGINE,WEB,HUB,*units.values())) for state in states.values()):return
        # The caller has confirmed ACDC hardware release. Overlap the target's
        # normal startup with the hub's CPU-only imports; no extra owner exists.
        self.start(units[model])

    def prepare_dual(self, target):
        state = self.hub_root / 'state'
        state.mkdir(exist_ok=True)
        temp = state / 'last_model.entry.tmp'
        temp.write_text(target)
        temp.replace(state / 'last_model')


class Controller:
    def __init__(self, board, state_dir=None):
        self.board = board
        self.lock = threading.RLock()
        self.proxy_lock = threading.Lock()
        self.model = None
        self.target = 'acdc'
        self.phase = 'idle'
        self.error = ''
        self.url = ''
        self.thread = None
        self.stopping = False
        self.state_dir = Path(state_dir) if state_dir else None
        self.settings = dict(auto_hdmi=True, auto_segment=True, last_model='acdc')
        if self.state_dir:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            path = self.state_dir / 'preferences.json'
            if path.exists():
                saved = json.loads(path.read_text())
                for key in ('auto_hdmi', 'auto_segment'):
                    if type(saved.get(key)) is bool: self.settings[key] = saved[key]
                if saved.get('last_model') in MODELS: self.settings['last_model'] = saved['last_model']
        self.stage = ''
        self.metrics = {}
        self.switch_started = None
        self.boot = dict(state='idle', completed=[], target=None, default_model=None, elapsed_seconds=None)
        self.boot_started = None
        self.display = DisplayControl(self)
        self.recovery_demo = RecoveryDemo(self.state_dir)

    def save_settings(self):
        if self.state_dir:
            path = self.state_dir / 'preferences.json'
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(self.settings))
            temp.replace(path)

    def set_stage(self, stage):
        now = time.monotonic()
        with self.lock:
            if self.stage and self.switch_started is not None:
                self.metrics[self.stage] = now - self.stage_started
            self.stage = stage
            self.stage_started = now

    def snapshot(self):
        with self.lock:
            boot = dict(self.boot, completed=list(self.boot['completed']))
            if boot['state'] == 'running' and self.boot_started is not None:
                boot['elapsed_seconds'] = time.monotonic()-self.boot_started
            return dict(boot=boot, model=self.model, target=self.target,
                        phase='preparing' if self.boot['state']=='running' else self.phase,
                        error=self.error, url=self.url, stage=self.stage,
                        settings=dict(self.settings), display=self.display.snapshot(),
                        recovery_demo=self.recovery_demo.snapshot(), stages_seconds=dict(self.metrics),
                        elapsed_seconds=(time.monotonic()-self.switch_started
                                         if self.switch_started is not None else None))

    def switch(self, target):
        if target not in MODELS:
            raise ValueError('未知模型')
        with self.lock:
            if self.boot['state']=='running' or self.stopping or self.proxy_lock.locked() or (self.thread and self.thread.is_alive()):
                raise RuntimeError('正在切换，请稍候')
            if self.phase == 'error':
                raise RuntimeError('切换失败，请检查服务后重启入口')
            if self.phase == 'ready' and self.model == target:
                return self.snapshot()
            self.target = target
            self.display.cancel()
            self.phase = 'switching'
            self.error = ''
            self.url = ''
            self.thread = threading.Thread(target=self._switch, args=(target,), daemon=True)
            self.thread.start()
            return self.snapshot()

    def start_recovery_demo(self, body):
        # This demonstration has no hardware references and never marks a real
        # job recovered. Existing fault gates and unique ownership stay intact.
        with self.lock:
            if self.stopping or self.boot['state'] == 'running' or self.phase != 'ready':
                raise RuntimeError('请等待模型准备完成后运行模拟演示')
            if set(body) - {'scenario'}:
                raise ValueError('模拟演示只接受场景参数')
            self.recovery_demo.start(body.get('scenario', 'recover'))
            return self.snapshot()

    def start_boot_preparation(self):
        """Prepare services sequentially through the existing hardware lifecycle."""
        with self.lock:
            if (self.stopping or self.boot['state'] != 'idle' or self.phase != 'idle'
                    or self.proxy_lock.locked() or (self.thread and self.thread.is_alive())):
                raise RuntimeError('启动准备已经执行或当前入口不允许启动')
            default = self.settings['last_model']
            self.boot.update(state='running', default_model=default, target='acdc')
            self.boot_started = time.monotonic()
            self.display.cancel()
            self.thread = threading.Thread(target=self._prepare_boot, args=(default,), daemon=True)
            self.thread.start()

    def _prepare_boot(self, default):
        try:
            # A cold board has no ACDC engine. Establish its resident handoff
            # first: otherwise switching dual -> ACDC uses the old stop path
            # and discards the dual services that were just prepared.
            order = ['acdc'] + [m for m in ('skin', 'camus') if m != default]
            if order[-1] != default: order.append(default)
            for target in order:
                with self.lock:
                    if self.stopping: raise RuntimeError('启动准备已停止')
                    self.boot['target'] = self.target = target
                    self.phase = 'switching'
                self._switch(target)
                if self.phase != 'ready':
                    raise RuntimeError(self.error or '模型准备失败')
                if target == 'acdc':
                    state = self.board.health()
                    if (state.get('lifecycle') != 'pause-resume-v1' or
                            state.get('resident_handoff') != 'resident-hub-v1' or
                            not self.board.can_park_heart()):
                        raise RuntimeError('ACDC 缺少常驻暂停支持，请先核对部署版本')
                else:
                    state = self.board.hub()
                    if (state.get('lifecycle') != 'resident-hub-v1' or
                            state.get('backend', {}).get('lifecycle') != 'pause-resume-v1'):
                        raise RuntimeError('双模型缺少常驻暂停支持，请先核对部署版本')
                with self.lock:
                    if target not in self.boot['completed']: self.boot['completed'].append(target)
            for unit in (ENGINE, WEB, HUB, 'pld-web.service', 'pld-camus.service'):
                if not self.board.running(unit):
                    raise RuntimeError(unit + ' 未保留常驻进程，不能宣布全部准备完成')
            with self.lock:
                if self.stopping: raise RuntimeError('启动准备已停止')
                self.boot['state'] = 'ready'
                self.phase = 'ready'
            self.display.restore(default)
        except Exception as exc:
            with self.lock:
                self.boot['state'] = 'error'
                self.phase = 'error'
                self.error = '开机准备未完成：' + str(exc)
            print('BOOT_PREPARATION_ERROR', str(exc), flush=True)
        finally:
            with self.lock:
                if self.boot_started is not None:
                    self.boot['elapsed_seconds'] = time.monotonic()-self.boot_started

    def wait(self, check, seconds=300):
        deadline = time.monotonic() + seconds
        while True:
            if check():
                return
            if time.monotonic() > deadline:
                raise RuntimeError('等待任务完成或模型就绪超时，保留现场')
            time.sleep(.2)

    def _switch(self, target):
        self.switch_started = time.monotonic()
        self.stage = ''
        self.metrics = {}
        self.set_stage('wait_current_task')
        try:
            board = self.board
            if target == 'acdc':
                resident=False
                if board.running(HUB):
                    def dual_idle():
                        state = board.hub()
                        if state['phase'] == 'error':
                            raise RuntimeError(state.get('error') or '原入口报告故障')
                        return (state['phase'] in ('ready','paused') and not state.get('evaluation_busy') and not state.get('proxy_busy')
                                and not state.get('interactive_busy')
                                and not state.get('backend', {}).get('active'))
                    self.wait(dual_idle)
                    self.set_stage('release_previous_model')
                    state=board.hub()
                    heart_running=board.running(ENGINE)
                    heart=board.health() if heart_running else {}
                    capable=(heart.get('resident_handoff')=='resident-hub-v1' if heart_running else
                             hasattr(board,'can_start_resident_heart') and board.can_start_resident_heart())
                    resident=state.get('lifecycle')=='resident-hub-v1' and capable
                    if resident:
                        board.request(8765,'/hub/lifecycle',{'action':'pause'})
                        state=board.hub()
                        if state.get('phase')!='paused' or not state.get('hardware_released'):
                            raise RuntimeError('双模型硬件释放未确认')
                    else:board.stop(HUB)
                for unit in ('pld-web.service', 'pld-camus.service', 'pld-acdc.service'):
                    if board.running(unit) and (unit=='pld-acdc.service' or not resident):
                        raise RuntimeError(f'{unit} 仍在运行，禁止加载心脏模型')
                self.set_stage('start_target')
                board.start(WEB)  # Its existing Wants= starts the engine and waits for READY.
                self.set_stage('initialize_target')
                try:state=board.health()
                except (OSError,http.client.HTTPException):state={}
                if state.get('suspended'):
                    board.request(8092,'/api/v2/lifecycle',{'action':'resume'})
                self.wait(lambda: self._heart_ready(), 200)
                url = '8092/'
            else:
                if board.running(ENGINE):
                    def heart_idle():
                        state = board.health()
                        if state.get('status') == 'fault':
                            raise RuntimeError('心脏模型报告硬件故障')
                        return (state.get('ready') or state.get('suspended')) and not state.get('active_job')
                    self.wait(heart_idle)
                self.set_stage('release_previous_model')
                state=board.health() if board.running(ENGINE) else {}
                if state.get('lifecycle')=='pause-resume-v1' and board.can_park_heart():
                    if not state.get('suspended'):
                        board.request(8092,'/api/v2/lifecycle',{'action':'pause'})
                    def paused():
                        state=board.health()
                        if state.get('hdmi')=='error':raise RuntimeError(state.get('hdmi_error'))
                        return state.get('suspended') and state.get('hardware_released') and state.get('hdmi')=='off'
                    self.wait(paused,90)
                else:
                    board.stop(WEB)
                    board.stop(ENGINE)
                if not board.running(HUB):
                    self.set_stage('start_target')
                    board.prepare_dual(target)
                    if hasattr(board,'prestart_dual'):board.prestart_dual(target)
                    board.start(HUB)
                def hub_started():
                    try:
                        return board.hub()['phase'] in ('ready', 'paused', 'error')
                    except (OSError, http.client.HTTPException):
                        return False
                self.set_stage('initialize_target')
                self.wait(hub_started, 200)
                state = board.hub()
                if state['phase'] == 'error':
                    raise RuntimeError(state.get('error') or '原入口初始化失败')
                if state['phase']=='paused':
                    board.request(8765,'/hub/lifecycle',{'action':'resume','model':target})
                elif state['model'] != target:
                    board.request(8765, '/hub/switch', {'model': target})
                def selected():
                    state = board.hub()
                    if state['phase'] == 'error':
                        raise RuntimeError(state.get('error') or '原入口切换失败')
                    if state['phase'] == 'ready' and state['model'] == target:
                        self.url = f"/{target}/g/{state['generation']}/"
                        return True
                    return False
                self.wait(selected)
                url = self.url
            with self.lock:
                self.set_stage('ready')
                self.model = target
                self.url = url
                self.phase = 'ready'
                if self.boot['state'] != 'running':
                    self.settings['last_model'] = target
                    self.save_settings()
                self.metrics['total'] = time.monotonic()-self.switch_started
                print('ENTRY_SWITCH_TIMING', json.dumps(dict(model=target, stages=self.metrics)), flush=True)
                self.switch_started = None
            if self.boot['state'] != 'running': self.display.restore(target)
        except Exception as exc:
            with self.lock:
                self.phase = 'error'
                self.error = str(exc)
            print('ENTRY_SWITCH_ERROR', str(exc), flush=True)

    def _heart_ready(self):
        try:
            state = self.board.health()
        except (OSError, http.client.HTTPException):
            return False
        if state.get('status') == 'fault':
            raise RuntimeError('心脏模型初始化失败')
        return state.get('ready') and state.get('backend') == 'pl'


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def send(self, code, data, kind='application/json; charset=utf-8'):
        body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def proxy(self):
        controller = self.server.controller
        # Keep the old hub and its SAMEORIGIN iframe on this entry's origin.
        # Its own generation check continues to reject stale pages.
        # Only mutations participate in hardware/lifecycle serialization. A
        # slow display request must not queue status, previews or downloads.
        started = time.monotonic()
        with controller.proxy_lock if self.command == 'POST' else nullcontext():
            lock_wait = time.monotonic() - started
            if self.command == 'POST':
                print('ENTRY_PROXY_WAIT', round(lock_wait, 6), self.path, flush=True)
            with controller.lock:
                match = re.match(r'^/(skin|camus)/g/(\d+)/(.*)', self.path)
                if controller.boot['state']=='running' or not match or controller.phase != 'ready' or match[1] != controller.model:
                    return self.send(409, {'error': '模型已切换，请等待新页面'})
            board = controller.board
            prefix = f'/{match[1]}/g/{match[2]}/'
            rest = match[3]
            interactive = (rest.split('?', 1)[0] in ('interactive', 'interactive.js', 'hub/status')
                           or rest.startswith('interactive/'))
            upstream = self.path
            if interactive:
                if match[1] != 'skin':
                    return self.send(409, {'error': 'ROI 交互模式仅支持皮肤镜'})
                state = board.hub()
                if (state['phase'] != 'ready' or state['model'] != match[1]
                        or state['generation'] != int(match[2])):
                    return self.send(409, {'error': '模型已切换，请重新打开交互分割'})
                upstream = '/' + rest
            body = None
            if self.command == 'POST':
                if self.headers.get('Origin') != 'http://' + self.headers.get('Host', ''):
                    return self.send(403, {'error': '不允许跨站操作'})
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 32 * 1024 * 1024:
                    return self.send(413, {'error': '上传大小无效'})
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError('上传未完成')
            conn = http.client.HTTPConnection('127.0.0.1', 8765, timeout=60)
            try:
                conn.request(self.command, upstream, body, {
                    'Content-Type': self.headers.get('Content-Type', 'application/json'),
                    'Origin': 'http://127.0.0.1:8765',
                    'Cookie': board.cookie.split(';', 1)[0],
                })
                response = conn.getresponse()
                kind = response.getheader('Content-Type', 'application/octet-stream')
                if interactive and any(t in kind for t in ('text/html', 'javascript')):
                    raw = response.read().decode('utf-8')
                    raw = re.sub(r'''(["'])/(interactive|hub/status)''',
                                 lambda m: m[1] + prefix + m[2], raw)
                    if 'text/html' in kind:
                        raw = raw.replace('href="/"', 'href="/" target="_top"')
                    return self.send(response.status, raw.encode('utf-8'), kind)
                self.send_response(response.status)
                for key in ('Content-Type', 'Content-Length', 'Content-Disposition'):
                    if response.getheader(key):
                        self.send_header(key, response.getheader(key))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Frame-Options', 'SAMEORIGIN')
                self.send_header('X-PLD-Proxy-Wait-Seconds', str(round(lock_wait, 6)))
                self.end_headers()
                while True:
                    chunk = response.read(128 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            finally:
                conn.close()

    def do_GET(self):
        parsed=urlsplit(self.path)
        if parsed.path == '/entry/metrics/view':
            return self.send(200,(ROOT/'metrics.html').read_bytes(),'text/html; charset=utf-8')
        if parsed.path == '/entry/metrics':
            model=parse_qs(parsed.query).get('model',[None])[0]
            if model is not None and model not in MODELS:
                return self.send(400,dict(error='未知模型'))
            status=catalog_status()
            if model is not None:status['records']=[r for r in status['records'] if r['model']==model]
            return self.send(200,status)
        if self.path == '/':
            return self.send(200, (ROOT / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        if self.path == '/entry/status':
            return self.send(200, self.server.controller.snapshot())
        try:
            self.proxy()
        except (OSError, ValueError, http.client.HTTPException) as exc:
            self.send(502, {'error': str(exc)})

    def do_POST(self):
        try:
            if self.path not in ('/entry/switch', '/entry/task', '/entry/settings', '/entry/recovery-demo'):
                return self.proxy()
            if self.headers.get('Origin') != 'http://' + self.headers.get('Host', ''):
                return self.send(403, {'error': '不允许跨站切换'})
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 2048:
                raise ValueError('请求大小无效')
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError('请求必须为 JSON 对象')
            if self.path == '/entry/recovery-demo':
                return self.send(202, self.server.controller.start_recovery_demo(body))
            if self.path == '/entry/task':
                if self.client_address[0] != '127.0.0.1':
                    return self.send(403, {'error': '任务通知仅供板内使用'})
                return self.send(200, self.server.controller.display.task(body))
            if self.path == '/entry/settings':
                result=self.server.controller.display.settings(body)
                return self.send(200,result)
            self.send(202, self.server.controller.switch(body['model']))
        except (ValueError, KeyError, TypeError) as exc:
            self.send(400, {'error': str(exc)})
        except RuntimeError as exc:
            self.send(409, {'error': str(exc)})
        except (OSError, http.client.HTTPException) as exc:
            self.send(502, {'error': str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hub-root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8088)
    args = parser.parse_args()
    controller = Controller(Board(args.hub_root), args.hub_root / 'state' / 'entry')
    server = ThreadingHTTPServer(('0.0.0.0', args.port), Handler)
    server.controller = controller
    controller.display.start()
    def stop(*_):
        controller.stopping = True
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    controller.start_boot_preparation()
    try:
        server.serve_forever()
    finally:
        controller.recovery_demo.close()
        controller.display.close()
        server.server_close()
        if controller.thread:
            controller.thread.join()


if __name__ == '__main__':
    main()
