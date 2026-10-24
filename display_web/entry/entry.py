"""Small LAN entry around the existing ACDC and dual-model applications."""
import argparse
import http.client
import json
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent
ENGINE = 'pld-acdc-v2-engine.service'
WEB = 'pld-acdc-v2-web.service'
HUB = 'pld-hub.service'
MODELS = ('acdc', 'skin', 'camus')


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
                raise RuntimeError(f'{path}: HTTP {response.status}')
            if response.getheader('Set-Cookie'):
                self.cookie = response.getheader('Set-Cookie')
            return json.loads(raw)
        finally:
            conn.close()

    def props(self, unit):
        result = subprocess.run(['systemctl', 'show', unit, '-p', 'ActiveState', '-p', 'MainPID',
                                 '-p', 'Result', '-p', 'ControlGroup'],
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
        self.request(8765, '/hub/login', {'token': self.token})
        return self.request(8765, '/hub/status')

    def health(self):
        return self.request(8092, '/api/v2/health')

    def prepare_dual(self, target):
        state = self.hub_root / 'state'
        state.mkdir(exist_ok=True)
        temp = state / 'last_model.entry.tmp'
        temp.write_text(target)
        temp.replace(state / 'last_model')


class Controller:
    def __init__(self, board):
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

    def snapshot(self):
        with self.lock:
            return dict(model=self.model, target=self.target, phase=self.phase,
                        error=self.error, url=self.url)

    def switch(self, target):
        if target not in MODELS:
            raise ValueError('未知模型')
        with self.lock:
            if self.stopping or self.proxy_lock.locked() or (self.thread and self.thread.is_alive()):
                raise RuntimeError('正在切换，请稍候')
            if self.phase == 'error':
                raise RuntimeError('切换失败，请检查服务后重启入口')
            if self.phase == 'ready' and self.model == target:
                return self.snapshot()
            self.target = target
            self.phase = 'switching'
            self.error = ''
            self.url = ''
            self.thread = threading.Thread(target=self._switch, args=(target,), daemon=True)
            self.thread.start()
            return self.snapshot()

    def wait(self, check, seconds=300):
        deadline = time.monotonic() + seconds
        while True:
            if check():
                return
            if time.monotonic() > deadline:
                raise RuntimeError('等待任务完成或模型就绪超时，保留现场')
            time.sleep(.5)

    def _switch(self, target):
        try:
            board = self.board
            if target == 'acdc':
                if board.running(HUB):
                    def dual_idle():
                        state = board.hub()
                        if state['phase'] == 'error':
                            raise RuntimeError(state.get('error') or '原入口报告故障')
                        return (state['phase'] == 'ready' and not state.get('evaluation_busy')
                                and not state.get('interactive_busy')
                                and not state.get('backend', {}).get('active'))
                    self.wait(dual_idle)
                    # Original hub waits for jobs, releases HDMI, and checks both backend cgroups.
                    board.stop(HUB)
                for unit in ('pld-web.service', 'pld-camus.service', 'pld-acdc.service'):
                    if board.running(unit):
                        raise RuntimeError(f'{unit} 仍在运行，禁止加载心脏模型')
                board.start(WEB)  # Its existing Wants= starts the engine and waits for READY.
                self.wait(lambda: self._heart_ready(), 200)
                url = '8092/'
            else:
                if board.running(ENGINE):
                    def heart_idle():
                        state = board.health()
                        if state.get('status') == 'fault':
                            raise RuntimeError('心脏模型报告硬件故障')
                        return state.get('ready') and not state.get('active_job')
                    self.wait(heart_idle)
                board.stop(WEB)
                board.stop(ENGINE)
                if not board.running(HUB):
                    board.prepare_dual(target)
                    board.start(HUB)
                def hub_started():
                    try:
                        return board.hub()['phase'] in ('ready', 'error')
                    except (OSError, http.client.HTTPException):
                        return False
                self.wait(hub_started, 200)
                state = board.hub()
                if state['phase'] == 'error':
                    raise RuntimeError(state.get('error') or '原入口初始化失败')
                if state['model'] != target:
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
                self.model = target
                self.url = url
                self.phase = 'ready'
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
        with controller.proxy_lock:
            with controller.lock:
                match = re.match(r'^/(skin|camus)/g/(\d+)/(.*)', self.path)
                if not match or controller.phase != 'ready' or match[1] != controller.model:
                    return self.send(409, {'error': '模型已切换，请等待新页面'})
            board = controller.board
            prefix = f'/{match[1]}/g/{match[2]}/'
            rest = match[3]
            interactive = (rest.split('?', 1)[0] in ('interactive', 'interactive.js', 'hub/status')
                           or rest.startswith('interactive/'))
            upstream = self.path
            if interactive:
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
                self.end_headers()
                while True:
                    chunk = response.read(128 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            finally:
                conn.close()

    def do_GET(self):
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
            if self.path != '/entry/switch':
                return self.proxy()
            if self.headers.get('Origin') != 'http://' + self.headers.get('Host', ''):
                return self.send(403, {'error': '不允许跨站切换'})
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 1024:
                raise ValueError('请求大小无效')
            self.send(202, self.server.controller.switch(json.loads(self.rfile.read(size))['model']))
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
    controller = Controller(Board(args.hub_root))
    server = ThreadingHTTPServer(('0.0.0.0', args.port), Handler)
    server.controller = controller
    def stop(*_):
        controller.stopping = True
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    controller.switch('acdc')
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if controller.thread:
            controller.thread.join()


if __name__ == '__main__':
    main()
