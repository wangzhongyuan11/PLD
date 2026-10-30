"""Bounded software-only watchdog demonstration. No board/driver imports."""
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid


EXPECTED = '36c804019d7eaa56c34f980ea676d28c4826847eadb46964b3fc567e0cccace0'


class SimulatedDevice:
    def __init__(self, corrupt=False):
        self.mapped, self.ready, self.reset_confirmed = True, False, False
        self.corrupt = corrupt

    def poll(self):
        if not self.ready or not self.mapped:
            return None
        return b'corrupted simulated response' if self.corrupt else b'PLD simulated recovery verification v1'

    def reset_then_release(self):
        self.reset_confirmed = True
        self.mapped = False

    def reload_and_map(self):
        if not self.reset_confirmed or self.mapped:
            raise RuntimeError('模拟复位与缓冲区释放顺序错误')
        self.ready, self.mapped = True, True


class RecoveryDemo:
    def __init__(self, state_dir=None, *, timeout=.6, step_seconds=.3):
        if not 0 < timeout <= 2 or not 0 < step_seconds <= 1:
            raise ValueError('bounded demo timings required')
        self.path = Path(state_dir)/'recovery_demo.json' if state_dir else None
        self.timeout, self.step_seconds = timeout, step_seconds
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.thread = None
        self.state, self.stage, self.run_id = 'idle', '', None
        self.events = deque(maxlen=24)
        self.started = None
        self.error = ''

    def snapshot(self):
        with self.lock:
            return dict(simulated=True, state=self.state, stage=self.stage, run_id=self.run_id,
                        error=self.error, events=[dict(e) for e in self.events])

    def _record(self, code, message, stage, state='running'):
        with self.lock:
            self.stage, self.state = stage, state
            self.events.append(dict(code=code, message=message,
                host_utc=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=round(time.monotonic()-self.started, 4), simulated=True))
            if self.path:
                temporary = self.path.with_suffix('.tmp')
                temporary.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
                temporary.replace(self.path)
            print('RECOVERY_DEMO', code, message, flush=True)

    def start(self, scenario='recover'):
        if scenario not in ('recover', 'recovery_failure'):
            raise ValueError('未知模拟演示场景')
        with self.lock:
            if self.closed.is_set():
                raise RuntimeError('演示已关闭')
            if self.thread and self.thread.is_alive():
                raise RuntimeError('模拟恢复演示正在运行')
            self.events.clear()
            self.run_id = uuid.uuid4().hex
            self.started = time.monotonic()
            self.error = ''
            try:
                self._record('SIM_START', '软件模拟开始；不操作 FPGA、DMA、位流或真实任务', 'waiting')
            except OSError:
                self.state, self.stage, self.error = 'failed', 'failed', '模拟日志保存失败，演示未启动'
                raise
            self.thread = threading.Thread(target=self._run, args=(scenario,), daemon=True)
            self.thread.start()
            return self.snapshot()

    def _pause(self, seconds):
        if self.closed.wait(seconds):
            raise InterruptedError('模拟演示已取消')

    def _run(self, scenario):
        try:
            # Model an unresponsive device with a monotonic deadline. The
            # simulated request never returns; the watchdog owns the finite wait.
            device = SimulatedDevice(corrupt=scenario == 'recovery_failure')
            deadline = time.monotonic()+self.timeout
            while device.poll() is None and time.monotonic() < deadline:
                self._pause(min(.05, max(0, deadline-time.monotonic())))
            self._record('SIM_TIMEOUT', '模拟任务超过看门狗时限，进入一次恢复流程', 'timeout')
            self._pause(self.step_seconds)
            device.reset_then_release()
            self._record('SIM_RESET', '模拟 DMA 复位已确认，再释放模拟缓冲区', 'reset')
            self._pause(self.step_seconds)
            device.reload_and_map()
            self._record('SIM_RELOAD', '模拟重新加载位流与内存映射', 'reload')
            self._pause(self.step_seconds)
            # One simulated retry and an independent expected-output comparison.
            digest = hashlib.sha256(device.poll()).hexdigest()
            if digest != EXPECTED:
                raise RuntimeError('模拟输出校验失败')
            self._record('SIM_VERIFIED', '模拟重试一次，输出 SHA256 比较通过', 'verified')
            self._pause(self.step_seconds)
            self._record('SIM_OK', '模拟恢复完成；真实硬件恢复仍需独立板测', 'done', 'done')
        except InterruptedError:
            self._record('SIM_CANCELLED', '模拟演示已取消', 'cancelled', 'cancelled')
        except Exception as exc:
            with self.lock:
                self.error = '模拟演示失败：'+str(exc)
            try:
                self._record('SIM_FAILED', self.error, 'failed', 'failed')
            except OSError:
                # Storage failure cannot be converted into a successful demo.
                with self.lock:
                    self.state, self.stage = 'failed', 'failed'

    def close(self):
        self.closed.set()
        if self.thread:
            self.thread.join(timeout=5)
