"""PLD local web console. Standard-library HTTP + existing NumPy/Pillow pipeline.

Only one processing/HDMI owner; board inference delegates to the frozen B runtime.
No network access to a board is implied by running this application on Windows.
"""
import argparse
import base64
import hashlib
import io
import json
import mimetypes
import os
import platform
import secrets
import subprocess
import sys
import threading
import time
import uuid
import warnings
import zipfile
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, ImageOps
from contour import process
from pld_hdmi import binary_mask, fit_rgb
from pld_result import load_result
from infer_then_display import inference_command, verify_runtime

ROOT = Path(__file__).resolve().parent
MAX_BODY = 24 * 1024 * 1024
MAX_FILE = 8 * 1024 * 1024
MAX_PIXELS = 12_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
VIEWS = ("original", "mask", "contour", "outlined", "cutout", "comparison")


@lru_cache(maxsize=1)
def current_boot_id():
    """Stable across service restarts, different after an OS reboot."""
    if platform.system() == 'Linux':
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if platform.system() == 'Windows':
        import ctypes
        # SystemBootEnvironmentInformation starts with the boot GUID.
        # Unlike WMI this read-only query works without elevated permissions.
        query = ctypes.WinDLL('ntdll').NtQuerySystemInformation
        query.argtypes = [ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong,
                          ctypes.POINTER(ctypes.c_ulong)]
        query.restype = ctypes.c_long
        buffer = ctypes.create_string_buffer(32)
        length = ctypes.c_ulong()
        status = query(90, buffer, len(buffer), ctypes.byref(length))
        if status != 0 or length.value < 16 or not any(buffer.raw[:16]):
            raise RuntimeError('无法读取电脑开机标识，请检查系统环境')
        return 'windows-' + str(uuid.UUID(bytes_le=buffer.raw[:16]))
    raise RuntimeError('本次开机历史目前支持 Windows 和 Linux')


def persistent_token(data):
    """Reuse this installation's access code; never silently rotate on errors."""
    path = Path(data) / ".access_token"
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "w", encoding="ascii") as f:
            f.write(secrets.token_urlsafe(24))
    value = path.read_text(encoding="ascii").strip()
    if len(value) != 32 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
        raise ValueError("data/.access_token 文件损坏，请先备份并检查；未自动更换访问码")
    return value


def decode_upload(value, is_mask=False):
    if not isinstance(value, str):
        raise ValueError("请选择图片文件")
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError("上传文件编码无效") from exc
    if not raw or len(raw) > MAX_FILE:
        raise ValueError("单个文件须在 8 MB 以内")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as im:
            if im.format not in ("JPEG", "PNG") or im.width * im.height > MAX_PIXELS:
                raise ValueError("仅支持 JPG/PNG，最多 1200 万像素")
            if is_mask:
                arr = np.asarray(im).copy()
                binary_mask(arr)
            else:
                arr = np.asarray(ImageOps.exif_transpose(im).convert("RGB")).copy()
    return raw, arr


class Console:
    def __init__(self, data, runtime=None, token=None, portable=False, boot_id=None):
        self.data = Path(data).resolve(); self.data.mkdir(parents=True, exist_ok=True)
        self.runtime = Path(runtime).resolve() if runtime else None
        self.token = token if token is not None else persistent_token(self.data)
        self.portable = portable
        self.board = platform.system() == "Linux" and platform.machine().lower().startswith("arm")
        self.lock = threading.RLock()
        self.jobs = {}
        self.boot_id = boot_id or current_boot_id()
        self.restore_history()
        self.active = None
        self.hdmi = None
        self.hdmi_job = None
        self.hdmi_view = None
        self.fault = None
        self.runtime_ok = False
        self.runtime_error = None
        if self.board and self.runtime:
            try:
                verify_runtime(self.runtime)
                inference_command(self.runtime, Path("check.jpg"), Path("check_out"), self.portable)
                self.runtime_ok = True
            except Exception as exc:
                self.runtime_error = str(exc)

    def restore_history(self):
        for folder in self.data.iterdir():
            if not folder.is_dir() or folder.is_symlink() or len(folder.name) != 32 or any(c not in '0123456789abcdef' for c in folder.name):
                continue
            try:
                saved = json.loads((folder/'job.json').read_text(encoding='utf-8'))
                if saved.get('boot_id') != self.boot_id: continue
                j = saved['job']
                if j['id'] != folder.name or j['state'] != 'done': continue
                if j['mode'] not in ('mask','infer','demo') or not isinstance(j['name'],str): continue
                if not isinstance(j['created'],(int,float)) or not isinstance(j['statistics'],dict): continue
                if not j['views'] or any(v not in VIEWS for v in j['views']): continue
                if not all((folder/(v+'.png')).is_file() for v in j['views']): continue
                if not (folder/'results.zip').is_file(): continue
                self.jobs[j['id']] = j
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def history(self):
        with self.lock:
            entries = [dict(j) for j in self.jobs.values() if j.get('state') == 'done']
            entries.sort(key=lambda j: (j['created'],j['id']), reverse=True)
            return {'boot_id': self.boot_id, 'items': entries}

    def status(self):
        with self.lock:
            running = self.hdmi is not None and self.hdmi.poll() is None
            if self.hdmi is not None and not running and self.hdmi.returncode != 0:
                self.fault = "HDMI 子进程异常退出，请检查板端日志并重启服务确认硬件状态"
            ready = running and (self.data / self.hdmi_job / "hdmi.ready").exists()
            return {"mode": "board" if self.board else "local", "inference_available": self.runtime_ok and not self.fault,
                    "hdmi_available": self.board and not self.fault, "active": self.active,
                    "hdmi": "showing" if ready else "starting" if running else "off",
                    "hdmi_view": self.hdmi_view if running else None,
                    "fault": self.fault, "runtime_error": self.runtime_error}

    def job(self, ident):
        with self.lock:
            if ident not in self.jobs: raise KeyError("任务不存在")
            return json.loads(json.dumps(self.jobs[ident]))

    def update(self, ident, **values):
        with self.lock: self.jobs[ident].update(values)

    def submit(self, body):
        mode = body.get("mode")
        if mode not in ("demo", "mask", "infer"): raise ValueError("未知处理模式")
        width = body.get("thickness", 2)
        if type(width) is not int or not 1 <= width <= 8: raise ValueError("线宽须为 1–8 的整数")
        with self.lock:
            if self.active: raise RuntimeError("已有任务处理中，请等待完成")
            if self.fault: raise RuntimeError(self.fault)
            if mode == "infer" and not self.runtime_ok:
                raise ValueError("当前未连接可用的板端推理环境；可使用真实结果回放或上传配套掩膜")
            rgb = raw = mask = None
            if mode != "demo":
                raw, rgb = decode_upload(body.get("image"))
                if self.board and rgb.shape[0] * rgb.shape[1] > 3_000_000:
                    raise ValueError("板端网页版本限制为 300 万像素以内，请先缩小原图（掩膜模式需同步缩小掩膜）")
                if mode == "infer" and not raw.startswith(b"\xff\xd8"):
                    raise ValueError("当前 B 推理接口要求原始 JPG，请上传 JPG 图片")
                if mode == "mask":
                    _, mask = decode_upload(body.get("mask"), True)
                    binary_mask(mask, rgb.shape[:2])
            ident = uuid.uuid4().hex
            self.jobs[ident] = {"id": ident, "state": "running", "stage": "准备数据", "progress": 10,
                                "mode": mode, "name": str(body.get("name", "ISIC_0000003 · 真实掩膜回放"))[:120],
                                "thickness": width, "created": time.time()}
            self.active = ident
            threading.Thread(target=self._run, args=(ident, mode, raw, rgb, mask, width), daemon=True).start()
            return self.job(ident)

    def stop_hdmi(self):
        with self.lock:
            p = self.hdmi
            if p is not None and p.poll() is None:
                p.terminate()
                try: p.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    self.fault = "HDMI 未能正常释放，请在板端检查并重启服务；已禁止新硬件操作"
                    raise RuntimeError(self.fault)
                if p.returncode != 0:
                    self.fault = "HDMI 关闭失败，请检查板端日志；已禁止新硬件操作"
                    raise RuntimeError(self.fault)
            elif p is not None and p.returncode != 0:
                self.fault = "HDMI 异常退出，请检查板端日志后重启服务"
                raise RuntimeError(self.fault)
            self.hdmi = None; self.hdmi_job = None; self.hdmi_view = None

    def start_hdmi(self, ident, view):
        with self.lock:
            self.status()
            if not self.board: raise ValueError("HDMI 需要在 PYNQ 板端运行本服务")
            if self.fault: raise RuntimeError(self.fault)
            if self.active: raise RuntimeError("请等待处理完成后再显示")
            j = self.job(ident)
            if j["state"] != "done" or view not in j["views"]: raise ValueError("该显示画面尚未生成")
            self.stop_hdmi()
            folder = self.data / ident
            (folder / "hdmi.ready").unlink(missing_ok=True)
            with (folder / "hdmi.log").open("w", encoding="utf-8") as log:
                self.hdmi = subprocess.Popen([sys.executable, str(ROOT / "hdmi_worker.py"), str(folder), view], stdout=log, stderr=log)
            self.hdmi_job, self.hdmi_view = ident, view
            return self.status()

    def _run(self, ident, mode, raw, rgb, mask, width):
        folder = self.data / ident; folder.mkdir()
        start = time.perf_counter()
        try:
            b = None
            if mode == "demo":
                b = load_result(ROOT / "samples/frame_000001.json")
                mask = b.original_mask()
            elif mode == "infer":
                self.stop_hdmi()
                self.update(ident, stage="FPGA 分割中", progress=30)
                source = folder / "input.jpg"; source.write_bytes(raw)
                verify_runtime(self.runtime)
                command = inference_command(self.runtime, source, folder / "inference", self.portable)
                with (folder / "inference.log").open("w", encoding="utf-8") as log:
                    p = subprocess.Popen(command, cwd=str(self.runtime), stdout=log, stderr=log)
                    try: code = p.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        p.kill(); p.wait()
                        self.fault = "推理超时，需在板端确认 DMA 状态并重启服务；已禁止新硬件操作"
                        raise RuntimeError(self.fault)
                if code != 0:
                    self.fault = "板端推理失败，请检查 inference.log 并确认硬件状态后重启服务"
                    raise RuntimeError(self.fault)
                b = load_result(folder / "inference/frame_000001.json", source)
                mask, rgb = b.original_mask(), b.rgb
            self.update(ident, stage="提取轮廓与统计面积", progress=75)
            result = process(mask, rgb, thickness=width)
            views = ["mask", "contour", "outlined", "cutout"] if rgb is not None else ["mask", "contour", "outlined"]
            arrays = {k: result[k] for k in views}
            if rgb is not None:
                arrays["original"] = rgb; views.insert(0, "original")
            left = rgb if rgb is not None else np.repeat((mask.astype(np.uint8)*255)[:, :, None], 3, axis=2)
            arrays["comparison"] = np.concatenate((fit_rgb(left, 640, 480), fit_rgb(result["outlined"], 640, 480)), axis=1)
            views.append("comparison")
            for key, a in arrays.items():
                Image.fromarray(a.astype(np.uint8)*255 if a.dtype == bool else a).save(folder / (key+".png"))
            stats = result["statistics"]
            if b is not None:
                stats["area_percent"] = b.stats["area_percent"]
                stats["foreground_pixels"] = b.stats["mask_pixels"]
                stats["image_pixels"] = b.stats["valid_pixels"]
                stats["area_scope"] = "模型有效区域（排除补边）"
                stats["b_seconds"] = b.metadata.get("online_b_seconds")
                stats["historical_b_time"] = mode == "demo"
            else:
                stats["area_scope"] = "上传掩膜的原图像素网格"
            stats["processing_seconds"] = time.perf_counter()-start
            stats["processing_scope"] = "本次读取/处理/保存，不含上传、浏览器和 HDMI；回放不含推理"
            stats["data_source"] = "历史板测掩膜，无原始 JPG；未重新推理" if mode == "demo" else "用户图片及掩膜；未执行模型推理" if mode == "mask" else "本次板端 FPGA 推理"
            (folder / "report.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
            with zipfile.ZipFile(folder / "results.zip", "w", zipfile.ZIP_DEFLATED) as z:
                for name in views: z.write(folder / (name+".png"), name+".png")
                z.write(folder / "report.json", "report.json")
            completed = self.job(ident)
            completed.update(state="done", stage="处理完成", progress=100, views=views, statistics=stats)
            # Commit only after every output is complete. Interrupted jobs never
            # appear as successful history after a service restart.
            temp = folder/'job.json.tmp'
            temp.write_text(json.dumps({'boot_id': self.boot_id, 'job': completed}, ensure_ascii=False), encoding='utf-8')
            temp.replace(folder/'job.json')
            self.update(ident, **{k:v for k,v in completed.items() if k != 'id'})
        except Exception as exc:
            self.update(ident, state="error", stage="处理失败", error=str(exc), progress=0)
        finally:
            with self.lock: self.active = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    @property
    def app(self): return self.server.app

    def send(self, code, data, content_type="application/json; charset=utf-8"):
        if not isinstance(data, bytes): data = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; script-src 'self'; style-src 'self'; frame-ancestors 'none'; connect-src 'self'")
        self.end_headers(); self.wfile.write(data)

    def authorized(self):
        return secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + self.app.token)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/app.js", "/style.css", "/favicon.svg"):
            f = ROOT / "static" / ("index.html" if path == "/" else path[1:])
            return self.send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        if not self.authorized(): return self.send(401, {"error": "请使用启动程序提供的完整链接打开网页"})
        try:
            if path == "/api/status": return self.send(200, self.app.status())
            if path == "/api/history": return self.send(200, self.app.history())
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                return self.send(200, self.app.job(parts[2]))
            if len(parts) == 3 and parts[0] == "files":
                ident, name = parts[1:]
                job = self.app.job(ident)
                allowed = [k+".png" for k in job.get("views", [])] + ["results.zip", "report.json"]
                if job["state"] != "done" or name not in allowed: raise KeyError("文件尚不可用")
                f = self.app.data / ident / name
                return self.send(200, f.read_bytes(), mimetypes.guess_type(name)[0] or "application/octet-stream")
            raise KeyError("页面不存在")
        except KeyError as exc: self.send(404, {"error": str(exc)})

    def do_POST(self):
        if not self.authorized(): return self.send(401, {"error": "访问凭证失效，请重新打开启动链接"})
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + self.headers.get("Host", ""):
            return self.send(403, {"error": "不允许跨站控制"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY: return self.send(413, {"error": "上传内容过大或为空"})
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict): raise ValueError("请求格式错误")
            if self.path == "/api/jobs": return self.send(202, self.app.submit(body))
            if self.path == "/api/hdmi":
                if body.get("action") == "stop": self.app.stop_hdmi(); return self.send(200, self.app.status())
                if body.get("action") == "show": return self.send(202, self.app.start_hdmi(body.get("id"), body.get("view")))
                raise ValueError("未知显示操作")
            self.send(404, {"error": "接口不存在"})
        except (ValueError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            self.send(400, {"error": str(exc)})
        except KeyError as exc: self.send(404, {"error": str(exc)})
        except RuntimeError as exc: self.send(409, {"error": str(exc)})


def make_server(app, host="127.0.0.1", port=8765):
    server = ThreadingHTTPServer((host, port), Handler); server.app = app
    return server


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8765)
    p.add_argument("--runtime"); p.add_argument("--data", default=str(ROOT / "data"))
    p.add_argument("--portable", action="store_true"); p.add_argument("--open", action="store_true")
    args = p.parse_args()
    app = Console(args.data, args.runtime, portable=args.portable)
    server = make_server(app, args.host, args.port)
    url = "http://127.0.0.1:{}/#{}".format(server.server_port, app.token)
    print("PLD URL: " + url, flush=True)
    if args.host == "0.0.0.0": print("局域网访问：将上述 127.0.0.1 换成开发板 IP，保留 # 后全部内容", flush=True)
    if args.open:
        import webbrowser
        webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        if app.active:
            print("正在等待当前任务安全结束，请勿强制关闭窗口。", flush=True)
            while app.active:
                time.sleep(.2)
        try: app.stop_hdmi()
        except RuntimeError as exc: print(exc, flush=True)
