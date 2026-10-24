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
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, ImageOps
from contour import process
from pld_hdmi import binary_mask, fit_rgb
from pld_result import load_result
from infer_then_display import inference_command, verify_runtime, RuntimeVerifier
from persistent_worker import (PersistentBWorker, PersistentWorkerError,
                               runtime_manifest_sha256)

ROOT = Path(__file__).resolve().parent
MAX_BODY = 24 * 1024 * 1024
MAX_FILE = 8 * 1024 * 1024
MAX_PIXELS = 12_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
VIEWS = ("original", "mask", "contour", "outlined", "cutout", "comparison")
# Lossless PNG: vectorized Sub + zlib RLE, benchmarked locally in
# tests/benchmark_png.py. Keep two bounded writers and close all files before
# packaging/history. Set PLD_PNG_BACKEND=pillow to restore the original encoder.
PNG_BACKEND = os.environ.get("PLD_PNG_BACKEND", "fast_sub")
if PNG_BACKEND not in ("fast_sub", "pillow"):
    raise ValueError("PLD_PNG_BACKEND must be fast_sub or pillow")
PNG_COMPRESS_LEVEL = 1
PNG_SAVE_WORKERS = 2
ZIP_COMPRESSION = zipfile.ZIP_STORED


def save_lossless_png(array, path):
    """8-bit L/RGB PNG; vectorized Sub filter plus zlib RLE level 1.

    Pixels and dimensions are unchanged. Avoid Pillow's per-row adaptive filter
    search on ARM; keep its original encoder available for rollback/benchmark.
    """
    a = np.asarray(array)
    if a.dtype == bool:
        a = a.astype(np.uint8)*255
    if a.dtype != np.uint8 or not (a.ndim == 2 or (a.ndim == 3 and a.shape[2] == 3)):
        raise ValueError('PNG requires uint8 L or RGB')
    if PNG_BACKEND == 'pillow':
        Image.fromarray(a).save(path, compress_level=PNG_COMPRESS_LEVEL)
        return
    h, w = a.shape[:2]
    if h < 1 or w < 1: raise ValueError('empty PNG')
    channels = 1 if a.ndim == 2 else 3
    raw = np.ascontiguousarray(a).reshape(h, w*channels)
    rows = np.empty((h, w*channels+1), dtype=np.uint8)
    rows[:,0] = 1 # PNG Sub filter, applied independently to each byte channel.
    rows[:,1:1+channels] = raw[:,:channels]
    np.subtract(raw[:,channels:], raw[:,:-channels], out=rows[:,1+channels:])
    def chunk(kind, data):
        crc = zlib.crc32(data, zlib.crc32(kind)) & 0xffffffff
        return struct.pack('!I',len(data))+kind+data+struct.pack('!I',crc)
    header = struct.pack('!IIBBBBB',w,h,8,0 if channels==1 else 2,0,0,0)
    compressor = zlib.compressobj(level=1, strategy=zlib.Z_RLE)
    compressed = compressor.compress(rows.tobytes()) + compressor.flush()
    encoded = (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',header)+
               chunk(b'IDAT',compressed)+chunk(b'IEND',b''))
    Path(path).write_bytes(encoded)


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
                # Avoid full-frame copies when no orientation/conversion is needed.
                oriented = ImageOps.exif_transpose(im) if im.getexif().get(274, 1) in range(2, 9) else im
                arr = np.array(oriented if oriented.mode == "RGB" else oriented.convert("RGB"))
    return raw, arr


class Console:
    def __init__(self, data, runtime=None, token=None, portable=False, boot_id=None,
                 persistent_b_worker=False):
        self.data = Path(data).resolve(); self.data.mkdir(parents=True, exist_ok=True)
        self.runtime = Path(runtime).resolve() if runtime else None
        self.token = token if token is not None else persistent_token(self.data)
        self.portable = portable
        self.persistent_b_worker = persistent_b_worker
        self.b_worker = None
        self.verifier = RuntimeVerifier(self.runtime) if self.runtime else None
        self.warmup_state = "not_started"
        self.warmup_seconds = None
        self.warmup_thread = None
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
                self.verify_deployment()
                inference_command(self.runtime, Path("check.jpg"), Path("check_out"), self.portable)
                if self.persistent_b_worker:
                    runtime_manifest_sha256(self.runtime)
                self.runtime_ok = True
            except Exception as exc:
                self.runtime_error = str(exc)

    def verify_deployment(self):
        try:
            if self.verifier is None:
                self.verifier = RuntimeVerifier(self.runtime)
            return self.verifier.check()
        except Exception:
            self.fault = "运行包校验失败或已变化，请重启服务并检查部署文件"
            raise

    def begin_warmup(self):
        with self.lock:
            if not (self.board and self.runtime_ok and self.persistent_b_worker):
                return
            if self.warmup_state != 'not_started': return
            if self.active or self.hdmi is not None:
                raise RuntimeError('不能在处理或显示期间预热')
            self.warmup_state = 'running'
            self.warmup_thread = threading.Thread(target=self._warmup, name='pld-warmup', daemon=True)
            self.warmup_thread.start()

    def _warmup(self):
        started = time.perf_counter()
        try:
            self.verify_deployment()
            command = inference_command(self.runtime, self.data/'warmup-unused.jpg',
                                        self.data/'warmup-unused-output', self.portable)
            if self.b_worker is None: self.b_worker = PersistentBWorker(self.runtime)
            self.b_worker.run(command[2:]+['--warmup-only'], self.data/'warmup.log')
            with self.lock: self.warmup_state = 'ready'
        except Exception as exc:
            with self.lock:
                self.warmup_state = 'failed'
                self.fault = '开机预热失败，请查看 warmup.log 并重启服务：'+str(exc)
        finally:
            with self.lock: self.warmup_seconds = time.perf_counter()-started

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
            return {"mode": "board" if self.board else "local", "inference_available": self.runtime_ok and not self.fault and self.warmup_state != "running",
                    "warmup": self.warmup_state, "warmup_seconds": self.warmup_seconds,
                    "hdmi_available": self.board and not self.fault and self.warmup_state != "running", "active": self.active,
                    "hdmi": "showing" if ready else "starting" if running else "off",
                    "hdmi_view": self.hdmi_view if running else None,
                    "fault": self.fault, "runtime_error": self.runtime_error}

    def job(self, ident):
        with self.lock:
            if ident not in self.jobs: raise KeyError("任务不存在")
            return json.loads(json.dumps(self.jobs[ident]))

    def update(self, ident, **values):
        with self.lock: self.jobs[ident].update(values)

    def submit(self, body, request_started=None):
        submit_started = time.perf_counter()
        mode = body.get("mode")
        if mode not in ("demo", "mask", "infer"): raise ValueError("未知处理模式")
        width = body.get("thickness", 2)
        if type(width) is not int or not 1 <= width <= 8: raise ValueError("线宽须为 1–8 的整数")
        with self.lock:
            if self.warmup_state == "running": raise RuntimeError("模型和 FPGA 正在预热，请稍候")
            if self.active: raise RuntimeError("已有任务处理中，请等待完成")
            if self.fault: raise RuntimeError(self.fault)
            if mode == "infer" and not self.runtime_ok:
                raise ValueError("当前未连接可用的板端推理环境；可使用真实结果回放或上传配套掩膜")
            upload_started = time.perf_counter()
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
            upload_stages = {"request_receive_parse": max(0, submit_started-(request_started or submit_started)),
                             "upload_decode_validate": time.perf_counter()-upload_started}
            ident = uuid.uuid4().hex
            self.jobs[ident] = {"id": ident, "state": "running", "stage": "准备数据", "progress": 10,
                                "mode": mode, "name": str(body.get("name", "ISIC_0000003 · 真实掩膜回放"))[:120],
                                "thickness": width, "created": time.time()}
            self.active = ident
            threading.Thread(target=self._run,
                             args=(ident, mode, raw, rgb, mask, width,
                                   request_started or submit_started, upload_stages),
                             daemon=True).start()
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

    def close(self):
        """Stop the optional idle B worker without disturbing HDMI ownership."""
        if self.warmup_thread is not None:
            self.warmup_thread.join()
        with self.lock:
            worker = self.b_worker
            self.b_worker = None
        if worker is not None:
            worker.close()

    def start_hdmi(self, ident, view):
        with self.lock:
            self.status()
            if not self.board: raise ValueError("HDMI 需要在 PYNQ 板端运行本服务")
            if self.fault: raise RuntimeError(self.fault)
            if self.warmup_state == "running": raise RuntimeError("模型和 FPGA 正在预热，请稍候")
            if self.active: raise RuntimeError("请等待处理完成后再显示")
            j = self.job(ident)
            if j["state"] != "done" or (view not in j["views"] and view != "interactive"): raise ValueError("该显示画面尚未生成")
            self.stop_hdmi()
            folder = self.data / ident
            if self.b_worker is not None:
                try:
                    self.b_worker.release_hardware(folder / "inference_release.log")
                except Exception as exc:
                    self.fault = "推理硬件释放未确认，禁止启动 HDMI；请检查日志并重启服务"
                    raise RuntimeError(self.fault) from exc
            (folder / "hdmi.ready").unlink(missing_ok=True)
            with (folder / "hdmi.log").open("w", encoding="utf-8") as log:
                self.hdmi = subprocess.Popen([sys.executable, str(ROOT / "hdmi_worker.py"), str(folder), view], stdout=log, stderr=log)
            self.hdmi_job, self.hdmi_view = ident, view
            return self.status()

    def _run(self, ident, mode, raw, rgb, mask, width, request_started, upload_stages=None):
        folder = self.data / ident; folder.mkdir()
        job_start = time.perf_counter()
        stages = dict(upload_stages or {})
        stages["request_to_job_start"] = job_start - request_started
        png_timings = {}
        try:
            b = None
            if mode == "demo":
                stage = time.perf_counter()
                b = load_result(ROOT / "samples/frame_000001.json")
                stages["result_load"] = time.perf_counter() - stage
                stage = time.perf_counter()
                mask = b.original_mask()
                stages["mask_remap"] = time.perf_counter() - stage
            elif mode == "infer":
                stage = time.perf_counter()
                self.stop_hdmi()
                stages["stop_hdmi"] = time.perf_counter() - stage
                self.update(ident, stage="FPGA 分割中", progress=30)
                source = folder / "input.jpg"
                stage = time.perf_counter(); source.write_bytes(raw)
                stages["input_write"] = time.perf_counter() - stage
                if self.persistent_b_worker and self.b_worker is not None:
                    # A live worker holds in-memory code from the manifest it was
                    # launched with. Refuse it after any deployment change.
                    try:
                        self.b_worker.check_manifest()
                    except PersistentWorkerError as exc:
                        self.fault = str(exc)
                        raise
                stage = time.perf_counter(); stages["runtime_check_kind"] = self.verify_deployment()
                stages["runtime_verify"] = time.perf_counter() - stage
                command = inference_command(self.runtime, source, folder / "inference", self.portable)
                if self.persistent_b_worker:
                    stage = time.perf_counter()
                    if self.b_worker is None:
                        self.b_worker = PersistentBWorker(self.runtime)
                    try:
                        response = self.b_worker.run(command[2:], folder / "inference.log")
                    except PersistentWorkerError as exc:
                        self.fault = str(exc)
                        raise
                    stages["model_reused"] = response.get("model_reused", False)
                    stages["hardware_reused"] = response.get("hardware_reused", False)
                    stages["persistent_worker_startup"] = response["startup_seconds"]
                    stages["persistent_worker_request"] = response["request_seconds"]
                    stages["persistent_worker_pid"] = response["worker_pid"]
                    stages["persistent_worker_alive_after_reply"] = response[
                        "alive_after_reply"]
                    stages["persistent_worker_total"] = time.perf_counter() - stage
                    stages["inference_backend"] = "persistent_b_worker"
                else:
                    subprocess_start = time.perf_counter()
                    with (folder / "inference.log").open("w", encoding="utf-8") as log:
                        p = subprocess.Popen(command, cwd=str(self.runtime), stdout=log, stderr=log)
                        launch_done = time.perf_counter()
                        stages["inference_launch"] = launch_done - subprocess_start
                        try: code = p.wait(timeout=180)
                        except subprocess.TimeoutExpired:
                            p.kill(); p.wait()
                            self.fault = "推理超时，需在板端确认 DMA 状态并重启服务；已禁止新硬件操作"
                            raise RuntimeError(self.fault)
                        stages["inference_wait"] = time.perf_counter() - launch_done
                        stages["inference_subprocess"] = time.perf_counter() - subprocess_start
                    if code != 0:
                        self.fault = "板端推理失败，请检查 inference.log 并确认硬件状态后重启服务"
                        raise RuntimeError(self.fault)
                    stages["inference_backend"] = "subprocess"
                stage = time.perf_counter()
                b = load_result(folder / "inference/frame_000001.json", source, decoded_rgb=rgb)
                stages["result_load"] = time.perf_counter() - stage
                stage = time.perf_counter()
                mask = b.original_mask()
                stages["mask_remap"] = time.perf_counter() - stage
                rgb = b.rgb
            self.update(ident, stage="提取轮廓与统计面积", progress=75)
            stage = time.perf_counter(); result = process(mask, rgb, thickness=width)
            stages["contour_process"] = time.perf_counter() - stage
            views = ["mask", "contour", "outlined", "cutout"] if rgb is not None else ["mask", "contour", "outlined"]
            arrays = {k: result[k] for k in views}
            if rgb is not None:
                arrays["original"] = rgb; views.insert(0, "original")
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
            # Publish only after every small preview has been fully written.
            # Full arrays remain untouched for export and HDMI.
            def save_preview(key):
                array = arrays[key]
                pixels = array.astype(np.uint8)*255 if array.dtype == bool else array
                im = Image.fromarray(pixels)
                resample = getattr(Image, "Resampling", Image)
                started = time.perf_counter()
                im.thumbnail((800, 800), resample.NEAREST if pixels.ndim == 2 else resample.BILINEAR)
                resize_seconds = time.perf_counter()-started
                started = time.perf_counter()
                save_lossless_png(np.asarray(im), folder / ("preview_"+key+".png"))
                preview_details[key] = {"resize":resize_seconds, "encode_write":time.perf_counter()-started}
            preview_started = time.perf_counter()
            preview_details = {}
            preview_views = ["original" if rgb is not None else "mask", "outlined"]
            for key in preview_views: save_preview(key)
            stages["preview_build"] = time.perf_counter() - preview_started
            stages["request_to_preview_ready"] = time.perf_counter() - request_started
            stages["preview_images"] = dict(preview_details)
            stats["stage_seconds"] = dict(stages)
            stats["preview_ready_seconds"] = stages["request_to_preview_ready"]
            stats["data_source"] = "历史板测掩膜回放" if mode == "demo" else "用户图片及掩膜" if mode == "mask" else "本次板端 FPGA 推理"
            self.update(ident, preview_ready=True, package_ready=False,
                        preview_views=preview_views, views=list(views),
                        statistics=json.loads(json.dumps(stats)), progress=82,
                        stage="预览已就绪，后台生成原尺寸图片和结果包")
            remaining_started = time.perf_counter()
            for key in views:
                if key not in preview_views: save_preview(key)
            stages["preview_remaining"] = time.perf_counter()-remaining_started
            stages["preview_images"] = preview_details
            self.update(ident, preview_views=list(views))
            left = rgb if rgb is not None else np.repeat((mask.astype(np.uint8)*255)[:, :, None], 3, axis=2)
            stage = time.perf_counter()
            arrays["comparison"] = np.concatenate((fit_rgb(left, 640, 480), fit_rgb(result["outlined"], 640, 480)), axis=1)
            stages["view_composition"] = time.perf_counter() - stage
            views.append("comparison")

            def save_png_group(group):
                for key, array in group:
                    started = time.perf_counter()
                    save_lossless_png(array, folder / (key+".png"))
                    png_timings[key] = time.perf_counter() - started

            # Split this one batch into exactly one deterministic group per worker:
            # at most two PNG encoders run concurrently, there is no pending queue
            # beyond the six views, and each view is converted and written once.
            # Leaving the executor context joins both workers, so ZIP/report and
            # the successful history entry are only created after every file is
            # closed. Any save error propagates and fails the whole job instead.
            items = [(key, arrays[key]) for key in views]
            png_batch_started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=PNG_SAVE_WORKERS,
                                    thread_name_prefix="pld-png") as pool:
                futures = [pool.submit(save_png_group, items[index::PNG_SAVE_WORKERS])
                           for index in range(PNG_SAVE_WORKERS)]
                for future in futures:
                    future.result()
            stages["png_batch_wall"] = time.perf_counter()-png_batch_started
            stages["png_save"] = {key: png_timings[key] for key in views}
            stats["processing_seconds"] = time.perf_counter()-job_start
            stats["processing_scope"] = "旧口径：任务线程启动到全部 PNG 写出；不含上传/浏览器、report/ZIP/历史提交；推理模式包含停止HDMI、校验和B子进程"
            stats["data_source"] = "历史板测掩膜，无原始 JPG；未重新推理" if mode == "demo" else "用户图片及掩膜；未执行模型推理" if mode == "mask" else "本次板端 FPGA 推理"
            stats["output_png_compress_level"] = PNG_COMPRESS_LEVEL
            stats["output_png_workers"] = PNG_SAVE_WORKERS
            stats["output_png_backend"] = PNG_BACKEND
            stats["output_png_bytes"] = sum((folder/(name+".png")).stat().st_size for name in views)
            stats["output_zip_compression"] = "stored"
            # Package the costly PNG payload first. This lets report.json state
            # exactly which stages were already complete when it was serialized;
            # the API/job index later receives the report/append/final totals.
            stage = time.perf_counter()
            with zipfile.ZipFile(folder / "results.zip", "w", compression=ZIP_COMPRESSION) as z:
                for name in views: z.write(folder / (name+".png"), name+".png")
            stages["zip_payload"] = time.perf_counter() - stage
            stats["stage_seconds"] = dict(stages)
            stats["stage_timing_scope"] = ("report/ZIP snapshot at report serialization: request→job start and stages "
                                           "through all PNG writes and ZIP PNG payload; excludes report_write, "
                                           "ZIP report append and final totals")
            stage = time.perf_counter()
            report = json.dumps(stats, ensure_ascii=False, indent=2)
            (folder / "report.json").write_text(report, encoding="utf-8")
            stages["report_write"] = time.perf_counter() - stage
            # Append keeps member order identical to the historical package.
            stage = time.perf_counter()
            with zipfile.ZipFile(folder / "results.zip", "a", compression=ZIP_COMPRESSION) as z:
                z.write(folder / "report.json", "report.json")
            stages["zip_report_append"] = time.perf_counter() - stage
            stages["zip_package"] = stages["zip_payload"] + stages["zip_report_append"]
            stages["job_total"] = time.perf_counter() - job_start
            stages["request_to_job_complete"] = time.perf_counter() - request_started
            stats["final_stage_seconds"] = dict(stages)
            stats["final_stage_timing_scope"] = ("time.perf_counter wall seconds through ZIP close (report included); "
                                                 "inference_subprocess overlaps its launch/wait components; history "
                                                 "index commit is not included")
            stats["job_total_seconds"] = stages["job_total"]
            stats["request_to_job_complete_seconds"] = stages["request_to_job_complete"]
            completed = self.job(ident)
            completed.update(state="done", package_ready=True, stage="完整结果包已就绪", progress=100, views=views, statistics=stats)
            # Commit only after every output is complete. Interrupted jobs never
            # appear as successful history after a service restart.
            temp = folder/'job.json.tmp'
            temp.write_text(json.dumps({'boot_id': self.boot_id, 'job': completed}, ensure_ascii=False), encoding='utf-8')
            temp.replace(folder/'job.json')
            self.update(ident, **{k:v for k,v in completed.items() if k != 'id'})
        except Exception as exc:
            self.update(ident, state="error", package_ready=False, stage="完整结果生成失败" if self.job(ident).get("preview_ready") else "处理失败", error=str(exc), progress=0,
                        stage_seconds=stages, png_save_seconds=png_timings)
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
                previews = ["preview_"+k+".png" for k in job.get("preview_views", [])]
                allowed = [k+".png" for k in job.get("views", [])] + ["results.zip", "report.json"]
                ready_preview = job.get("preview_ready") and name in previews
                ready_export = job["state"] == "done" and name in allowed
                if not (ready_preview or ready_export): raise KeyError("文件尚不可用")
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
            request_started = time.perf_counter()
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY: return self.send(413, {"error": "上传内容过大或为空"})
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict): raise ValueError("请求格式错误")
            if self.path == "/api/jobs": return self.send(202, self.app.submit(body, request_started))
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
    p.add_argument("--persistent-b-worker", action="store_true",
                   help="persistent B session; explicit hardware release before HDMI")
    args = p.parse_args()
    app = Console(args.data, args.runtime, portable=args.portable,
                  persistent_b_worker=args.persistent_b_worker)
    server = make_server(app, args.host, args.port)
    url = "http://127.0.0.1:{}/#{}".format(server.server_port, app.token)
    print("PLD URL: " + url, flush=True)
    if args.host == "0.0.0.0": print("局域网访问：将上述 127.0.0.1 换成开发板 IP，保留 # 后全部内容", flush=True)
    if args.open:
        import webbrowser
        webbrowser.open(url)
    app.begin_warmup()
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
        app.close()
