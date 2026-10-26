# -*- coding: utf-8 -*-
"""HTTP client for the board's existing three-model web services.

Pure client, exactly like a browser: no listening socket, no FPGA access.
It replicates the proven call pattern from display_web/entry/entry.py
(Origin header, hub token -> HttpOnly cookie, generation URLs).

Services:
  entry (8088): GET /entry/status, POST /entry/switch      (three-model owner)
  hub   (8765): /hub/login, /hub/status, /{model}/g/{gen}/... proxy
  acdc  (8092): /api/v2/* raw ED/ES upload, job poll, assets
"""
import http.client
import json
import time
from urllib.parse import quote


class BoardError(Exception):
    """kind: offline | auth | busy | paused | stale_generation | not_found |
             rejected | fault | switch_failed | protocol"""

    def __init__(self, message, kind="rejected", status=None):
        super().__init__(message)
        self.kind = kind
        self.status = status

    @property
    def retryable(self):
        return self.kind in ("busy", "stale_generation")


SKIN_VIEW_LABELS = {
    "original": "原图", "mask": "二值掩膜", "contour": "轮廓描边",
    "outlined": "轮廓叠加", "cutout": "切割图", "comparison": "对比图",
}
CAMUS_FILES = (
    ("overlay.png", "叠加图"), ("segmentation.png", "分割图"),
    ("label.png", "标签图"), ("label_raw.png", "原始标签图"),
    ("report.json", "结果报告"), ("result.zip", "结果包"),
)
ACDC_MPR_FILES = tuple(
    ("mpr/%s/%s.png" % (phase, axis), "%s %s位" % (phase, label))
    for phase in ("ED", "ES")
    for axis, label in (("axial", "轴"), ("coronal", "冠"), ("sagittal", "矢"))
)
ACDC_DONE_STATUS = "assets_ready"
ACDC_BAD_STATUS = ("failed", "cancelled", "interrupted",
                   "upload_expired", "upload_failed")


def _classify(status, message):
    text = message or ""
    if status == 401:
        return BoardError(text, "auth", status)
    if status == 400 and any(value in text for value in (
            "another task is active", "PL engine is not ready", "HDMI transition is not ready")):
        return BoardError(text, "busy", status)
    if status == 409:
        if "切换" in text or "初始化" in text:
            return BoardError(text, "stale_generation", status)
        return BoardError(text, "busy", status)
    if status == 404:
        return BoardError(text or "资源不存在", "not_found", status)
    if status == 503:
        return BoardError(text or "服务故障", "fault", status)
    if status == 413:
        return BoardError(text or "上传内容过大", "rejected", status)
    return BoardError(text or ("HTTP %d" % status), "rejected", status)


class BoardClient:
    def __init__(self, host, entry_port=8088, hub_port=8765, acdc_port=8092,
                 hub_token="", timeout=15):
        self.host = host
        self.entry_port = int(entry_port)
        self.hub_port = int(hub_port)
        self.acdc_port = int(acdc_port)
        self.hub_token = hub_token
        self.timeout = timeout
        self._hub_cookie = None
        self._generation = None
        # Live boards may bind the hub to 127.0.0.1 only; skin/camus then go
        # through the entry proxy (8088) with the same /{model}/g/{gen}/ paths.
        self._hub_mode = "direct"  # direct | proxy

    # ---------------- low level ----------------

    def _request(self, port, method, path, body=None, headers=None, timeout=None):
        conn = http.client.HTTPConnection(self.host, port, timeout=timeout or self.timeout)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
            data = resp.read()
            return resp.status, dict(resp.getheaders()), data
        finally:
            conn.close()

    def _json_call(self, port, method, path, payload=None, extra_headers=None,
                   raw_body=None, timeout=None):
        headers = dict(extra_headers or {})
        body = raw_body
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        if body is not None:
            headers["Content-Length"] = str(len(body))
        try:
            status, resp_headers, data = self._request(port, method, path, body,
                                                       headers, timeout)
        except (OSError, http.client.HTTPException) as exc:
            if method == "POST" and not isinstance(exc, ConnectionRefusedError):
                raise BoardError("提交响应中断，板端可能已接收任务，请检查板端状态后重试",
                                 "uncertain") from exc
            raise BoardError("板卡 %s:%d 连接失败：%s" % (self.host, port, exc),
                             "offline") from exc
        if status == 404 and method == "GET" and path.startswith("/entry/"):
            return None  # entry not deployed (dual-model board)
        parsed = None
        if data:
            try:
                parsed = json.loads(data.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                parsed = None
        if status not in (200, 202):
            message = None
            if isinstance(parsed, dict):
                message = parsed.get("error") or parsed.get("errmsg")
            raise _classify(status, message)
        if parsed is None and not data:
            return {}
        if parsed is None:
            raise BoardError("%s 返回非 JSON 内容" % path, "protocol", status)
        return parsed

    # ---------------- entry (three-model owner) ----------------

    def entry_status(self):
        """Snapshot dict, or None when the three-model entry is absent."""
        return self._json_call(self.entry_port, "GET", "/entry/status")

    def entry_switch(self, model):
        origin = "http://%s:%d" % (self.host, self.entry_port)
        return self._json_call(self.entry_port, "POST", "/entry/switch",
                               {"model": model},
                               {"Origin": origin, "Content-Type": "application/json"})

    def wait_entry_ready(self, model, deadline, interval=1.0, sleep=time.sleep,
                         clock=time.monotonic):
        """Drive a switch to completion; returns final entry snapshot."""
        return self._wait_ready(self.entry_status, model, deadline, interval, sleep, clock)

    def wait_hub_ready(self, model, deadline, interval=1.0, sleep=time.sleep,
                       clock=time.monotonic):
        return self._wait_ready(self.hub_status, model, deadline, interval, sleep, clock)

    @staticmethod
    def _wait_ready(read, model, deadline, interval, sleep, clock):
        while True:
            try:
                snap = read()
            except BoardError:
                snap = None
            if snap:
                if snap.get("phase") == "error":
                    raise BoardError(snap.get("error") or "模型切换失败",
                                     "switch_failed")
                if snap.get("phase") == "ready" and snap.get("model") == model:
                    return snap
            if clock() >= deadline:
                raise BoardError("等待模型就绪超时", "switch_failed")
            sleep(interval)

    # ---------------- hub (skin / camus) ----------------

    def _origin_hub(self):
        port = self.entry_port if self._hub_mode == "proxy" else self.hub_port
        return "http://%s:%d" % (self.host, port)

    def _use_proxy_mode(self):
        """Switch skin/camus access to the entry proxy (hub loopback-only)."""
        if self._hub_mode == "proxy":
            return
        self._hub_mode = "proxy"
        self._hub_cookie = None
        self._generation = None

    def hub_login(self):
        if self._hub_mode == "proxy":
            return {"ok": True}  # the entry proxy holds its own hub session
        body = json.dumps({"token": self.hub_token}, ensure_ascii=False).encode("utf-8")
        headers = {"Origin": self._origin_hub(),
                   "Content-Type": "application/json",
                   "Content-Length": str(len(body))}
        try:
            status, resp_headers, data = self._request(self.hub_port, "POST",
                                                       "/hub/login", body, headers)
        except (OSError, http.client.HTTPException) as exc:
            raise BoardError("板卡 %s:%d 连接失败：%s" % (self.host, self.hub_port, exc),
                             "offline") from exc
        if status != 200:
            raise BoardError("hub 登录被拒绝（HTTP %d）" % status, "auth", status)
        parsed = {}
        try:
            parsed = json.loads(data.decode("utf-8")) if data else {}
        except (ValueError, UnicodeDecodeError):
            pass
        if not parsed.get("ok"):
            raise BoardError("hub 登录被拒绝", "auth", status)
        cookie = resp_headers.get("Set-Cookie") or ""
        first = cookie.split(";", 1)[0].strip()
        if not first.startswith("pld_hub="):
            raise BoardError("hub 登录未返回会话 Cookie", "auth", status)
        self._hub_cookie = first
        self._generation = None
        return parsed

    def _hub_headers(self, with_origin=True):
        headers = {}
        if with_origin:
            headers["Origin"] = self._origin_hub()
        if self._hub_cookie:
            headers["Cookie"] = self._hub_cookie
        return headers

    def hub_status(self, relogin=True):
        if self._hub_mode == "proxy":
            snap = self.entry_status()
            if snap is None:
                raise BoardError("入口不可达（代理模式）", "offline")
            model = snap.get("model")
            url = snap.get("url") or ""
            generation = None
            if "/g/" in url:
                try:
                    generation = int(url.split("/g/")[1].split("/")[0])
                except (IndexError, ValueError):
                    generation = None
            self._generation = generation
            return {"model": model, "phase": snap.get("phase"),
                    "generation": generation, "error": snap.get("error") or None}
        try:
            if not self._hub_cookie:
                self.hub_login()
                self._generation = None
            return self._json_call(self.hub_port, "GET", "/hub/status", None,
                                   self._hub_headers(False))
        except BoardError as exc:
            if exc.kind == "offline" and relogin and self._hub_mode == "direct":
                self._use_proxy_mode()
                return self.hub_status(relogin=False)
            if exc.kind == "auth" and relogin:
                self._hub_cookie = None
                self._generation = None
                return self.hub_status(relogin=False)
            raise

    def hub_generation(self):
        status = self.hub_status()
        model = status.get("model")
        phase = status.get("phase")
        generation = status.get("generation")
        if phase != "ready":
            raise BoardError("hub 当前状态 %s，不能提交任务" % phase, "busy")
        if generation is None:
            raise BoardError("hub 代数不可用（当前模型 %s）" % model, "busy")
        self._generation = int(generation)
        return model, self._generation

    def hub_switch(self, model):
        if model not in ("skin", "camus"):
            raise BoardError("双模型入口不支持 %s" % model)
        self.hub_status()  # establish/refresh the browser session
        if self._hub_mode != "direct":
            raise BoardError("三模型代理模式请使用统一入口切换")
        return self._json_call(self.hub_port, "POST", "/hub/switch",
                               {"model": model}, self._hub_headers())

    def _hub_path(self, model, rest):
        if self._generation is None:
            selected, _ = self.hub_generation()
            if selected != model:
                raise BoardError("板卡当前运行 %s，请先切换到 %s" % (selected, model), "paused")
        return "/%s/g/%d/%s" % (model, self._generation, rest.lstrip("/"))

    def _hub_call(self, model, method, rest, payload=None, raw_body=None,
                  timeout=None, _retry=True):
        try:
            return self._hub_call_core(model, method, rest, payload,
                                       raw_body, timeout, _retry)
        except BoardError as exc:
            # Live board binds the hub to loopback only: fall back to the
            # entry proxy (same paths, its own hub session) and retry once.
            if exc.kind == "offline" and self._hub_mode == "direct" and _retry:
                self._use_proxy_mode()
                return self._hub_call_core(model, method, rest, payload,
                                           raw_body, timeout, False)
            raise

    def _hub_call_core(self, model, method, rest, payload=None, raw_body=None,
                       timeout=None, _retry=True):
        path = self._hub_path(model, rest)
        port = self.entry_port if self._hub_mode == "proxy" else self.hub_port
        if payload is not None or raw_body is not None:
            headers = self._hub_headers(True)
        else:
            headers = self._hub_headers(False)
        try:
            return self._json_call(port, method, path, payload,
                                   headers, raw_body=raw_body, timeout=timeout)
        except BoardError as exc:
            if self._hub_mode == "direct" and exc.kind == "auth" \
                    and self._hub_cookie and _retry:
                self._hub_cookie = None
                self._generation = None
                return self._hub_call_core(model, method, rest, payload,
                                           raw_body, timeout, _retry=False)
            if exc.kind == "stale_generation" and _retry:
                self._generation = None
                return self._hub_call_core(model, method, rest, payload,
                                           raw_body, timeout, _retry=False)
            if exc.kind == "busy" and "已暂停" in str(exc):
                raise BoardError(str(exc), "paused", exc.status)
            raise

    def invalidate_hub(self):
        self._hub_cookie = None
        self._generation = None

    def _model_file(self, model, job_id, name):
        path = self._hub_path(model, "files/%s/%s" % (job_id, quote(name)))
        port = self.entry_port if self._hub_mode == "proxy" else self.hub_port
        try:
            status, _, data = self._request(port, "GET", path, None,
                                            self._hub_headers(False))
        except (OSError, http.client.HTTPException) as exc:
            raise BoardError("板卡连接失败：%s" % exc, "offline") from exc
        if status != 200:
            raise _classify(status, "%s 文件 %s 获取失败 HTTP %d" % (model, name, status))
        return data

    def _model_file_retry(self, model, job_id, name):
        try:
            return self._model_file(model, job_id, name)
        except BoardError as exc:
            if exc.kind in ("auth", "busy", "stale_generation"):
                self.invalidate_hub()
                self.hub_generation()
                return self._model_file(model, job_id, name)
            if exc.kind == "offline" and self._hub_mode == "direct":
                self._use_proxy_mode()
                return self._model_file(model, job_id, name)
            raise

    # ---------------- skin jobs ----------------

    def skin_submit(self, image_jpeg_bytes, thickness=2, name="miniprogram.jpg"):
        import base64
        payload = {"mode": "infer",
                   "image": base64.b64encode(image_jpeg_bytes).decode("ascii"),
                   "thickness": int(thickness),
                   "name": str(name)[:120],
                   "auto_display": False}
        return self._hub_call("skin", "POST", "/api/jobs", payload, timeout=60)

    def skin_job(self, job_id):
        return self._hub_call("skin", "GET", "/api/jobs/%s" % job_id)

    def skin_file(self, job_id, name):
        return self._model_file_retry("skin", job_id, name)

    # ---------------- camus jobs ----------------

    def camus_submit(self, files, input_name, spacing=None):
        import base64
        payload = {"files": [{"name": name, "data_base64":
                              base64.b64encode(data).decode("ascii")}
                             for name, data in files],
                   "input_name": input_name,
                   "auto_display": False}
        if spacing:
            payload["spacing"] = [float(spacing[0]), float(spacing[1])]
        return self._hub_call("camus", "POST", "/api/jobs", payload, timeout=120)

    def camus_job(self, job_id):
        return self._hub_call("camus", "GET", "/api/jobs/%s" % job_id)

    def camus_file(self, job_id, name):
        return self._model_file_retry("camus", job_id, name)

    # ---------------- acdc jobs (direct, raw stream upload) ----------------

    def acdc_health(self):
        return self._json_call(self.acdc_port, "GET", "/api/v2/health")

    def acdc_submit(self, ed_bytes, es_bytes, ed_name="ED.nii.gz", es_name="ES.nii.gz"):
        total = len(ed_bytes) + len(es_bytes)
        if not 0 < len(ed_bytes) < total:
            raise BoardError("ED/ES 文件内容无效", "rejected")
        if total > 33554432:
            raise BoardError("ED/ES 总体积超过 32MiB", "rejected")
        for n in (ed_name, es_name):
            if not (n.lower().endswith(".nii") or n.lower().endswith(".nii.gz")):
                raise BoardError("ACDC 输入必须是 .nii 或 .nii.gz：%s" % n, "rejected")
        path = "/api/v2/jobs?ed=%s&es=%s" % (quote(ed_name, safe=""),
                                             quote(es_name, safe=""))
        headers = {"Content-Type": "application/octet-stream",
                   "Content-Length": str(total),
                   "X-ED-Bytes": str(len(ed_bytes)), "X-Auto-Display": "false"}
        try:
            status, _, data = self._request(self.acdc_port, "POST", path,
                                            ed_bytes + es_bytes, headers, timeout=60)
        except (OSError, http.client.HTTPException) as exc:
            if not isinstance(exc, ConnectionRefusedError):
                raise BoardError("提交响应中断，板端可能已接收任务，请检查板端状态后重试",
                                 "uncertain") from exc
            raise BoardError("板卡连接失败：%s" % exc, "offline") from exc
        parsed = {}
        try:
            parsed = json.loads(data.decode("utf-8")) if data else {}
        except (ValueError, UnicodeDecodeError):
            pass
        if status not in (200, 202):
            message = parsed.get("error") or parsed.get("detail") if isinstance(parsed, dict) else None
            raise _classify(status, message)
        return parsed

    def acdc_job(self, job_id):
        return self._json_call(self.acdc_port, "GET",
                               "/api/v2/jobs/%s" % job_id)

    def acdc_file(self, job_id, name):
        path = "/api/v2/jobs/%s/assets/%s" % (job_id, quote(name, safe="/"))
        try:
            status, _, data = self._request(self.acdc_port, "GET", path)
        except (OSError, http.client.HTTPException) as exc:
            raise BoardError("板卡连接失败：%s" % exc, "offline") from exc
        if status != 200:
            raise BoardError("ACDC 文件 %s 获取失败 HTTP %d" % (name, status),
                             "not_found", status)
        return data

    def acdc_cancel(self, job_id):
        try:
            return self._json_call(self.acdc_port, "POST",
                                   "/api/v2/jobs/%s/cancel" % job_id, {})
        except BoardError:
            return None  # best effort only
