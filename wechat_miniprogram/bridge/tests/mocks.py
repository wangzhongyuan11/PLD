# -*- coding: utf-8 -*-
"""In-memory mocks for bridge tests: board services, WeChat API, cloud DB."""
import base64
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs


class _Silent(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass


class MockBoard:
    """Entry(8088-like) + hub(8765-like) + acdc(8092-like) on ephemeral ports."""

    HUB_TOKEN = "test-hub-token-000000000"

    def __init__(self):
        self.lock = threading.Lock()
        # entry state
        self.entry_enabled = True
        self.entry_state = {"model": "skin", "phase": "ready",
                            "url": "/skin/g/3/"}
        self.entry_next_ready = None      # model to expose after next status call
        self.entry_switch_calls = []
        self.entry_require_origin = True
        self.entry_generation = 3
        # hub state
        self.hub_generation = 3
        self.hub_model = "skin"
        self.hub_suspended_model = None   # model whose backend answers '已暂停'
        self.hub_busy_polls = 0           # submit 409 while > 0
        self.hub_closed = False           # simulate loopback-only hub
        # jobs shared by skin/camus mocks (poll-driven progression)
        self.jobs = {}
        self.polls_to_done = 2
        # acdc state
        self.acdc_busy_polls = 0
        self.acdc_cancelled = []
        self.acdc_uploads = []
        self.acdc_polls_to_done = 2
        self._servers = []
        self._hub_server = None

    # ---------- lifecycle ----------

    def start(self):
        for factory, attr in ((self._entry_handler, "entry"),
                              (self._hub_handler, "hub"),
                              (self._acdc_handler, "acdc")):
            server = ThreadingHTTPServer(("127.0.0.1", 0), factory())
            server.board = self
            setattr(self, attr + "_port", server.server_address[1])
            if attr == "hub":
                self._hub_server = server
            thread = threading.Thread(target=server.serve_forever,
                                      kwargs={"poll_interval": 0.02},
                                      daemon=True)
            thread.start()
            self._servers.append(server)
        return self

    def close_hub(self):
        """Simulate a hub bound to loopback only: port refuses connections."""
        if self._hub_server:
            self._hub_server.shutdown()
            self._hub_server.server_close()
            self._servers.remove(self._hub_server)
            self._hub_server = None

    def stop(self):
        for server in self._servers:
            server.shutdown()
            server.server_close()

    # ---------- helpers ----------

    @staticmethod
    def _send(handler, code, payload, headers=None):
        body = payload if isinstance(payload, bytes) else json.dumps(
            payload, ensure_ascii=False).encode("utf-8")
        handler.send_response(code)
        handler.send_header("Content-Type",
                            "application/json; charset=utf-8"
                            if not isinstance(payload, bytes) else
                            "application/octet-stream")
        handler.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            handler.send_header(key, value)
        handler.end_headers()
        handler.wfile.write(body)

    def _advance_job(self, job_id):
        job = self.jobs.get(job_id)
        if job is None:
            return None
        job["polls"] = job.get("polls", 0) + 1
        if job["polls"] >= self.polls_to_done and job.get("state") != "done":
            job["state"] = "done"
            job.update(job["finish"])
        return job

    # ---------- shared skin/camus job logic (hub + entry proxy) ----------

    def _handle_job_submit(self, model, raw):
        body = json.loads(raw or b"{}")
        if self.hub_busy_polls > 0:
            self.hub_busy_polls -= 1
            return 409, {"error": "已有任务处理中，请等待完成"}
        if self.hub_suspended_model == model:
            return 409, {"error": "模型已暂停，请先通过统一入口选择皮肤镜"}
        if body.get("auto_display") is not False:
            return 400, {"error": "auto_display 必须为 false"}
        job_id = uuid.uuid4().hex
        if model == "skin":
            job = {"id": job_id, "state": "running", "stage": "FPGA 分割中",
                   "progress": 30,
                   "finish": {"state": "done", "progress": 100,
                              "views": ["original", "mask", "contour",
                                        "outlined", "cutout", "comparison"],
                              "statistics": {
                                  "area_percent": 23.45,
                                  "foreground_pixels": 1000,
                                  "image_pixels": 5000,
                                  "b_seconds": 0.758,
                                  "job_total_seconds": 4.2}}}
        else:
            job = {"id": job_id, "state": "running",
                   "finish": {"state": "done",
                              "result": {"backend": "pl",
                                         "end_to_end_seconds": 3.1}}}
        self.jobs[job_id] = job
        return 202, {"id": job_id, "state": "queued"}

    def _serve_file(self, name):
        if name == "report.json":
            payload = json.dumps({"output": {
                "class_names": ["background", "lv_cavity",
                                "myocardium", "left_atrium"],
                "areas": {"background": {"pixels": 0}, "lv_cavity": {"pixels": 50, "area_mm2": 12.5},
                          "myocardium": {"pixels": 33}, "left_atrium": {"pixels": 20}}}}).encode("utf-8")
        else:
            payload = b"\x89PNG-fake-" + name.encode()
        return payload

    def _ready_state(self, model):
        if model == "acdc":
            return {"model": "acdc", "phase": "ready", "url": "8092/"}
        return {"model": model, "phase": "ready",
                "url": "/%s/g/%d/" % (model, self.hub_generation)}

    # ---------- entry proxy routes (used when hub is loopback-only) ----------

    def _proxy_get(self, parts):
        """Handle GET /{model}/g/{n}/... on the entry origin. None = 404."""
        if int(parts[2]) != self.entry_generation:
            return 409, {"error": "模型已切换或正在初始化"}
        if len(parts) == 6 and parts[3] == "api" and parts[4] == "jobs":
            job = self._advance_job(parts[5])
            if job is None:
                return 404, {"error": "任务不存在"}
            return 200, job
        if len(parts) >= 6 and parts[3] == "files":
            return 200, self._serve_file(parts[5])
        return None

    # ---------- entry ----------

    def _entry_handler(self):
        board = self

        class Handler(_Silent):
            def do_GET(self):
                path = urlsplit(self.path).path
                if not board.entry_enabled or path != "/entry/status":
                    # proxied dual-model pages when the hub is loopback-only
                    parts = path.strip("/").split("/")
                    if (board.entry_enabled and len(parts) >= 4
                            and parts[0] in ("skin", "camus")
                            and parts[1] == "g"):
                        result = board._proxy_get(parts)
                        if result is not None:
                            code, payload = result
                            return board._send(self, code, payload)
                    board._send(self, 404, {"error": "not found"})
                    return
                if board.entry_next_ready is not None:
                    with board.lock:
                        board.entry_state = board._ready_state(
                            board.entry_next_ready)
                        board.entry_generation = board.hub_generation
                    board.entry_next_ready = None
                board._send(self, 200, board.entry_state)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length or b"")
                path = urlsplit(self.path).path
                parts = path.strip("/").split("/")
                # proxied dual-model submit (entry proxy holds hub session)
                if (board.entry_enabled and len(parts) >= 5
                        and parts[0] in ("skin", "camus") and parts[1] == "g"
                        and parts[3] == "api" and parts[4:] == ["jobs"]):
                    origin = self.headers.get("Origin", "")
                    if origin != "http://127.0.0.1:%d" % board.entry_port:
                        return board._send(self, 403, {"error": "不允许跨站控制"})
                    if int(parts[2]) != board.entry_generation:
                        return board._send(self, 409,
                                           {"error": "模型已切换或正在初始化"})
                    code, payload = board._handle_job_submit(parts[0], body)
                    return board._send(self, code, payload)
                if path != "/entry/switch":
                    return board._send(self, 404, {"error": "not found"})
                origin = self.headers.get("Origin", "")
                if board.entry_require_origin and \
                        origin != "http://127.0.0.1:%d" % board.entry_port:
                    return board._send(self, 403, {"error": "不允许跨站控制"})
                parsed = json.loads(body or b"{}")
                model = parsed.get("model")
                if model not in ("skin", "camus", "acdc"):
                    return board._send(self, 400, {"error": "未知模型"})
                board.entry_switch_calls.append(model)
                with board.lock:
                    board.entry_state = {"model": model, "phase": "switching"}
                    board.entry_next_ready = model
                    if model != "acdc":
                        board.hub_model = model
                        board.hub_generation += 1
                board._send(self, 202, {"model": model, "phase": "switching"})

        return Handler

    # ---------- hub (login + generation proxy + skin/camus jobs) ----------

    def _hub_handler(self):
        board = self
        COOKIE = "pld_hub=s3ss"

        class Handler(_Silent):
            def _authed(self):
                return self.headers.get("Cookie", "").startswith("pld_hub=")

            def _origin_ok(self):
                origin = self.headers.get("Origin")
                return not origin or origin == "http://127.0.0.1:%d" % board.hub_port

            def do_GET(self):
                path = urlsplit(self.path).path
                if path == "/hub/status":
                    if not self._authed():
                        return board._send(self, 401, {"error": "未登录"})
                    return board._send(self, 200, {
                        "model": board.hub_model, "phase": "ready",
                        "generation": board.hub_generation})
                parts = path.strip("/").split("/")
                if len(parts) >= 4 and parts[1] == "g" and parts[3] == "api":
                    if not self._authed():
                        return board._send(self, 401, {"error": "未登录"})
                    if int(parts[2]) != board.hub_generation:
                        return board._send(self, 409, {"error": "模型已切换或正在初始化"})
                    if len(parts) == 5 and parts[4] == "status":
                        return board._send(self, 200, {"ready": True})
                    job = board._advance_job(parts[5]) if len(parts) == 6 else None
                    if job is None:
                        return board._send(self, 404, {"error": "任务不存在"})
                    return board._send(self, 200, job)
                if len(parts) >= 5 and parts[3] == "files" and parts[1] == "g":
                    if not self._authed():
                        return board._send(self, 401, {"error": "未登录"})
                    if int(parts[2]) != board.hub_generation:
                        return board._send(self, 409, {"error": "模型已切换或正在初始化"})
                    name = parts[5]
                    payload = board._serve_file(name)
                    return board._send(self, 200, payload)
                return board._send(self, 404, {"error": "not found"})

            def do_POST(self):
                path = urlsplit(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                if path == "/hub/login":
                    body = json.loads(raw or b"{}")
                    if body.get("token") != board.HUB_TOKEN:
                        return board._send(self, 401, {"error": "访问码错误"})
                    if not self._origin_ok():
                        return board._send(self, 403, {"error": "不允许跨站控制"})
                    return board._send(self, 200, {"ok": True},
                                       {"Set-Cookie": COOKIE + "; Path=/"})
                if not self._authed():
                    return board._send(self, 401, {"error": "未登录"})
                if not self._origin_ok():
                    return board._send(self, 403, {"error": "不允许跨站控制"})
                if path == "/hub/switch":
                    model = json.loads(raw)["model"]
                    board.hub_model = model
                    board.hub_generation += 1
                    return board._send(self, 202, {"model": model, "phase": "ready"})
                parts = path.strip("/").split("/")
                if len(parts) >= 4 and parts[1] == "g" and \
                        parts[3] == "api" and parts[4:] == ["jobs"]:
                    if int(parts[2]) != board.hub_generation:
                        return board._send(self, 409, {"error": "模型已切换或正在初始化"})
                    body = json.loads(raw or b"{}")
                    if board.hub_busy_polls > 0:
                        board.hub_busy_polls -= 1
                        return board._send(self, 409, {"error": "已有任务处理中，请等待完成"})
                    if board.hub_suspended_model == parts[0]:
                        return board._send(self, 409, {"error": "模型已暂停，请先通过统一入口选择皮肤镜"})
                    if body.get("auto_display") is not False:
                        return board._send(self, 400, {"error": "auto_display 必须为 false"})
                    job_id = uuid.uuid4().hex
                    if parts[0] == "skin":
                        job = {"id": job_id, "state": "running", "stage": "FPGA 分割中",
                               "progress": 30,
                               "finish": {"state": "done", "progress": 100,
                                          "views": ["original", "mask", "contour",
                                                    "outlined", "cutout", "comparison"],
                                          "statistics": {
                                              "area_percent": 23.45,
                                              "foreground_pixels": 1000,
                                              "image_pixels": 5000,
                                              "b_seconds": 0.758,
                                              "job_total_seconds": 4.2}}}
                    else:
                        job = {"id": job_id, "state": "running",
                               "finish": {"state": "done",
                                          "result": {"backend": "pl",
                                                     "end_to_end_seconds": 3.1}}}
                    board.jobs[job_id] = job
                    return board._send(self, 202, {"id": job_id, "state": "queued"})
                return board._send(self, 404, {"error": "not found"})

        return Handler

    # ---------- acdc ----------

    def _acdc_handler(self):
        board = self
        STATUSES = ["queued", {"status": "preparing", "phase": "load"},
                    {"status": "infer_restore", "phase_completed": 4, "phase_total": 8},
                    "assets_ready"]

        class Handler(_Silent):
            def do_GET(self):
                parts = urlsplit(self.path).path.strip("/").split("/")
                if parts[:3] == ["api", "v2", "health"]:
                    return board._send(self, 200,
                                       {"status": "ready", "ready": True,
                                        "active_job": None, "backend": "pl",
                                        "lifecycle": "pause-resume-v1",
                                        "suspended": False, "hdmi": "off", "task_auto_display": True})
                if parts[:3] == ["api", "v2", "jobs"] and len(parts) == 4:
                    job = board.jobs.get(parts[3])
                    if job is None:
                        return board._send(self, 404, {"error": "unknown job"})
                    job["polls"] = job.get("polls", 0) + 1
                    idx = min(job["polls"], len(STATUSES) - 1)
                    status = STATUSES[idx]
                    if isinstance(status, str):
                        job["status"] = status
                    else:
                        job.update(status)
                    return board._send(self, 200, job)
                if parts[:3] == ["api", "v2", "jobs"] and len(parts) >= 5 and \
                        parts[4] == "assets":
                    job = board.jobs.get(parts[3])
                    if job is None or job.get("status") != "assets_ready":
                        return board._send(self, 404, {"error": "not ready"})
                    name = "/".join(parts[5:])
                    if name == "statistics.csv":
                        payload = (b"phase,structure,voxels,volume_mL\n"
                                   b"ED,lv,1000,12.5\nES,lv,600,7.2\n")
                    elif name == "cardiac_function.csv":
                        payload = (b"structure,EDV_mL,ESV_mL,SV_mL,EF_percent,valid,reason\n"
                                   b"LV,120.5,55.2,65.3,54.3,True,\n")
                    elif name == "manifest.json":
                        payload = json.dumps({"files": {}}).encode("utf-8")
                    else:
                        payload = b"ACDC-asset:" + name.encode()
                    return board._send(self, 200, payload)
                return board._send(self, 404, {"error": "not found"})

            def do_POST(self):
                parts = urlsplit(self.path).path.strip("/").split("/")
                length = int(self.headers.get("Content-Length", "0"))
                if parts[:3] == ["api", "v2", "jobs"] and len(parts) == 3:
                    body = self.rfile.read(length)  # drain before rejecting
                    if "Origin" in self.headers and \
                            self.headers.get("Origin") != "http://127.0.0.1:%d" % board.acdc_port:
                        return board._send(self, 401, {"error": "cross-origin refused"})
                    if board.acdc_busy_polls > 0:
                        board.acdc_busy_polls -= 1
                        return board._send(self, 400, {"error": "ContractError: another task is active"})
                    ed_bytes = int(self.headers.get("X-ED-Bytes", "-1"))
                    query = parse_qs(urlsplit(self.path).query)
                    if not 0 < ed_bytes < len(body):
                        return board._send(self, 400, {"error": "invalid upload metadata"})
                    job_id = uuid.uuid4().hex
                    board.jobs[job_id] = {"run_id": job_id, "status": "queued", "polls": 0}
                    board.acdc_uploads.append({
                        "ed": body[:ed_bytes], "es": body[ed_bytes:],
                        "auto_display": self.headers.get("X-Auto-Display"),
                        "ed_name": query.get("ed", [""])[0],
                        "es_name": query.get("es", [""])[0]})
                    return board._send(self, 202, {"run_id": job_id, "status": "queued"})
                if parts[:3] == ["api", "v2", "jobs"] and len(parts) == 5 and \
                        parts[4] == "cancel":
                    board.acdc_cancelled.append(parts[3])
                    return board._send(self, 202, {"cancelling": True})
                return board._send(self, 404, {"error": "not found"})

        return Handler


class MockWechatApi:
    """Recording stub for api.weixin.qq.com endpoints."""

    def __init__(self):
        self.lock = threading.Lock()
        self.token_calls = 0
        self.expire_token_once = False
        self.calls = []
        self.uploads = {}
        self.files = {}
        self._server = None

    def start(self):
        api = self

        class Handler(_Silent):
            def do_GET(self):
                path = urlsplit(self.path).path
                if path == "/cgi-bin/token":
                    api.token_calls += 1
                    return api._send(self, 200, {"access_token": "TOK%d" % api.token_calls,
                                                 "expires_in": 7200})
                return api._send(self, 404, {"errcode": -1, "errmsg": "nope"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                path = urlsplit(self.path).path
                token = "TOK%d" % api.token_calls
                if path in ("/tcb/databaseadd", "/tcb/databasequery",
                            "/tcb/databaseupdate", "/tcb/uploadfile",
                            "/tcb/batchdownloadfile"):
                    if "access_token=TOK" not in self.path:
                        return api._send(self, 200,
                                         {"errcode": 40001, "errmsg": "bad token"})
                    with api.lock:
                        if api.expire_token_once:
                            api.expire_token_once = False
                            return api._send(self, 200,
                                             {"errcode": 40001, "errmsg": "expired"})
                    api.calls.append((path, raw))
                if path == "/tcb/databaseadd":
                    return api._send(self, 200, {"errcode": 0, "idlist": ["gen1"]})
                if path == "/tcb/databasequery":
                    return api._send(self, 200, {"errcode": 0,
                                                 "data": ['{"_id":"doc1","status":"pending"}'],
                                                 "pager": {"Total": 1}})
                if path == "/tcb/databaseupdate":
                    return api._send(self, 200, {"errcode": 0, "matched": 1, "updated": 1})
                if path == "/tcb/uploadfile":
                    body = json.loads(raw)
                    path_name = body["path"]
                    key = "fid-" + path_name.replace("/", "_")
                    api.uploads[key] = None
                    return api._send(self, 200, {
                        "errcode": 0, "url": "http://127.0.0.1:%d/cos" % api.port,
                        "token": "cos-tok", "authorization": "cos-sig",
                        "file_id": "cloud://" + key})
                if path == "/tcb/batchdownloadfile":
                    body = json.loads(raw)
                    out = []
                    for item in body["file_list"]:
                        key = item["fileid"].replace("cloud://", "")
                        out.append({"fileid": item["fileid"], "status": 0,
                                    "download_url": "http://127.0.0.1:%d/file/%s"
                                                    % (api.port, key)})
                    return api._send(self, 200, {"errcode": 0, "file_list": out})
                return api._send(self, 404, {"errcode": -1, "errmsg": "nope"})

        # extra routes for COS upload + file download
        orig_handler = Handler

        class Handler2(orig_handler):
            def do_POST(self):
                path = urlsplit(self.path).path
                if path == "/cos":
                    length = int(self.headers.get("Content-Length", "0"))
                    body = self.rfile.read(length)
                    boundary = self.headers.get("Content-Type", "").split(
                        "boundary=")[-1].encode("utf-8")
                    name = "?"
                    for segment in body.split(b"--" + boundary):
                        if b'name="key"' in segment:
                            parts = segment.split(b"\r\n\r\n", 1)
                            if len(parts) == 2:
                                name = parts[1].rstrip(b"\r\n").decode("utf-8")
                                break
                    store = "fid-" + name.replace("/", "_")
                    api.uploads[store] = body
                    return api._send(self, 204, b"")
                return orig_handler.do_POST(self)

            def do_GET(self):
                path = urlsplit(self.path).path
                if path.startswith("/file/"):
                    key = path[len("/file/"):]
                    data = api.files.get(key, b"missing:" + key.encode())
                    return api._send(self, 200, data)
                return orig_handler.do_GET(self)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler2)
        server.api = api
        self.port = server.server_address[1]
        self._server = server
        threading.Thread(target=server.serve_forever,
                         kwargs={"poll_interval": 0.02}, daemon=True).start()
        return self

    def stop(self):
        self._server.shutdown()
        self._server.server_close()

    def _send(self, handler, code, payload):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        handler.send_response(code)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)


class FakeCloud:
    """In-memory replacement for WechatCloud with the same interface."""

    def __init__(self):
        self.collections = {}
        self.storage = {}
        self._seq = 0

    def _col(self, name):
        return self.collections.setdefault(name, {})

    def add_doc(self, collection, doc):
        col = self._col(collection)
        doc = dict(doc)
        if not doc.get("_id"):
            self._seq += 1
            doc["_id"] = "id%d" % self._seq
        col[doc["_id"]] = doc
        return doc["_id"]

    def query_docs(self, collection, where=None, order_by=None, limit=20):
        col = self._col(collection)
        docs = []
        for doc in col.values():
            ok = True
            for key, value in (where or {}).items():
                actual = doc.get(key)
                if isinstance(value, dict) and "$in" in value:
                    ok = actual in value["$in"]
                else:
                    ok = actual == value
                if not ok:
                    break
            if ok:
                docs.append(dict(doc))
        if order_by:
            field, direction = order_by
            docs.sort(key=lambda d: d.get(field) if d.get(field) is not None else 0,
                      reverse=(direction == "desc"))
        return docs[:limit]

    def update_doc(self, collection, doc_id, fields):
        col = self._col(collection)
        if doc_id not in col:
            return 0
        col[doc_id].update(fields)
        return 1

    def update_where(self, collection, where, fields):
        col = self._col(collection)
        count = 0
        for doc_id, doc in list(col.items()):
            ok = True
            for key, value in (where or {}).items():
                if isinstance(value, dict) and "$in" in value:
                    ok = doc.get(key) in value["$in"]
                else:
                    ok = doc.get(key) == value
                if not ok:
                    break
            if ok:
                doc.update(fields)
                count += 1
        return count

    def upload_bytes(self, cloud_path, data):
        self._seq += 1
        file_id = "cloud://fake/%d/%s" % (self._seq, cloud_path.rsplit("/", 1)[-1])
        self.storage[file_id] = data
        return file_id

    def download_bytes(self, file_id):
        if file_id not in self.storage:
            raise KeyError(file_id)
        return self.storage[file_id]
