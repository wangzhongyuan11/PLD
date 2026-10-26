# -*- coding: utf-8 -*-
"""WeChat cloud-development server-side API client (standard library only).

Talks to https://api.weixin.qq.com with AppID + AppSecret:
  access_token, cloud database add/query/update, cloud storage upload/download.

The bridge keeps every secret in its local config file; nothing here logs them.
"""
import http.client
import json
import queue
import threading
import time
import uuid
from urllib.parse import urlsplit, quote


class WechatApiError(Exception):
    def __init__(self, message, errcode=None, kind="api"):
        super().__init__(message)
        self.errcode = errcode
        self.kind = kind  # api | network | upload


class WechatCloud:
    """Semantic helpers over the raw tcb/* HTTP endpoints.

    Tests replace this object with an in-memory fake exposing the same
    methods, so the bridge never builds query strings outside this class.
    """

    TOKEN_RETRY_CODES = (40001, 42001)

    def __init__(self, app_id, app_secret, env_id,
                 api_base="https://api.weixin.qq.com", timeout=30):
        self.app_id = app_id
        self.app_secret = app_secret
        self.env_id = env_id
        parts = urlsplit(api_base)
        self.api_scheme = parts.scheme
        self.api_host = parts.hostname
        self.api_port = parts.port or (443 if parts.scheme == "https" else 80)
        self.timeout = timeout
        self._token = None
        self._token_expries = 0.0
        self._token_retry_at = 0.0
        self._token_error = None
        self._token_lock = threading.RLock()
        self._api_pool = queue.LifoQueue(maxsize=4)

    # ---------------- http plumbing ----------------

    def _request(self, scheme, host, port, method, path, body=None,
                 headers=None, timeout=None):
        conn_cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        pooled = (scheme, host, port) == (self.api_scheme, self.api_host, self.api_port)
        conn = None
        if pooled:
            try:
                conn = self._api_pool.get_nowait()
            except queue.Empty:
                pass
        if conn is None:
            conn = conn_cls(host, port, timeout=timeout or self.timeout)
        conn.timeout = timeout or self.timeout
        if conn.sock is not None:
            conn.sock.settimeout(conn.timeout)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
            data = resp.read()
            if pooled and not resp.will_close:
                try:
                    self._api_pool.put_nowait(conn)
                    conn = None
                except queue.Full:
                    pass
            return resp.status, dict(resp.getheaders()), data
        finally:
            if conn is not None:
                conn.close()

    def close(self):
        while True:
            try:
                self._api_pool.get_nowait().close()
            except queue.Empty:
                return

    @staticmethod
    def _raise_network(exc):
        raise WechatApiError("微信服务端网络错误：%s" % exc, kind="network") from exc

    def _api_post_json(self, path, payload, params=None, retry=True):
        params = dict(params or {})
        if "access_token" in params and not params["access_token"]:
            del params["access_token"]
        query = "&".join("%s=%s" % (k, quote(str(v), safe="")) for k, v in params.items())
        url = path + ("?" + query if query else "")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json",
                   "Content-Length": str(len(body))}
        try:
            status, _, data = self._request(self.api_scheme, self.api_host,
                                            self.api_port, "POST", url, body, headers)
        except (OSError, http.client.HTTPException) as exc:
            self._raise_network(exc)
        if status != 200:
            raise WechatApiError("微信服务端 HTTP %d" % status, kind="network")
        result = json.loads(data.decode("utf-8"))
        errcode = result.get("errcode", 0)
        if errcode:
            if retry and errcode in self.TOKEN_RETRY_CODES:
                with self._token_lock:
                    if self._token == params.get("access_token"):
                        self._token = None
                if "access_token" in params:
                    params["access_token"] = self.access_token()
                return self._api_post_json(path, payload, params, retry=False)
            raise WechatApiError("微信接口错误 %s：%s" % (errcode, result.get("errmsg", "")),
                                 errcode=errcode)
        return result

    # ---------------- access token ----------------

    def access_token(self, force=False):
        with self._token_lock:
            return self._access_token_locked(force)

    def _access_token_locked(self, force=False):
        now = time.time()
        if not force and self._token and now < self._token_expries - 120:
            return self._token
        # Back off after real fetch failures so a wrong secret cannot hammer
        # the live /cgi-bin/token endpoint (and its daily quota) every loop.
        if now < self._token_retry_at:
            raise WechatApiError("access_token 获取退避中（%.0fs 后重试）：%s"
                                 % (self._token_retry_at - now, self._token_error))
        try:
            return self._fetch_token(now)
        except WechatApiError as exc:
            self._token_retry_at = now + 30
            self._token_error = str(exc)
            raise

    def _fetch_token(self, now):
        try:
            status, _, data = self._request(
                self.api_scheme, self.api_host, self.api_port, "GET",
                "/cgi-bin/token?grant_type=client_credential&appid=%s&secret=%s"
                % (quote(self.app_id, safe=""), quote(self.app_secret, safe="")))
        except (OSError, http.client.HTTPException) as exc:
            self._raise_network(exc)
        if status != 200:
            raise WechatApiError("获取 access_token HTTP %d" % status, kind="network")
        result = json.loads(data.decode("utf-8"))
        if "access_token" not in result:
            raise WechatApiError("获取 access_token 失败：%s" % result.get("errmsg", result),
                                 errcode=result.get("errcode"))
        self._token = result["access_token"]
        self._token_expries = now + int(result.get("expires_in", 7200))
        return self._token

    # ---------------- query-string builders ----------------

    @staticmethod
    def _value_literal(value):
        # {'$in': [...]} is the only command form the bridge needs.
        if isinstance(value, dict) and set(value) == {"$in"}:
            return "db.command.in(%s)" % json.dumps(value["$in"], ensure_ascii=False,
                                                    separators=(",", ":"))
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _where_clause(cls, where):
        if not where:
            return ""
        parts = ["%s:%s" % (json.dumps(k), cls._value_literal(v)) for k, v in where.items()]
        return ".where({%s})" % ",".join(parts)

    # ---------------- database ----------------

    def add_doc(self, collection, doc):
        token = self.access_token()
        result = self._api_post_json(
            "/tcb/databaseadd",
            {"env": self.env_id,
             "query": 'db.collection(%s).add({data:%s})'
                      % (json.dumps(collection), json.dumps(doc, ensure_ascii=False,
                                                             separators=(",", ":")))},
            params={"access_token": token})
        # 实测（2026-10-25，env pld-d7gzvyxp10bdc9437）：databaseadd 返回 id_list
        ids = result.get("id_list") or result.get("idlist") or []
        return doc.get("_id") or (ids[0] if ids else None)

    def query_docs(self, collection, where=None, order_by=None, limit=20):
        token = self.access_token()
        query = "db.collection(%s)%s" % (json.dumps(collection),
                                         self._where_clause(where or {}))
        if order_by:
            query += '.orderBy(%s,%s)' % (json.dumps(order_by[0]), json.dumps(order_by[1]))
        query += ".limit(%d).get()" % int(limit)
        result = self._api_post_json("/tcb/databasequery",
                                     {"env": self.env_id, "query": query},
                                     params={"access_token": token})
        docs = []
        for row in result.get("data") or []:
            docs.append(json.loads(row) if isinstance(row, str) else row)
        return docs

    def update_doc(self, collection, doc_id, fields):
        token = self.access_token()
        result = self._api_post_json(
            "/tcb/databaseupdate",
            {"env": self.env_id,
             "query": 'db.collection(%s).doc(%s).update({data:%s})'
                      % (json.dumps(collection), json.dumps(doc_id),
                         json.dumps(fields, ensure_ascii=False, separators=(",", ":")))},
            params={"access_token": token})
        # 实测（2026-10-25）：databaseupdate 返回 matched/modified，没有 updated 字段；
        # 误读会让 claim 永远返回 0，任务被置 running 却从不执行。
        return int(result.get("matched") or result.get("modified") or 0)

    def update_where(self, collection, where, fields):
        token = self.access_token()
        result = self._api_post_json(
            "/tcb/databaseupdate",
            {"env": self.env_id,
             "query": 'db.collection(%s)%s.update({data:%s})'
                      % (json.dumps(collection), self._where_clause(where),
                         json.dumps(fields, ensure_ascii=False, separators=(",", ":")))},
            params={"access_token": token})
        # 实测（2026-10-25）：databaseupdate 返回 matched/modified，没有 updated 字段；
        # 误读会让 claim 永远返回 0，任务被置 running 却从不执行。
        return int(result.get("matched") or result.get("modified") or 0)

    # ---------------- storage ----------------

    def upload_bytes(self, cloud_path, data):
        token = self.access_token()
        presign = self._api_post_json("/tcb/uploadfile",
                                      {"env": self.env_id, "path": cloud_path},
                                      params={"access_token": token})
        url = presign["url"]
        boundary = "----pldbridge" + uuid.uuid4().hex
        fields = [("key", cloud_path),
                  ("Signature", presign["authorization"]),
                  ("x-cos-security-token", presign["token"])]
        filename = cloud_path.rsplit("/", 1)[-1]
        parts = []
        for name, value in fields:
            parts.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s"
                          % (boundary, name, value)).encode("utf-8"))
        parts.append(("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n"
                      "Content-Type: application/octet-stream\r\n\r\n"
                      % (boundary, filename)).encode("utf-8"))
        # multipart 规范：文件段头部以 \r\n\r\n 结束后就是文件字节本身。
        # 这里不能再补 \r\n，否则它会进文件内容（实测 COS 存储被前置 \r\n 污染）。
        body = b"\r\n".join(parts) + data + b"\r\n--%s--\r\n" % boundary.encode()
        target = urlsplit(url)
        port = target.port or (443 if target.scheme == "https" else 80)
        path = target.path + ("?" + target.query if target.query else "")
        try:
            status, _, _ = self._request(target.scheme, target.hostname, port,
                                         "POST", path, body,
                                         {"Content-Type": "multipart/form-data; boundary=%s" % boundary},
                                         timeout=max(self.timeout, 120))
        except (OSError, http.client.HTTPException) as exc:
            self._raise_network(exc)
        if status not in (200, 204):
            raise WechatApiError("云存储上传失败 HTTP %d" % status, kind="upload")
        return presign["file_id"]

    def download_urls(self, file_ids, max_age=7200):
        token = self.access_token()
        result = self._api_post_json(
            "/tcb/batchdownloadfile",
            {"env": self.env_id,
             "file_list": [{"fileid": fid, "max_age": int(max_age)} for fid in file_ids]},
            params={"access_token": token})
        urls = {}
        for item in result.get("file_list") or []:
            if item.get("status") == 0:
                urls[item["fileid"]] = item["download_url"]
        missing = [f for f in file_ids if f not in urls]
        if missing:
            raise WechatApiError("云存储下载地址获取失败：%s" % ", ".join(missing[:3]))
        return urls

    def download_bytes(self, file_id, max_age=7200):
        url = self.download_urls([file_id], max_age)[file_id]
        target = urlsplit(url)
        port = target.port or (443 if target.scheme == "https" else 80)
        path = target.path + ("?" + target.query if target.query else "")
        try:
            status, _, data = self._request(target.scheme, target.hostname, port,
                                            "GET", path, None, {},
                                            timeout=max(self.timeout, 120))
        except (OSError, http.client.HTTPException) as exc:
            self._raise_network(exc)
        if status != 200:
            raise WechatApiError("云存储下载失败 HTTP %d" % status, kind="upload")
        return data
