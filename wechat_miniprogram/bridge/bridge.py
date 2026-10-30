# -*- coding: utf-8 -*-
"""Bridge between WeChat cloud development and the board's three models.

Polls cloud DB for switch commands and inference tasks, drives the board's
existing HTTP services (entry 8088 / hub 8765 / acdc 8092) exactly like a
browser would, then uploads results back to cloud storage.

Design guarantees:
  - HTTP client towards the board; no local listening port or direct FPGA access
  - task-level auto_display=false; inference still uses the existing HDMI handoff
  - one task at a time (the board itself is single-task serial)
  - waiting_model tasks run only after the matching model is active
"""
import argparse
import csv
import io
import json
import logging
import logging.handlers
import sys
import threading
import time
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from wechat_api import WechatCloud, WechatApiError
from board_client import (BoardClient, BoardError, SKIN_VIEW_LABELS,
                          CAMUS_FILES, ACDC_MPR_FILES, ACDC_DONE_STATUS,
                          ACDC_BAD_STATUS)
import preprocess

log = logging.getLogger("bridge")

BOARD_DOC_ID = "board_current"
TASK_MODELS = ("skin", "camus", "acdc")
PROGRESS_WRITE_INTERVAL = 2.5
HEARTBEAT_FORCE_INTERVAL = 20.0


class TaskFailure(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class WaitingModel(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class WaitingBoard(WaitingModel):
    """Transient board contention: keep queued without requiring a model switch."""


class Bridge:
    def __init__(self, cloud, board, cfg, sleep=time.sleep, clock=time.monotonic):
        self.cloud = cloud
        self.board = board
        self.cfg = cfg
        self.sleep = sleep
        self.clock = clock
        self.poll_interval = max(0.05, cfg.get("poll_interval_ms", 1200) / 1000.0)
        self.heartbeat_interval = max(0.5, cfg.get("heartbeat_interval_ms", 5000) / 1000.0)
        self.switch_timeout = cfg.get("switch_timeout_s", 300)
        self.task_timeouts = {"skin": 900, "camus": 900, "acdc": 1800}
        self.task_timeouts.update(cfg.get("task_timeouts") or {})
        self._state_lock = threading.Lock()
        self._board_doc = {"written": None, "written_at": 0.0}
        self._stop = threading.Event()
        self._last_progress_write = 0.0
        self._commands_skip_until = 0.0
        self.result_workers = max(1, min(3, int(cfg.get("result_workers", 2))))
        self.async_results = cfg.get("async_results", True)
        self._result_executor = None
        self._transfers = {}
        self._session = uuid.uuid4().hex
        self._uncertain_claim = None

    # ---------------- board heartbeat ----------------

    def read_board_state(self):
        """Merge entry (preferred) and hub status into one board snapshot."""
        state = {"online": False, "model": None, "phase": None,
                 "source": None, "error": None}
        entry = None
        offline_reason = None
        try:
            entry = self.board.entry_status()
        except BoardError as exc:
            offline_reason = str(exc)
        if entry is not None:
            state.update(online=True, source="entry",
                         model=entry.get("model"), phase=entry.get("phase"),
                         stage=entry.get("stage") or None,
                         error=entry.get("error") or None)
            state["boot"] = entry.get("boot") or {}
            return state
        # entry absent or unreachable: fall back to the dual-model hub
        try:
            status = self.board.hub_status()
        except BoardError:
            if offline_reason:
                state["error"] = offline_reason
            return state
        state.update(online=True, source="hub", model=status.get("model"),
                     phase=status.get("phase"),
                     error=status.get("error") or None)
        return state

    def heartbeat_once(self, force=False):
        state = self.read_board_state()
        payload = dict(state)
        with self._state_lock:
            self.board_state = state
        changed = state != self._board_doc.get("written")
        if not (force or changed or
                time.time() - self._board_doc["written_at"] > HEARTBEAT_FORCE_INTERVAL):
            return state
        try:
            payload["updated_at"] = int(time.time() * 1000)
            updated = self.cloud.update_doc("board", BOARD_DOC_ID, payload)
            if not updated:
                doc = dict(payload)
                doc["_id"] = BOARD_DOC_ID
                try:
                    self.cloud.add_doc("board", doc)
                except WechatApiError as exc:
                    if "already exist" not in str(exc) and "DUPLICATE" not in str(exc).upper():
                        raise
            self._board_doc["written"] = state
            self._board_doc["written_at"] = time.time()
        except WechatApiError as exc:
            log.warning("board 心跳写入失败：%s", exc)
        return state

    def current_model(self):
        with self._state_lock:
            state = getattr(self, "board_state", None)
        return state or {"online": False, "model": None, "source": None}

    # ---------------- stale-task recovery ----------------

    def recover_stale_tasks(self, force=False):
        """Fail tasks left 'running' by a previous bridge crash.

        Throttled: the query runs at most once per minute, not every poll,
        to stay far below cloud-API rate limits."""
        now = self.clock()
        if not force and now - getattr(self, "_last_recover", 0.0) < 60.0:
            return
        self._last_recover = now
        try:
            docs = self.cloud.query_docs("tasks", {"status": "running"}, None, 50)
        except WechatApiError as exc:
            log.warning("恢复检查查询失败：%s", exc)
            return
        now_ms = int(time.time() * 1000)
        for doc in docs:
            if doc.get("_id") in self._transfers:
                continue
            timeout_ms = self.task_timeouts.get(doc.get("type"), 900) * 1000
            started = doc.get("started_at") or doc.get("created_at") or now_ms
            # This is a single bridge installation. On startup, attach to an
            # accepted board job instead of submitting the input a second time.
            if force and doc.get("board_job") and doc.get("bridge_session") != self._session:
                self.cloud.update_doc("tasks", doc["_id"], {
                    "status": "pending", "resume_job": doc["board_job"],
                    "updated_at": now_ms})
                continue
            if now_ms - started > timeout_ms + 300000:
                self._fail_task(doc, "桥接程序中断，任务未完成；请重新提交")
                log.info("恢复清理过期 running 任务 %s", doc.get("_id"))
        try:
            commands = self.cloud.query_docs("commands", {"status": "running"}, None, 50)
            for command in commands:
                started = command.get("started_at") or command.get("created_at") or now_ms
                if now_ms - started > (self.switch_timeout + 300) * 1000:
                    self.cloud.update_doc("commands", command["_id"],
                                          {"status": "error", "error": "桥接中断，切换未确认；请检查当前模型后重试",
                                           "finished_at": now_ms})
        except WechatApiError as exc:
            log.warning("切换恢复检查失败：%s", exc)

    # ---------------- switch commands ----------------

    def process_commands(self):
        if self.clock() < self._commands_skip_until:
            return None
        try:
            docs = self.cloud.query_docs("commands", {"status": "pending"},
                                         ("created_at", "asc"), 1)
        except WechatApiError as exc:
            if getattr(exc, "errcode", None) == -501009:
                # commands 集合还没在控制台创建：退避 60 秒，避免每轮空查刷日志、耗配额
                self._commands_skip_until = self.clock() + 60
                log.warning("commands 集合不存在，60 秒后重试（请到云开发控制台创建，注意拼写 commands）")
            else:
                log.warning("命令查询失败：%s", exc)
            return None
        if not docs:
            return None
        cmd = docs[0]
        model = cmd.get("model")
        updated = self.cloud.update_where(
            "commands", {"_id": cmd["_id"], "status": "pending"},
            {"status": "running", "started_at": int(time.time() * 1000)})
        if not updated:
            return None
        log.info("处理切换命令 %s -> %s", cmd.get("_id"), model)
        try:
            if model not in TASK_MODELS:
                raise TaskFailure("未知模型 %s" % model)
            try:
                source = self.read_board_state().get("source")
                if source == "hub":
                    if model == "acdc":
                        raise TaskFailure("板卡未部署三模型入口，无法切换 ACDC")
                    self.board.hub_switch(model)
                else:
                    self.board.entry_switch(model)
            except BoardError as exc:
                if exc.kind == "offline":
                    raise TaskFailure("板卡不可达：%s" % exc)
                raise
            deadline = self.clock() + self.switch_timeout
            wait_ready = self.board.wait_hub_ready if source == "hub" else self.board.wait_entry_ready
            wait_ready(model, deadline, interval=max(0.2, self.poll_interval),
                       sleep=self.sleep, clock=self.clock)
            self.board.invalidate_hub()
            self.cloud.update_doc("commands", cmd["_id"],
                                  {"status": "done",
                                   "finished_at": int(time.time() * 1000)})
            log.info("切换完成 -> %s", model)
            self.heartbeat_once(force=True)
            return "done"
        except (TaskFailure, BoardError) as exc:
            message = str(exc)
            log.warning("切换命令失败：%s", message)
            self.cloud.update_doc("commands", cmd["_id"],
                                  {"status": "error", "error": message,
                                   "finished_at": int(time.time() * 1000)})
            return "error"

    # ---------------- task orchestration ----------------

    def pick_task(self):
        self._transfers = {key: future for key, future in self._transfers.items() if not future.done()}
        if len(self._transfers) >= 2:
            return None  # bounded buffering: at most two result sets await cloud upload
        board = self.current_model()
        if not board.get("online") or board.get("phase") != "ready":
            return None
        board_model = board.get("model")
        if board_model not in TASK_MODELS:
            return None
        try:
            docs = self.cloud.query_docs(
                "tasks", {"status": {"$in": ["pending", "waiting_model"]}, "type": board_model},
                ("created_at", "asc"), 5)
            if board.get("source") == "hub":
                unsupported = self.cloud.query_docs(
                    "tasks", {"status": {"$in": ["pending", "waiting_model"]}, "type": "acdc"},
                    ("created_at", "asc"), 1)
                for task in unsupported:
                    self._fail_task(task, "板卡未部署三模型入口，无法运行 ACDC 任务")
        except WechatApiError as exc:
            log.warning("任务查询失败：%s", exc)
            return None
        for doc in docs:
            kind = doc.get("type")
            if kind not in TASK_MODELS:
                self._fail_task(doc, "未知任务类型 %s" % kind)
                return None
            if kind == "acdc" and board.get("source") != "entry":
                self._fail_task(doc, "板卡未部署三模型入口，无法运行 ACDC 任务")
                return None
            # Skip while a different model owns the FPGA; the mini program
            # shows a "switch model" hint from the board heartbeat.
            if board_model is not None and board_model != kind:
                continue
            return doc
        return None

    def claim_task(self, doc):
        from_status = doc.get("status")
        claim = uuid.uuid4().hex
        fields = {"status": "running", "claim_id": claim,
                  "bridge_session": self._session, "started_at": int(time.time() * 1000)}
        try:
            return bool(self.cloud.update_where(
                "tasks", {"_id": doc["_id"], "status": from_status}, fields))
        except WechatApiError:
            self._uncertain_claim = dict(doc, claim_id=claim)
            # Never replay a conditional claim after losing its acknowledgement.
            rows = self.cloud.query_docs("tasks", {"_id": doc["_id"]}, None, 1)
            if rows and rows[0].get("claim_id") == claim:
                self._uncertain_claim = None
                return True
            self._uncertain_claim = None
            raise

    def process_task(self):
        if self._uncertain_claim:
            old = self._uncertain_claim
            rows = self.cloud.query_docs("tasks", {"_id": old["_id"]}, None, 1)
            doc = rows[0] if rows and rows[0].get("status") == "running" and rows[0].get("claim_id") == old["claim_id"] else None
            self._uncertain_claim = None
            claimed = doc is not None
        else:
            doc = self.pick_task()
            claimed = False
        if doc is None:
            return None
        if not claimed and not self.claim_task(doc):
            return None
        if "result" in doc and doc.get("result") is None:
            # 旧版小程序创建的任务带 result:null，嵌套结果回传必失败且无法原地
            # 修复（TCB 不能在 null 字段下建子字段）。直接判错，不浪费板卡算力。
            self._fail_task(doc, "任务由旧版小程序创建，结果字段与回传不兼容；请更新小程序后重新提交")
            log.warning("旧版任务文档（result=null）直接判错 %s", doc.get("_id"))
            return "error"
        kind = doc["type"]
        log.info("开始任务 %s（%s）", doc.get("_id"), kind)
        doc["_began"] = self.clock()
        self._last_progress_write = 0.0
        try:
            handler = getattr(self, "_run_%s" % kind)
            handler(doc)
            if doc["_id"] in self._transfers:
                return "uploading"
            log.info("任务完成 %s", doc.get("_id"))
            return "done"
        except WaitingModel as exc:
            self.cloud.update_doc("tasks", doc["_id"],
                                  {"status": "pending" if isinstance(exc, WaitingBoard) else "waiting_model",
                                   "note": exc.message,
                                   "updated_at": int(time.time() * 1000)})
            log.info("任务 %s 等待模型：%s", doc.get("_id"), exc.message)
            return "waiting"
        except TaskFailure as exc:
            self._fail_task(doc, exc.message)
            log.warning("任务失败 %s：%s", doc.get("_id"), exc.message)
            return "error"
        except (BoardError, WechatApiError) as exc:
            self._fail_task(doc, str(exc))
            log.warning("任务异常 %s：%s", doc.get("_id"), exc)
            return "error"
        except Exception as exc:  # 未预料的异常也必须落 error，否则任务永远停在 running
            self._fail_task(doc, "任务异常：%s" % exc)
            log.exception("任务未捕获异常 %s", doc.get("_id"))
            return "error"

    def _fail_task(self, doc, message):
        try:
            fields = {"status": "error", "error": str(message)[:500],
                      "updated_at": int(time.time() * 1000)}
            if doc.get("board_job"):
                fields["board_job"] = doc["board_job"]
            self.cloud.update_doc("tasks", doc["_id"],
                                  fields)
        except WechatApiError:
            pass

    def _write_progress(self, task_id, stage, detail=None, percent=None):
        now = self.clock()
        if now - self._last_progress_write < PROGRESS_WRITE_INTERVAL:
            return
        self._last_progress_write = now
        payload = {"stage": str(stage)[:60], "updated_at": int(time.time() * 1000)}
        if detail:
            payload["detail"] = str(detail)[:120]
        if percent is not None:
            payload["percent"] = max(0, min(100, int(percent)))
        try:
            self.cloud.update_doc("tasks", task_id, {"progress": payload})
        except WechatApiError:
            pass

    def _download_inputs(self, task):
        inputs = task.get("inputs") or []
        if not inputs:
            raise TaskFailure("任务缺少输入文件")
        blobs = []
        for item in inputs[:4]:
            file_id = item.get("fileID") or item.get("fileid")
            name = item.get("name") or "input.bin"
            self._write_progress(task["_id"], "下载输入", name)
            try:
                data = self.cloud.download_bytes(file_id)
            except WechatApiError as exc:
                raise TaskFailure("输入文件下载失败：%s" % exc)
            blobs.append((name, data))
        return blobs

    def _submit_with_retry(self, task_id, submit):
        deadline = self.clock() + 60
        last = None
        while True:
            try:
                return submit()
            except BoardError as exc:
                if exc.kind in ("paused", "stale_generation"):
                    raise WaitingModel("板卡当前未运行该模型，请先在小程序首页切换")
                if exc.kind == "busy":
                    last = exc
                    self._write_progress(task_id, "板卡忙，排队等待", str(exc))
                    if self.clock() >= deadline:
                        raise WaitingBoard("板卡持续忙碌，稍后自动重试：%s" % last)
                    self.sleep(self.poll_interval)
                    continue
                if exc.kind == "offline":
                    self._write_progress(task_id, "板卡连接中断，重试", str(exc))
                    if self.clock() >= deadline:
                        raise TaskFailure("板卡不可达：%s" % exc)
                    self.sleep(self.poll_interval)
                    continue
                raise TaskFailure("板端提交失败：%s" % exc)

    def _poll_with_retry(self, task_id, poll, is_done, is_failed, error_of,
                         deadline, progress_of=None, on_update=None):
        offline_since = None
        while True:
            try:
                job = poll()
                offline_since = None
            except BoardError as exc:
                if exc.kind != "offline":
                    raise TaskFailure("板端轮询失败：%s" % exc)
                offline_since = offline_since or self.clock()
                if self.clock() - offline_since > 60:
                    raise TaskFailure("板卡在任务执行中失联：%s" % exc)
                self.sleep(self.poll_interval)
                continue
            if progress_of:
                stage, percent = progress_of(job)
                self._write_progress(task_id, stage, None, percent)
            if on_update and not is_failed(job):
                on_update(job)
            if is_done(job):
                return job
            if is_failed(job):
                raise TaskFailure("板端任务失败：%s" % (error_of(job) or "未知错误"))
            if self.clock() >= deadline:
                raise TaskFailure("任务超时（板端未在限定时间内完成）")
            self.sleep(min(.3, self.poll_interval))

    def _finish_task(self, task, items, metrics, extra=None):
        if self.async_results:
            if self._result_executor is None:
                self._result_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="results")
            self._transfers[task["_id"]] = self._result_executor.submit(
                self._upload_result, task, items, metrics, extra)
        else:
            self._publish_result(task, items, metrics, extra)

    def _upload_result(self, task, items, metrics, extra):
        try:
            self._publish_result(task, items, metrics, extra)
            log.info("任务完成 %s", task["_id"])
        except Exception as exc:
            self._fail_task(task, "结果回传失败，可重传结果：%s" % exc)
            log.warning("结果回传失败 %s：%s", task["_id"], exc)

    def _publish_result(self, task, items, metrics, extra=None):
        upload_started = self.clock()
        key = task.get("key") or task["_id"]
        previous = task.get("result") or {}
        uploaded = {row["name"]: row["fileID"] for row in
                    previous.get("views", []) + previous.get("files", [])}
        if previous.get("zip"):
            for name, _, _ in items:
                if name.endswith(".zip"):
                    uploaded[name] = previous["zip"]
        if previous.get("report"):
            uploaded["report.json"] = previous["report"]
        result = {"views": [], "files": [], "metrics": metrics, "zip": None, "report": None,
                  "previews": previous.get("previews") or []}
        if extra:
            result.update(extra)

        def publish(done=False):
            result["views"] = [{"name": name, "label": label, "fileID": uploaded[name]}
                               for name, label, _ in items if name.endswith(".png") and name in uploaded]
            result["files"] = [{"name": name, "label": label, "fileID": uploaded[name]}
                               for name, label, _ in items if not name.endswith((".png", ".zip"))
                               and name != "report.json" and name in uploaded]
            result["report"] = uploaded.get("report.json")
            result["zip"] = next((uploaded[name] for name, _, _ in items
                                   if name.endswith(".zip") and name in uploaded), None)
            fields = {"status": "done" if done else "running", "result": result,
                      "error": None, "updated_at": int(time.time() * 1000),
                      "progress": {"stage": "已完成" if done else "回传结果，预览陆续就绪",
                                   "percent": 100 if done else 95}}
            if done:
                fields["finished_at"] = fields["updated_at"]
            self.cloud.update_doc("tasks", task["_id"], fields)

        # Publish each available image before larger exports finish. PNGs are
        # submitted first; result order still follows the board's original list.
        pending = sorted((item for item in items if item[0] not in uploaded),
                         key=lambda item: not item[0].endswith(".png"))
        with ThreadPoolExecutor(max_workers=self.result_workers) as pool:
            futures = {pool.submit(self.cloud.upload_bytes, "tasks/%s/out/%s" % (key, name), data): name
                       for name, _, data in pending}
            error = None
            last_publish = None
            for future in as_completed(futures):
                try:
                    uploaded[futures[future]] = future.result()
                    if last_publish is None or self.clock() - last_publish >= PROGRESS_WRITE_INTERVAL:
                        publish()
                        if last_publish is None and result["views"]:
                            log.info("首批预览计时 %s upload_to_preview_seconds=%.3f",
                                     task["_id"], self.clock() - upload_started)
                        last_publish = self.clock()
                except Exception as exc:
                    error = error or exc
            if error:
                publish()  # checkpoint successful files even if another upload failed
                raise error
        publish(done=True)
        log.info("结果回传计时 %s upload_seconds=%.3f workers=%d",
                 task["_id"], self.clock() - upload_started, self.result_workers)
        began = task.get("_began")
        if began is not None:
            log.info("任务计时 %s bridge_seconds=%.3f", task["_id"], self.clock() - began)

    def _resume_id(self, task):
        resume = task.get("resume_job") or {}
        if resume.get("model") == task["type"]:
            return resume.get("id")
        return None

    def _completed(self, task, job_id):
        task["board_job"] = {"model": task["type"], "id": job_id}
        task["board_completed"] = True
        self.cloud.update_doc(
            "tasks", task["_id"],
            {"board_job": task["board_job"], "board_completed": True,
             "updated_at": int(time.time() * 1000)})

    def _accepted(self, task, job_id):
        if not job_id:
            raise TaskFailure("板端未返回任务编号，请先检查板端状态")
        job = {"model": task["type"], "id": job_id}
        task["board_job"] = job
        # Idempotent checkpoint: do not submit again if only this write fails.
        fields = {"board_job": job, "board_completed": False,
                  "updated_at": int(time.time()*1000)}
        try:
            self.cloud.update_doc("tasks", task["_id"], fields)
        except WechatApiError:
            self.cloud.update_doc("tasks", task["_id"], fields)

    def _early_preview(self, task, model, job_id, job):
        if task.get("_preview_done") or self.clock() < task.get("_preview_retry", 0):
            return
        done = job.get("state") == "done" or job.get("status") == ACDC_DONE_STATUS
        if not (done or job.get("preview_ready") or job.get("quick_preview_ready")):
            return
        try:
            if model == "skin":
                # Public preview aliases can contain JPEG bytes despite .png.
                views = job.get("preview_views") or []
                view = next((v for v in ("outlined", "comparison", "mask", "original") if v in views), None)
                if not view and not done:
                    return
                view = view or next(iter(job.get("views") or []), "outlined")
                raw = self.board.skin_file(job_id, ("" if done else "preview_") + view + ".png")
                label = "ROI 分割预览" if (task.get("params") or {}).get("roi") else "分割预览"
            elif model == "camus":
                raw = self.board.camus_file(job_id, "overlay.png")
                label = "CAMUS 分割预览"
            else:
                quick = bool(job.get("quick_preview_ready") and not job.get("preview_ready"))
                raw = self.board.acdc_preview(job_id, quick=quick)
                label = "ED 中心切片快速预览" if quick else "ED 轴位预览"
            from PIL import Image
            with Image.open(io.BytesIO(raw)) as image:
                image.thumbnail((768, 768))
                out = io.BytesIO()
                image.convert("RGB").save(out, "JPEG", quality=80)
            key = task.get("key") or task["_id"]
            fid = self.cloud.upload_bytes("tasks/%s/out/mobile_preview.jpg" % key, out.getvalue())
            result = dict(task.get("result") or {})
            result["previews"] = [{"name": "mobile_preview.jpg", "label": label, "fileID": fid}]
            self.cloud.update_doc("tasks", task["_id"], {"result": result,
                "updated_at": int(time.time()*1000)})
            task["result"] = result
            task["_preview_done"] = True
            log.info("手机首张预览 %s bridge_to_preview_seconds=%.3f bytes=%d",
                     task["_id"], self.clock()-task.get("_began", self.clock()), len(out.getvalue()))
        except (BoardError, WechatApiError, OSError, ValueError) as exc:
            task["_preview_retry"] = self.clock() + 10
            log.warning("预览暂不可用，继续等待完整结果 %s：%s", task["_id"], exc)

    def _download_results(self, files, read, task_id):
        began = self.clock()
        with ThreadPoolExecutor(max_workers=self.result_workers) as pool:
            items = list(pool.map(lambda item: (item[0], item[1], read(item[0])), files))
        log.info("结果读取计时 %s board_download_seconds=%.3f bytes=%d",
                 task_id, self.clock() - began, sum(len(data) for _, _, data in items))
        return items

    # ---------------- skin ----------------

    def _run_skin(self, task):
        job_id = self._resume_id(task)
        if not job_id:
            name, raw = self._download_inputs(task)[0]
            self._write_progress(task["_id"], "准备图片", name)
            jpg = preprocess.prepare_skin_image(raw, (task.get("params") or {}).get("roi"))
            thickness = (task.get("params") or {}).get("thickness") or 2
            if (task.get("params") or {}).get("roi"):
                name = "ROI_" + name
            job = self._submit_with_retry(task["_id"], lambda: self.board.skin_submit(
                jpg, thickness=thickness, name=name))
            job_id = job.get("id")
            self._accepted(task, job_id)
        deadline = self.clock() + self.task_timeouts["skin"]
        job = self._poll_with_retry(
            task["_id"],
            lambda: self.board.skin_job(job_id),
            lambda j: j.get("state") == "done",
            lambda j: j.get("state") == "error",
            lambda j: j.get("error"),
            deadline,
            progress_of=lambda j: (j.get("stage"), j.get("progress")),
            on_update=lambda j: self._early_preview(task, "skin", job_id, j))
        views = job.get("views") or []
        self._completed(task, job_id)
        files = [("%s.png" % v, SKIN_VIEW_LABELS.get(v, v)) for v in views]
        files.extend([("report.json", "结果报告"), ("results.zip", "结果包")])
        items = self._download_results(files, lambda name: self.board.skin_file(job_id, name), task["_id"])
        stats = job.get("statistics") or {}
        metrics = []
        if "area_percent" in stats:
            metrics.append({"label": "ROI 内病变面积占比" if (task.get("params") or {}).get("roi") else "病变面积占比",
                            "value": "%.2f%%" % float(stats["area_percent"])})
        if "foreground_pixels" in stats:
            metrics.append({"label": "前景像素",
                            "value": "%d" % int(stats["foreground_pixels"])})
        if "image_pixels" in stats:
            metrics.append({"label": "有效区域像素",
                            "value": "%d" % int(stats["image_pixels"])})
        if stats.get("b_seconds") is not None:
            metrics.append({"label": "板端推理 B 耗时",
                            "value": "%.3f s" % float(stats["b_seconds"])})
        if stats.get("job_total_seconds") is not None:
            metrics.append({"label": "任务总耗时",
                            "value": "%.2f s" % float(stats["job_total_seconds"])})
        self._finish_task(task, items, metrics)

    # ---------------- camus ----------------

    def _run_camus(self, task):
        job_id = self._resume_id(task)
        if not job_id:
            blobs = self._download_inputs(task)
            params = task.get("params") or {}
            input_name = params.get("input_name") or blobs[0][0]
            allowed = (".png", ".jpg", ".jpeg", ".npy", ".npz", ".mhd", ".nii", ".gz")
            if not input_name.lower().endswith(allowed):
                raise TaskFailure("CAMUS 输入扩展名不受支持：%s" % input_name)
            if sum(len(d) for _, d in blobs) > 20 * 1024 * 1024:
                raise TaskFailure("CAMUS 输入总体积过大")
            job = self._submit_with_retry(task["_id"], lambda: self.board.camus_submit(
                blobs, input_name, params.get("spacing")))
            job_id = job.get("id")
            self._accepted(task, job_id)
        deadline = self.clock() + self.task_timeouts["camus"]
        job = self._poll_with_retry(
            task["_id"],
            lambda: self.board.camus_job(job_id),
            lambda j: j.get("state") == "done",
            lambda j: j.get("state") == "error",
            lambda j: j.get("error") or j.get("close_error"),
            deadline,
            progress_of=lambda j: ("CAMUS 推理中" if j.get("state") == "running"
                                   else "排队中", None),
            on_update=lambda j: self._early_preview(task, "camus", job_id, j))
        self._completed(task, job_id)
        items = self._download_results(CAMUS_FILES, lambda name: self.board.camus_file(job_id, name), task["_id"])
        report = next(data for name, _, data in items if name == "report.json")
        metrics = []
        if report:
            try:
                parsed = json.loads(report.decode("utf-8"))
                output = parsed.get("output") or {}
                names = output.get("class_names") or []
                areas = output.get("areas") or {}
                for i, cls in enumerate(names):
                    if cls == "background":
                        continue
                    if isinstance(areas, dict):
                        area = areas.get(cls) or {}
                        if area.get("area_mm2") is not None:
                            value = "%.2f mm²" % float(area["area_mm2"])
                        elif area.get("pixels") is not None:
                            value = "%d 像素" % int(area["pixels"])
                        else:
                            continue
                    elif i < len(areas):  # tolerate earlier bridge demo fixtures
                        value = "%.2f" % float(areas[i])
                    else:
                        continue
                    metrics.append({"label": "面积 %s" % cls, "value": value})
                result_info = job.get("result") or {}
                if result_info.get("end_to_end_seconds") is not None:
                    metrics.append({"label": "板端处理耗时",
                                    "value": "%.2f s" % float(result_info["end_to_end_seconds"])})
            except (ValueError, TypeError):
                pass
        self._finish_task(task, items, metrics)

    # ---------------- acdc ----------------

    def _run_acdc(self, task):
        job_id = self._resume_id(task)
        if not job_id:
            blobs = self._download_inputs(task)
            if len(blobs) != 2:
                raise TaskFailure("ACDC 需要 ED/ES 两个文件")
            inputs = task.get("inputs") or []
            if inputs[0].get("fileID") == inputs[1].get("fileID"):
                raise TaskFailure("ED 与 ES 指向同一云文件，请重新分别上传两个文件")
            params = task.get("params") or {}
            ed_name = params.get("ed_name") or blobs[0][0]
            es_name = params.get("es_name") or blobs[1][0]
            ed, es = blobs[0][1], blobs[1][1]

            def submit():
                state = self.board.acdc_health()
                if state.get("suspended"):
                    raise BoardError("ACDC 模型已暂停", "paused")
                if state.get("status") == "fault" or state.get("hdmi") == "error":
                    raise BoardError(state.get("hdmi_error") or "ACDC 服务故障", "fault")
                if not state.get("ready") or state.get("active_job"):
                    raise BoardError("ACDC 正在初始化或处理其他任务", "busy")
                if not state.get("task_auto_display"):
                    log.warning("ACDC 板端未声明任务级显示控制，仍可能按全局 auto_hdmi 显示结果")
                return self.board.acdc_submit(ed, es, ed_name, es_name)

            job = self._submit_with_retry(task["_id"], submit)
            job_id = job.get("run_id") or job.get("id")
            self._accepted(task, job_id)
        deadline = self.clock() + self.task_timeouts["acdc"]
        try:
            job = self._poll_with_retry(
                task["_id"],
                lambda: self.board.acdc_job(job_id),
                lambda j: j.get("status") == ACDC_DONE_STATUS,
                lambda j: j.get("status") in ACDC_BAD_STATUS,
                lambda j: j.get("error"),
                deadline,
                progress_of=lambda j: (self._acdc_stage(j), None),
                on_update=lambda j: self._early_preview(task, "acdc", job_id, j))
        except TaskFailure as exc:
            if "超时" in exc.message:
                self.board.acdc_cancel(job_id)
            raise

        self._completed(task, job_id)
        files = [*ACDC_MPR_FILES, ("statistics.csv", "统计表"), ("cardiac_function.csv", "心功能表"),
                 ("manifest.json", "清单"), ("result.zip", "结果包")]
        items = self._download_results(files, lambda name: self.board.acdc_file(job_id, name), task["_id"])
        contents = {name: data for name, _, data in items}
        stats_csv, func_csv = contents["statistics.csv"], contents["cardiac_function.csv"]
        metrics = self._acdc_metrics(func_csv, stats_csv)
        self._finish_task(task, items, metrics)

    @staticmethod
    def _acdc_stage(job):
        status = job.get("status") or ""
        if job.get("phase"):
            return "ACDC %s（%s）" % (status, job.get("phase"))
        if job.get("completed_slices") is not None:
            return "ACDC %s（%s/%s 切片）" % (status, job.get("completed_slices"),
                                             job.get("total_slices"))
        return "ACDC %s" % status

    @staticmethod
    def _acdc_metrics(func_csv, stats_csv):
        metrics = []
        try:
            for row in csv.DictReader(io.StringIO(func_csv.decode("utf-8-sig"))):
                if "structure" not in row:  # older metric/value CSV
                    if row.get("metric"):
                        metrics.append({"label": row["metric"], "value": row.get("value_mL", "")})
                    continue
                name = row["structure"]
                if row.get("valid", "true").lower() not in ("true", "1"):
                    metrics.append({"label": "%s 心功能" % name, "value": "无效：" + (row.get("reason") or "数据无效")})
                    continue
                for field in ("EDV_mL", "ESV_mL", "SV_mL", "EF_percent"):
                    if row.get(field) not in (None, ""):
                        unit = "%" if field == "EF_percent" else "mL"
                        metrics.append({"label": "%s %s" % (name, field.split("_")[0]),
                                        "value": "%s %s" % (row[field], unit)})
        except (ValueError, UnicodeError, csv.Error):
            pass
        try:
            for row in list(csv.DictReader(io.StringIO(stats_csv.decode("utf-8-sig"))))[:8]:
                if row.get("volume_mL") is not None:
                    metrics.append({"label": "%s · %s" % (row.get("phase", ""), row.get("structure", "")),
                                    "value": row["volume_mL"] + " mL"})
        except (ValueError, UnicodeError, csv.Error):
            pass
        return metrics

    # ---------------- main loops ----------------

    def run_forever(self):
        self.recover_stale_tasks(force=True)

        def beat():
            while not self._stop.is_set():
                try:
                    self.heartbeat_once()
                except Exception as exc:  # never let the heartbeat thread die
                    log.warning("心跳异常：%s", exc)
                self._stop.wait(self.heartbeat_interval)

        thread = threading.Thread(target=beat, name="heartbeat", daemon=True)
        thread.start()
        log.info("桥接已启动：%s", self.cfg.get("env_id"))
        while not self._stop.is_set():
            try:
                self.process_commands()
                self.process_task()
                self.recover_stale_tasks()
            except WechatApiError as exc:
                log.warning("云开发接口异常：%s", exc)
            except Exception as exc:  # keep the loop alive
                log.exception("桥接循环异常：%s", exc)
            self._stop.wait(self.poll_interval)

    def stop(self):
        self._stop.set()

    def close(self):
        self.stop()
        if self._result_executor is not None:
            self._result_executor.shutdown(wait=True)
        if hasattr(self.cloud, "close"):
            self.cloud.close()


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("app_id", "app_secret", "env_id", "board_host"):
        if not cfg.get(key):
            raise SystemExit("配置缺少 %s，请参考 config.example.json 填写 %s" % (key, path))
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description="PLD 小程序云开发桥接")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--once", action="store_true",
                        help="跑一轮命令+任务后退出（联调自检用）")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.handlers.RotatingFileHandler(
                      log_dir / "bridge.log", maxBytes=1024 * 1024,
                      backupCount=3, encoding="utf-8")])

    cloud = WechatCloud(cfg["app_id"], cfg["app_secret"], cfg["env_id"])
    board = BoardClient(cfg["board_host"], cfg.get("entry_port", 8088),
                        cfg.get("hub_port", 8765), cfg.get("acdc_port", 8092),
                        cfg.get("hub_token", ""))
    bridge = Bridge(cloud, board, cfg)
    if args.once:
        bridge.recover_stale_tasks(force=True)
        bridge.heartbeat_once(force=True)
        bridge.process_commands()
        bridge.process_task()
        bridge.close()
        return 0
    try:
        bridge.run_forever()
    except KeyboardInterrupt:
        log.info("收到退出信号，桥接停止")
        bridge.stop()
    finally:
        bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
