# -*- coding: utf-8 -*-
"""Real-board verification tool for the bridge's HTTP contracts.

Runs against the LIVE board exactly like the bridge does (same BoardClient
code paths). Read-only probes plus opt-in task/switch operations:

  python tools/board_probe.py --status
  python tools/board_probe.py --acdc-task out_demo/acdc_demo_1_ED.nii.gz out_demo/acdc_demo_1_ES.nii.gz
  python tools/board_probe.py --switch skin --hub-token <token>
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from board_client import BoardClient, BoardError  # noqa: E402


def make_client(args):
    return BoardClient(args.host, args.entry_port, args.hub_port,
                       args.acdc_port, args.hub_token or "")


def probe_status(board):
    print("== entry /entry/status (8088) ==")
    try:
        snap = board.entry_status()
        print(json.dumps(snap, ensure_ascii=False, indent=1))
    except BoardError as exc:
        print("entry 探测失败:", exc)
    print("== acdc /api/v2/health (8092) ==")
    try:
        health = board.acdc_health()
        keys = ("status", "ready", "active_job", "backend", "lifecycle",
                "suspended", "hardware_released", "hdmi", "synthetic")
        print({k: health.get(k) for k in keys})
    except BoardError as exc:
        print("acdc 探测失败:", exc)
    print("== hub /hub/status (8765, 仅在双模型运行时可达) ==")
    try:
        status = board.hub_status()
        print({k: status.get(k) for k in ("model", "phase", "generation")})
    except BoardError as exc:
        print("hub 不可达（当前应为 acdc 模型）:", exc.kind)


def probe_acdc_task(board, ed_path, es_path, timeout):
    ed = Path(ed_path).read_bytes()
    es = Path(es_path).read_bytes()
    print("提交 ACDC 任务: ED=%d B, ES=%d B" % (len(ed), len(es)))
    job = board.acdc_submit(ed, es, "ED.nii.gz", "ES.nii.gz")
    job_id = job.get("run_id") or job.get("id")
    print("提交成功:", job)
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        job = board.acdc_job(job_id)
        stage = dict(job)
        stage.pop("_files", None)
        if stage != last:
            print("  状态:", {k: stage.get(k) for k in
                             ("status", "phase", "phase_completed", "phase_total",
                              "completed_slices", "total_slices", "error")})
            last = stage
        if job.get("status") == "assets_ready":
            break
        if job.get("status") in ("failed", "cancelled", "interrupted",
                                 "upload_expired", "upload_failed"):
            print("任务失败:", job.get("error"))
            return 1
        time.sleep(1.0)
    else:
        print("超时未完成")
        board.acdc_cancel(job_id)
        return 1
    print("== 结果资产探测 ==")
    ok = True
    for name in ("mpr/ED/axial.png", "mpr/ES/axial.png", "statistics.csv",
                 "cardiac_function.csv", "manifest.json", "result.zip"):
        try:
            data = board.acdc_file(job_id, name)
            print("  %-24s %8d B" % (name, len(data)))
        except BoardError as exc:
            ok = False
            print("  %-24s 获取失败: %s" % (name, exc))
    print("ACDC_TASK_PROBE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _poll_job(board, poll, is_done, is_failed, error_of, timeout, label):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        job = poll()
        marker = {k: job.get(k) for k in ("state", "status", "stage", "error")}
        if marker != last:
            print("  状态:", marker)
            last = marker
        if is_done(job):
            return job
        if is_failed(job):
            print("任务失败:", error_of(job))
            return None
        time.sleep(1.0)
    print("超时未完成")
    return None


def _with_hdmi_guard(board, run):
    """Suppress entry auto-display around a probe task, restore after."""
    was = None
    try:
        snap = board.entry_status() or {}
        was = (snap.get("settings") or {}).get("auto_hdmi", True)
    except BoardError:
        pass
    if was:
        set_auto_hdmi(board, False)
    try:
        return run()
    finally:
        if was:
            set_auto_hdmi(board, True)


def probe_skin_task(board, image_path, timeout):
    import preprocess
    raw = Path(image_path).read_bytes()
    jpg = preprocess.prepare_skin_image(raw)
    print("提交皮肤镜任务: 原图 %d B → 合同化 JPG %d B" % (len(raw), len(jpg)))
    job = board.skin_submit(jpg, thickness=2, name=Path(image_path).name)
    job_id = job.get("id")
    print("提交成功:", {k: job.get(k) for k in ("id", "state", "stage")})
    job = _poll_job(board, lambda: board.skin_job(job_id),
                    lambda j: j.get("state") == "done",
                    lambda j: j.get("state") == "error",
                    lambda j: j.get("error"), timeout, "skin")
    if job is None:
        return 1
    print("views:", job.get("views"))
    ok = True
    for name in [v + ".png" for v in (job.get("views") or [])] + \
            ["report.json", "results.zip"]:
        try:
            data = board.skin_file(job_id, name)
            print("  %-22s %8d B" % (name, len(data)))
        except BoardError as exc:
            ok = False
            print("  %-22s 获取失败: %s" % (name, exc))
    stats = (job.get("statistics") or {}).get("area_percent")
    print("  面积占比: %s" % stats)
    print("SKIN_TASK_PROBE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def probe_camus_task(board, file_path, timeout):
    data = Path(file_path).read_bytes()
    name = Path(file_path).name
    print("提交 CAMUS 任务: %s %d B" % (name, len(data)))
    job = board.camus_submit([(name, data)], name)
    job_id = job.get("id")
    print("提交成功:", job)
    job = _poll_job(board, lambda: board.camus_job(job_id),
                    lambda j: j.get("state") == "done",
                    lambda j: j.get("state") == "error",
                    lambda j: j.get("error") or j.get("close_error"),
                    timeout, "camus")
    if job is None:
        return 1
    ok = True
    for name in ("overlay.png", "segmentation.png", "label.png",
                 "label_raw.png", "report.json", "result.zip"):
        try:
            data = board.camus_file(job_id, name)
            print("  %-22s %8d B" % (name, len(data)))
        except BoardError as exc:
            ok = False
            print("  %-22s 获取失败: %s" % (name, exc))
    print("CAMUS_TASK_PROBE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def probe_switch(board, model, timeout, expect_back=None):
    print("切换到 %s ..." % model)
    snap = board.entry_switch(model)
    print("  切换已受理:", {k: snap.get(k) for k in ("model", "phase")})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = board.entry_status() or {}
        if snap.get("phase") == "ready" and snap.get("model") == model:
            print("  就绪:", snap.get("model"))
            return 0
        if snap.get("phase") == "error":
            print("  切换失败:", snap.get("error"))
            return 1
        time.sleep(1.5)
    print("  切换超时")
    return 1


def set_auto_hdmi(board, value):
    """Temporarily toggle entry auto_hdmi so probe tasks never hijack HDMI."""
    import http.client
    conn = http.client.HTTPConnection(board.host, board.entry_port, timeout=10)
    body = json.dumps({"auto_hdmi": bool(value)}).encode()
    headers = {"Content-Type": "application/json",
               "Origin": "http://%s:%d" % (board.host, board.entry_port),
               "Content-Length": str(len(body))}
    conn.request("POST", "/entry/settings", body, headers)
    resp = conn.getresponse()
    data = json.loads(resp.read() or b"{}")
    conn.close()
    print("auto_hdmi -> %s : %s" % (value, data))
    return resp.status == 200


def board_idle(board):
    """True when the entry is ready and the active backend has no job."""
    try:
        snap = board.entry_status()
    except BoardError:
        return False, "entry 不可达"
    if snap is None:
        return False, "无三模型入口"
    if snap.get("phase") != "ready":
        return False, "entry phase=%s" % snap.get("phase")
    model = snap.get("model")
    if model == "acdc":
        try:
            health = board.acdc_health()
        except BoardError as exc:
            return False, "acdc health: %s" % exc.kind
        if health.get("status") == "fault":
            return False, "acdc fault"
        if health.get("active_job"):
            return False, "acdc 任务 %s 运行中" % health.get("active_job")
        if not (health.get("ready") or health.get("suspended")):
            return False, "acdc 未就绪"
        return True, "acdc 空闲"
    # skin / camus: hub status carries backend activity via last_backend
    try:
        status = board.hub_status()
    except BoardError as exc:
        return False, "hub: %s" % exc.kind
    if status.get("phase") != "ready":
        return False, "hub phase=%s" % status.get("phase")
    backend = status.get("backend") or {}
    if backend.get("active"):
        return False, "%s 任务 %s 运行中" % (model, backend.get("active"))
    return True, "%s 空闲" % model


def wait_idle(board, timeout, quiet_seconds=10.0):
    """Wait until the board has been continuously idle for quiet_seconds."""
    print("等待板卡空闲（连续 %.0fs 空闲即认为可用，最长 %ds）..."
          % (quiet_seconds, timeout))
    deadline = time.monotonic() + timeout
    idle_since = None
    while time.monotonic() < deadline:
        ok, why = board_idle(board)
        now = time.monotonic()
        if ok:
            idle_since = idle_since or now
            if now - idle_since >= quiet_seconds:
                print("BOARD_IDLE: %s" % why)
                return 0
        else:
            idle_since = None
            print("  忙: %s" % why)
        time.sleep(2.0)
    print("等待超时")
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.2.99")
    parser.add_argument("--entry-port", type=int, default=8088)
    parser.add_argument("--hub-port", type=int, default=8765)
    parser.add_argument("--acdc-port", type=int, default=8092)
    parser.add_argument("--hub-token", default="")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--acdc-task", nargs=2, metavar=("ED", "ES"))
    parser.add_argument("--skin-task")
    parser.add_argument("--camus-task")
    parser.add_argument("--switch")
    parser.add_argument("--set-auto-hdmi", choices=("on", "off"))
    parser.add_argument("--wait-idle", action="store_true",
                        help="等待板卡持续空闲后退出（用于整体验证前确认无人使用）")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    board = make_client(args)
    rc = 0
    if args.status:
        probe_status(board)
    if args.wait_idle:
        rc |= wait_idle(board, timeout=args.timeout)
    if args.skin_task:
        rc |= _with_hdmi_guard(
            board, lambda: probe_skin_task(board, args.skin_task,
                                           timeout=args.timeout))
    if args.camus_task:
        rc |= _with_hdmi_guard(
            board, lambda: probe_camus_task(board, args.camus_task,
                                            timeout=args.timeout))
    if args.set_auto_hdmi:
        rc |= 0 if set_auto_hdmi(board, args.set_auto_hdmi == "on") else 1
    if args.acdc_task:
        was = None
        try:
            snap = board.entry_status() or {}
            was = (snap.get("settings") or {}).get("auto_hdmi", True)
        except BoardError:
            pass
        if was:
            set_auto_hdmi(board, False)  # never hijack the live HDMI screen
        try:
            rc |= probe_acdc_task(board, *args.acdc_task, timeout=args.timeout)
        finally:
            if was:
                set_auto_hdmi(board, True)
    if args.switch:
        rc |= probe_switch(board, args.switch, timeout=args.timeout)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
