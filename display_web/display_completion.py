"""Board-only completion notifications; inference works without the LAN entry."""
import http.client
import json
import sys
import uuid


def notify(body, attempts=2):
    if sys.platform != 'linux': return {}
    error = ''
    for attempt in range(attempts):
        connection = http.client.HTTPConnection('127.0.0.1', 8088, timeout=2)
        try:
            connection.request('POST', '/entry/task', json.dumps(body).encode(),
                               {'Content-Type': 'application/json', 'Origin': 'http://127.0.0.1:8088'})
            response = connection.getresponse()
            value = json.loads(response.read())
            if response.status < 300 and isinstance(value, dict): return value
            error = 'entry HTTP ' + str(response.status)
            if response.status < 500: break
        except (OSError, ValueError, http.client.HTTPException) as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        finally:
            connection.close()
        print('DISPLAY_NOTIFY_RETRY', body.get('action'), attempt+1, error, file=sys.stderr, flush=True)
    print('DISPLAY_NOTIFY_ERROR', body.get('action'), error, file=sys.stderr, flush=True)
    return dict(accepted=False, error=error)


class Completion:
    def __init__(self, model, job, view=None, identity=None):
        self.context = dict(model=model, job=job, view=view, identity=identity, request_id=uuid.uuid4().hex)
        self.begin_reply = notify(dict(self.context, action='begin'))
        self.token = self.begin_reply.get('token')

    def finish(self, success, display_job=None):
        if self.token:
            return notify(dict(self.context, token=self.token, display_job=display_job,
                               action='done' if success else 'failed'))
        if sys.platform != 'linux': return dict(skipped=True)
        # Never start a replacement intent at finish: a stop or model switch
        # may have invalidated this task while inference was running.
        error = self.begin_reply.get('error') or '自动显示未取得任务凭据；入口可能已切换'
        print('DISPLAY_NOTIFY_ERROR', 'finish', error, file=sys.stderr, flush=True)
        return dict(accepted=False, error=error)


def user_display(action,model=None,job=None,view=None):
    # Manual actions are not replayed: retrying a late stop could cancel a
    # subsequent task. Only task begin/done use the idempotent retry protocol.
    return notify(dict(action='display',display_action=action,model=model,job=job,view=view), attempts=1)
