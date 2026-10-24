"""Board-only completion notifications; inference works without the LAN entry."""
import http.client
import json
import sys


def notify(body):
    if sys.platform != 'linux': return {}
    connection = http.client.HTTPConnection('127.0.0.1', 8088, timeout=2)
    try:
        connection.request('POST', '/entry/task', json.dumps(body).encode(),
                           {'Content-Type': 'application/json', 'Origin': 'http://127.0.0.1:8088'})
        response = connection.getresponse()
        value = json.loads(response.read())
        return value if response.status < 300 else {}
    except (OSError, ValueError, http.client.HTTPException):
        return {}
    finally:
        connection.close()


class Completion:
    def __init__(self, model, job, view=None, identity=None):
        self.context = dict(model=model, job=job, view=view, identity=identity)
        self.token = notify(dict(self.context, action='begin')).get('token')

    def finish(self, success, display_job=None):
        if self.token:
            notify(dict(self.context, token=self.token, display_job=display_job,
                        action='done' if success else 'failed'))


def user_display(action,model=None,job=None,view=None):
    notify(dict(action='display',display_action=action,model=model,job=job,view=view))
