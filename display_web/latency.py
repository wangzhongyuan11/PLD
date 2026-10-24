"""Small monotonic stage trace for model/display lifecycle diagnostics."""
import json
import time


class Stages:
    def __init__(self, operation):
        self.operation = operation
        self.started = self.previous = time.monotonic()
        self.seconds = {}

    def mark(self, stage):
        now = time.monotonic()
        self.seconds[stage] = now - self.previous
        self.previous = now
        print('PLD_STAGE ' + json.dumps(dict(operation=self.operation, stage=stage,
            monotonic=now, seconds=self.seconds[stage], total=now-self.started)), flush=True)

    def snapshot(self):
        return dict(seconds=time.monotonic()-self.started, stages_seconds=dict(self.seconds))
