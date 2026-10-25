"""Serialize automatic HDMI intent with the existing model lifecycle."""
import http.client
import json
import re
import threading
import time
import uuid


class DisplayControl:
    def __init__(self, controller):
        self.controller = controller
        self.current = None
        self.pending = None
        self.error = ''
        self.changed = threading.Event()
        self.closed = threading.Event()
        self.thread = None
        self.restored = None
        self.result_path = controller.state_dir/'last_results.json' if controller.state_dir else None
        self.last_results = {}
        if self.result_path and self.result_path.exists():
            saved=json.loads(self.result_path.read_text())
            if isinstance(saved,dict):
                self.last_results={model:row for model,row in saved.items()
                    if model in ('acdc','skin','camus') and isinstance(row,dict) and row.get('model')==model
                    and isinstance(row.get('job'),str) and re.fullmatch('[0-9a-f]{32}',row['job'])
                    and isinstance(row.get('identity'),str) and row.get('view') in (None,'outlined','quad','overlay','segmentation','label','label_raw')}

    def save_results(self):
        if self.result_path:
            temporary=self.result_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(self.last_results))
            temporary.replace(self.result_path)

    def cancel(self):
        # Caller owns controller.lock. A completed old task cannot reopen HDMI.
        self.current = self.pending = None
        self.restored = None
        self.changed.set()

    def task(self, body):
        c = self.controller
        with c.lock:
            action = body.get('action')
            if action == 'preferences': return dict(c.settings)
            if action == 'display':
                if body.get('display_action') not in ('show', 'stop', 'leave'):
                    raise ValueError('未知显示操作')
                c.settings['auto_hdmi'] = body['display_action'] == 'show'
                previous=self.last_results.get(c.model)
                view=body.get('view')
                if (previous and body.get('model')==c.model and body.get('job')==previous['job']
                        and isinstance(view,str) and view in ('outlined','quad','overlay','segmentation','label','label_raw')):
                    previous['view']=view;self.save_results()
                c.save_settings()
                self.cancel()
                return dict(auto_hdmi=c.settings['auto_hdmi'])
            if c.phase != 'ready' or c.model != body.get('model'):
                return {}
            if action == 'begin':
                self.cancel()
                self.current = dict(model=c.model, job=body['job'], view=body.get('view'),
                                    identity=body.get('identity'),token=uuid.uuid4().hex)
                return dict(token=self.current['token'])
            if action in ('done', 'failed'):
                if (not self.current or body.get('token') != self.current['token']
                        or body.get('job') != self.current['job']):
                    return {}
                completed=dict(self.current,job=body.get('display_job') or self.current['job'])
                if action=='done':
                    if completed['view']=='interactive':
                        # A model-generation change invalidates the live ROI
                        # workspace. Do not present an older normal result as it.
                        self.last_results.pop(c.model,None)
                    elif isinstance(completed['identity'],str):
                        self.last_results[c.model]={k:completed[k] for k in ('model','job','view','identity')}
                    self.save_results()
                if action == 'done' and c.settings['auto_hdmi']:
                    self.pending = completed
                    self.error = ''
                    self.changed.set()
                    self.current = None
                else:
                    self.cancel()
                return dict(queued=self.pending is not None)
            raise ValueError('未知任务通知')

    def settings(self, body):
        c = self.controller
        with c.lock:
            for key in ('auto_hdmi', 'auto_segment'):
                if key in body:
                    if type(body[key]) is not bool: raise ValueError('设置必须是布尔值')
                    c.settings[key] = body[key]
            if not c.settings['auto_hdmi']: self.cancel()
            c.save_settings()
            return dict(c.settings)

    def snapshot(self):
        return dict(pending=dict(self.pending) if self.pending else None, error=self.error,restored=self.restored)

    def restore(self, model):
        c=self.controller
        with c.lock:
            saved=self.last_results.get(model)
            if not saved or not c.settings['auto_hdmi'] or self.current or self.pending:return
        try:
            if model=='acdc':
                status=c.board.health()
                identity=status.get('model_sha256','')+':'+str(status.get('runtime_sha256'))
                job=c.board.request(8092,'/api/v2/jobs/'+saved['job'])
                complete=job.get('status')=='assets_ready'
            else:
                hub=c.board.hub()
                if hub.get('model')!=model:return
                status=c.board.request(8765,'/'+model+'/g/'+str(hub['generation'])+'/api/status')
                identity=status.get('runtime_identity')
                job=c.board.request(8765,'/'+model+'/g/'+str(hub['generation'])+'/api/jobs/'+saved['job'])
                complete=job.get('state')=='done'
            with c.lock:
                if identity!=saved.get('identity') or not complete:
                    self.last_results.pop(model,None);self.save_results();return
                if c.model!=model or c.phase!='ready' or self.current or self.pending:return
                self.pending=dict(saved,token=uuid.uuid4().hex)
                self.restored=dict(model=model,job=saved['job'])
                self.changed.set()
        except (OSError,RuntimeError,ValueError,http.client.HTTPException) as exc:
            with c.lock:
                if getattr(exc,'status',None)==404:
                    self.last_results.pop(model,None);self.save_results()
                else:self.error='上次结果恢复失败：'+str(exc)

    def start(self):
        self.thread = threading.Thread(target=self.run, name='pld-auto-hdmi', daemon=True)
        self.thread.start()

    def close(self):
        self.closed.set(); self.changed.set()
        if self.thread: self.thread.join(timeout=65)

    def run(self):
        while not self.closed.is_set():
            self.changed.wait(.2); self.changed.clear()
            c = self.controller
            try:
                with c.proxy_lock:
                    with c.lock:
                        request = self.pending
                        if not request or c.phase != 'ready' or not c.settings['auto_hdmi']:
                            continue
                        if c.model != request['model']:
                            self.cancel(); continue
                    if request['model'] == 'acdc':
                        status = c.board.health()
                        busy = status.get('active_job') or status.get('hdmi') in ('starting', 'restoring')
                    else:
                        hub = c.board.hub()
                        status = c.board.request(8765, '/'+request['model']+'/g/'+str(hub['generation'])+'/api/status')
                        busy = (hub.get('interactive_busy') or hub.get('evaluation_busy')
                                or status.get('active') or status.get('warmup') == 'running')
                    if busy: continue
                    with c.lock:
                        if self.pending is not request or c.phase != 'ready': continue
                        self.pending = None
                    started = time.monotonic()
                    if request['model'] == 'acdc':
                        c.board.request(8092, '/api/v2/hdmi',
                            dict(action='show', job=request['job'], _automatic=True))
                    else:
                        c.board.request(8765, '/'+request['model']+'/g/'+str(hub['generation'])+'/api/hdmi',
                            dict(action='show', id=request['job'], view=request['view'] or 'quad', _automatic=True))
                    print('AUTO_HDMI_REQUEST', request['model'], request['job'],
                          round(time.monotonic()-started, 3), flush=True)
            except (OSError, RuntimeError, ValueError, http.client.HTTPException) as exc:
                with c.lock:
                    self.error = str(exc)
                    self.pending = None
                print('AUTO_HDMI_ERROR', str(exc), flush=True)
