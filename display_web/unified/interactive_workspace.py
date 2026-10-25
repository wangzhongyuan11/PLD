"""Shared skin ROI controls for the browser and the existing HDMI child."""
import json

from prompt_seg.contracts import Request
from prompt_seg.service import png_data

TOOLS = ('foreground', 'background', 'roi', 'box')


def parameters():
    return dict(variant='delivery', backend='int8', target_class=1,
                threshold=.45, roi=None, box=None, points=[])


class SkinWorkspace:
    def reset_workspace(self):
        self.parameters = parameters()
        self.revision = 0
        self.result_revision = None
        self.tool = 'foreground'
        self.hdmi_requested = False

    def require_skin(self):
        if self.controller.model != 'skin':
            raise ValueError('ROI 交互模式仅支持皮肤镜')

    def state(self, image_token=None):
        try:from display_web.display_completion import notify
        except ModuleNotFoundError:from display_completion import notify
        preference=notify(dict(action='preferences'))
        if 'auto_hdmi' in preference:self.hdmi_requested=preference['auto_hdmi']
        with self.controller.lock:
            self.require_skin()
            with self.guard:
                slot = self.image_slot
                job = self.job
                result = dict(model='skin', generation=self.controller.generation,
                    phase=self.controller.phase, revision=self.revision, tool=self.tool,
                    parameters=json.loads(json.dumps(self.parameters)),
                    busy=self._busy(), hdmi_requested=self.hdmi_requested,
                    image_token=slot['token'] if slot else None,
                    job_id=job['id'] if job else None, job_state=job['state'] if job else None,
                    error=job.get('error', '') if job else '',
                    hdmi_error=getattr(self,'hdmi_error',''),
                    result_ready=bool(job and job['state']=='done' and self.result_revision==self.revision))
                if slot:
                    result.update(width=slot['width'], height=slot['height'])
                    if image_token != slot['token']:
                        result['image_png'] = slot['png']
                return result

    def control(self, body):
        with self.controller.lock:
            self.require_skin()
            self._check_context(body)
            if self._busy():
                raise RuntimeError('正在处理，请稍候')
            if not self.image_slot or body.get('image_token') != self.image_slot['token']:
                raise ValueError('图像已改变，请刷新后重试')
            if body.get('revision') != self.revision:
                raise ValueError('网页或触屏已更新提示，请同步后重试')
            if 'tool' in body:
                if body['tool'] not in TOOLS:
                    raise ValueError('未知交互工具')
                self.tool = body['tool']
            if 'parameters' in body:
                raw = dict(parameters(), **body['parameters'])
                req = Request.parse('skin', raw, self.image_slot['width'], self.image_slot['height'])
                if req.variant!='delivery' or req.backend!='int8' or req.target_class!=1 or req.threshold!=.45:
                    raise ValueError('皮肤 ROI 固定使用当前模型与 0.45 阈值')
                self.parameters = dict(parameters(), roi=None if req.roi is None else req.roi.as_dict(),
                    box=None if req.box is None else req.box.as_dict(),
                    points=[dict(x=p.x,y=p.y,label='foreground' if p.foreground else 'background') for p in req.points])
                self.revision += 1
            return self.state(body.get('image_token'))

    def workspace_upload(self, slot):
        self.parameters = parameters()
        self.revision = getattr(self, 'revision', 0) + 1
        self.result_revision = None
        slot['png'] = png_data(slot['image'])
        with self.guard:
            self.job = None

    def sync_hdmi(self, result):
        status = self.request('skin', 'GET', '/api/status')
        if status.get('hdmi') in ('starting', 'showing') and status.get('hdmi_view')=='interactive':
            return status
        return self.request('skin', 'POST', '/api/hdmi', dict(action='show',
            id=result['identity']['source_job'], view='interactive', _automatic=True))

    def set_hdmi(self, body):
        try:from display_web.display_completion import user_display
        except ModuleNotFoundError:from display_completion import user_display
        with self.controller.lock:
            self.require_skin()
            self._check_context(body)
            if self._busy():
                raise RuntimeError('请等待分割完成')
            action = body.get('action')
            if not body.get('_automatic'):user_display(action)
            if action in ('stop', 'leave'):
                self.hdmi_requested = False
                # The on-screen Back button exits its own child after this reply.
                if action=='stop':
                    self.request('skin', 'POST', '/api/hdmi', dict(action='stop'))
            elif action=='show':
                with self.guard:
                    job = self.job
                if not job or job['state']!='done' or self.result_revision!=self.revision:
                    raise ValueError('请先运行当前分割，再显示到 HDMI')
                self.sync_hdmi(job['result'])
                self.hdmi_requested = True
            else:
                raise ValueError('未知显示操作')
            return dict(hdmi_requested=self.hdmi_requested)
