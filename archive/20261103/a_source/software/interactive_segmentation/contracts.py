"""Native-image coordinate contract shared by the two separate APIs."""
from dataclasses import dataclass
import math


class InputError(ValueError):
    pass


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise InputError(f'{name} 必须是有限数值')
    return float(value)


@dataclass(frozen=True)
class Rect:
    x0: int
    y0: int
    x1: int
    y1: int

    @classmethod
    def parse(cls, raw, width, height, name):
        if raw is None:
            return None
        if not isinstance(raw, dict) or set(raw) != {'x0', 'y0', 'x1', 'y1'}:
            raise InputError(f'{name} 需要 x0,y0,x1,y1，右下边界不包含在区域中')
        x0, y0, x1, y1 = [number(raw[k], f'{name}.{k}') for k in ('x0','y0','x1','y1')]
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise InputError(f'{name} 越界或为空，请在原图范围内重新框选')
        result = cls(math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))
        if result.x1-result.x0 < 2 or result.y1-result.y0 < 2:
            raise InputError(f'{name} 至少为 2×2 像素')
        return result

    def contains(self, x, y):
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def as_dict(self):
        return dict(x0=self.x0, y0=self.y0, x1=self.x1, y1=self.y1)


@dataclass(frozen=True)
class Point:
    x: int
    y: int
    foreground: bool


@dataclass(frozen=True)
class Request:
    model: str
    variant: str
    backend: str
    target_class: int
    threshold: float | None
    roi: Rect | None
    box: Rect | None
    points: tuple[Point, ...]

    @classmethod
    def parse(cls, model, body, width, height):
        if model not in ('skin','camus'):
            raise InputError('未知模型接口')
        if not isinstance(body, dict):
            raise InputError('请求必须为 JSON 对象')
        allowed = {'variant','backend','target_class','threshold','roi','box','points'}
        if set(body)-allowed:
            raise InputError('未知参数：'+', '.join(sorted(set(body)-allowed)))
        variant = body.get('variant', 'delivery')
        if variant not in (('delivery',) if model == 'skin' else ('delivery','candidate')):
            raise InputError('该接口不支持所选模型版本')
        backend = body.get('backend', 'int8')
        if backend not in ('int8','fp32'):
            raise InputError('backend 仅支持 int8 或 fp32')
        target = body.get('target_class', 1)
        if type(target) is not int or target not in ((1,) if model == 'skin' else (1,2,3)):
            raise InputError('target_class 无效：皮肤为1，CAMUS为1左室腔/2心肌/3左房')
        threshold = body.get('threshold', 0.45 if model == 'skin' else None)
        if threshold is None and model == 'skin':
            threshold = 0.45
        if threshold is not None:
            threshold = number(threshold, 'threshold')
            if not 0 < threshold < 1:
                raise InputError('threshold 必须大于0且小于1')
        roi = Rect.parse(body.get('roi'), width, height, 'roi')
        domain = roi or Rect(0,0,width,height)
        box = Rect.parse(body.get('box'), width, height, 'box')
        if box and not (domain.contains(box.x0, box.y0) and domain.contains(box.x1-1,box.y1-1)):
            raise InputError('提示框必须位于 ROI 内')
        raw_points = body.get('points', [])
        if not isinstance(raw_points, list) or len(raw_points) > 128:
            raise InputError('points 必须为列表，最多128个点')
        points, occupied = [], {}
        for i, p in enumerate(raw_points):
            if not isinstance(p, dict) or set(p) != {'x','y','label'}:
                raise InputError('每个提示点需要 x,y,label；label为foreground或background')
            if p['label'] not in ('foreground','background'):
                raise InputError('点标签必须为 foreground 或 background')
            x, y = number(p['x'], f'points[{i}].x'), number(p['y'], f'points[{i}].y')
            if not domain.contains(x,y):
                raise InputError('提示点必须位于图像和 ROI 内')
            x, y = min(width-1,math.floor(x)), min(height-1,math.floor(y))
            foreground = p['label'] == 'foreground'
            if foreground and box and not box.contains(x,y):
                raise InputError('前景点必须在提示框内')
            if (x,y) in occupied and occupied[x,y] != foreground:
                raise InputError('同一像素不能同时标记为前景和背景')
            occupied[x,y] = foreground
            points.append(Point(x,y,foreground))
        return cls(model,variant,backend,target,threshold,roi,box,tuple(points))

    @property
    def interactive(self):
        return bool(self.points or self.box)

