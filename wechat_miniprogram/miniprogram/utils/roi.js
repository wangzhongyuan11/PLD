function point(touch, rect) {
  return { x: Math.max(0, Math.min(1, (touch.clientX - rect.left) / rect.width)),
    y: Math.max(0, Math.min(1, (touch.clientY - rect.top) / rect.height)) };
}
function box(a, b) {
  return { x0: Math.min(a.x, b.x), y0: Math.min(a.y, b.y),
    x1: Math.max(a.x, b.x), y1: Math.max(a.y, b.y) };
}
function style(r) {
  return "left:" + r.x0*100 + "%;top:" + r.y0*100 + "%;width:" + (r.x1-r.x0)*100 + "%;height:" + (r.y1-r.y0)*100 + "%;";
}
module.exports = { point: point, box: box, style: style };
