// Node 直接测试 miniprogram/utils/api.js 的纯函数（无需微信环境）
const assert = require("assert");
const path = require("path");
const api = require(path.join(__dirname, "..", "miniprogram", "utils", "api.js"));

// statusInfo: 全部状态分支
assert.deepStrictEqual(api.statusInfo({ status: "pending" }, null),
  { text: "排队中", cls: "tag-info" });
assert.deepStrictEqual(api.statusInfo({ status: "pending", type: "skin" }, "camus"),
  { text: "等待切换模型", cls: "tag-warn" });
assert.deepStrictEqual(api.statusInfo({ status: "pending", type: "skin" }, "skin"),
  { text: "排队中", cls: "tag-info" });
assert.deepStrictEqual(api.statusInfo({ status: "waiting_model" }, null),
  { text: "等待切换模型", cls: "tag-warn" });
assert.deepStrictEqual(api.statusInfo({ status: "running" }, null),
  { text: "运行中", cls: "tag-info" });
assert.deepStrictEqual(api.statusInfo({ status: "done" }, null),
  { text: "已完成", cls: "tag-ok" });
assert.deepStrictEqual(api.statusInfo({ status: "error" }, null),
  { text: "失败", cls: "tag-err" });

// fmtTime: 毫秒时间戳 → 固定格式
const d = new Date(2026, 9, 24, 8, 5, 9);
const out = api.fmtTime(d.getTime());
assert.strictEqual(out, "10-24 08:05:09");
assert.strictEqual(api.fmtTime(0), "");
assert.strictEqual(api.fmtTime(undefined), "");

// genKey: 唯一性与格式
const seen = new Set();
for (let i = 0; i < 500; i++) seen.add(api.genKey());
assert.strictEqual(seen.size, 500);
assert.ok(/^[a-z0-9]+$/.test(api.genKey()));

// MODELS 元数据完整性（页面路由都指向存在的页面）
["skin", "camus", "acdc"].forEach((key) => {
  assert.ok(api.MODELS[key].name);
  assert.ok(api.MODELS[key].page.startsWith("/pages/"));
});

console.log("MINIPROGRAM_JS_LOGIC_PASS");
