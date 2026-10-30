// utils/api.js — 三模型共用的小程序侧工具
const ENV = "pld-d7gzvyxp10bdc9437";

const MODELS = {
  skin: {
    name: "皮肤镜分割",
    desc: "皮损照片 → 二值掩膜 / 轮廓 / 面积占比",
    page: "/pages/skin/skin",
  },
  camus: {
    name: "CAMUS 心超分割",
    desc: "超声图像或 NPZ/NIfTI → 多腔室分割与面积",
    page: "/pages/camus/camus",
  },
  acdc: {
    name: "ACDC 心脏 MRI",
    desc: "ED/ES NIfTI 体数据 → 心功能指标与 MPR 视图",
    page: "/pages/acdc/acdc",
  },
};

function fmtTime(ms) {
  if (!ms) return "";
  var d = new Date(ms);
  var pad = function (n) { return (n < 10 ? "0" : "") + n; };
  return pad(d.getMonth() + 1) + "-" + pad(d.getDate()) + " " +
    pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds());
}

function statusInfo(task, boardModel) {
  var s = task.status;
  if (s === "pending") {
    if (boardModel && boardModel !== task.type) {
      return { text: "等待切换模型", cls: "tag-warn" };
    }
    return { text: "排队中", cls: "tag-info" };
  }
  if (s === "waiting_model") return { text: "等待切换模型", cls: "tag-warn" };
  if (s === "running") return { text: "运行中", cls: "tag-info" };
  if (s === "done") return { text: "已完成", cls: "tag-ok" };
  if (s === "error") return { text: "失败", cls: "tag-err" };
  return { text: s || "未知", cls: "tag-info" };
}

function genKey() {
  return "k" + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function boardSnapshot(doc, now) {
  var fresh = !!(doc && doc.updated_at && (now == null ? Date.now() : now) - doc.updated_at < 60000);
  return {
    online: !!(doc && doc.online && fresh),
    model: (doc && doc.model) || "",
    phase: (doc && doc.phase) || "",
    stage: (doc && doc.stage) || "",
    source: (doc && doc.source) || "",
    error: (doc && doc.error) || "",
    updatedAt: fmtTime(doc && doc.updated_at),
  };
}

function uploadInput(key, file, index) {
  // file: {path, name}
  return new Promise(function (resolve, reject) {
    var cloudPath = "tasks/" + key + "/in/" + (index == null ? genKey() : index) + "/" + file.name;
    wx.cloud.uploadFile({
      cloudPath: cloudPath,
      filePath: file.path,
      success: function (res) {
        resolve({ fileID: res.fileID, name: file.name });
      },
      fail: function (err) {
        reject(new Error("上传失败：" + ((err && err.errMsg) || "未知错误")));
      },
    });
  });
}

function createTask(opts) {
  // opts: {type, inputs:[{fileID,name}], params, title, resume_job?, result?}
  var db = wx.cloud.database();
  return new Promise(function (resolve, reject) {
    var data = {
      type: opts.type,
      status: "pending",
      inputs: opts.inputs,
      params: opts.params || {},
      title: opts.title || "",
      key: opts.key,
      // resume_job/result 仅在确有值时写入。写入 null 会让桥接的嵌套结果回传
      // 失败（MongoDB 不能在 null 字段下建子字段，TCB 无法原地修复）。
      created_at: Date.now(),
      updated_at: Date.now(),
    };
    if (opts.resume_job) data.resume_job = opts.resume_job;
    if (opts.result) data.result = opts.result;
    db.collection("tasks").add({
      data: data,
      success: function (res) { resolve(res._id); },
      fail: function (err) {
        reject(new Error("任务创建失败：" + ((err && err.errMsg) || "未知错误")));
      },
    });
  });
}

function submitTaskFlow(opts) {
  // opts: {type, files:[{path,name}], params, title}
  var key = genKey();
  var uploads = [];
  wx.showLoading({ title: "上传输入中", mask: true });
  opts.files.forEach(function (f, i) { uploads.push(uploadInput(key, f, i)); });
  return Promise.all(uploads)
    .then(function (inputs) {
      return createTask({
        type: opts.type, inputs: inputs, params: opts.params,
        title: opts.title, key: key,
      });
    })
    .then(function (id) {
      wx.hideLoading();
      return id;
    })
    .catch(function (err) {
      wx.hideLoading();
      throw err;
    });
}

function ensureDemoCode() {
  // 按需求移除演示访问码门：直接使用。保留函数以免旧引用报错。
  return Promise.resolve(true);
}

// 选图/选文件失败必须在界面显示原因（隐私指引未声明时 API 直接 fail，
// 没有提示的话用户只看到"按钮没反应"）。
function pickFailToast(where, err) {
  var msg = (err && err.errMsg) || "未知错误";
  console.error(where + "失败：", err);
  wx.showToast({ title: where + "失败：" + msg, icon: "none", duration: 4000 });
}

module.exports = {
  ENV: ENV,
  MODELS: MODELS,
  fmtTime: fmtTime,
  boardSnapshot: boardSnapshot,
  statusInfo: statusInfo,
  genKey: genKey,
  uploadInput: uploadInput,
  createTask: createTask,
  submitTaskFlow: submitTaskFlow,
  ensureDemoCode: ensureDemoCode,
  pickFailToast: pickFailToast,
};
