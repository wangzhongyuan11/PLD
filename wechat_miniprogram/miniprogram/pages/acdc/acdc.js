var api = require("../../utils/api.js");

Page({
  data: {
    ed: null,      // {path, name}
    es: null,
    demos: [],
    submitting: false,
    demoRunning: "",
  },

  onShow: function () { this.loadDemos(); },

  loadDemos: function () {
    var self = this;
    var db = wx.cloud.database();
    db.collection("demos").where({ model: "acdc" }).limit(10).get({
      success: function (res) {
        self.setData({ demos: res.data || [] });
      },
      fail: function () { self.setData({ demos: [] }); },
    });
  },

  pickFile: function (e) {
    var slot = e.currentTarget.dataset.slot; // ed | es
    var self = this;
    wx.chooseMessageFile({
      count: 1,
      type: "file",
      extension: ["nii", "gz"],
      success: function (res) {
        var f = res.tempFiles && res.tempFiles[0];
        if (!f) return;
        var name = f.name || (slot === "ed" ? "ED.nii.gz" : "ES.nii.gz");
        if (!(name.endsWith(".nii") || name.endsWith(".nii.gz"))) {
          wx.showToast({ title: "请选择 .nii 或 .nii.gz 文件", icon: "none" });
          return;
        }
        var patch = {};
        patch[slot] = { path: f.path, name: name, size: f.size };
        self.setData(patch);
      },
      fail: function (err) { api.pickFailToast("选择病例文件", err); },
    });
  },

  clearSlot: function (e) {
    var slot = e.currentTarget.dataset.slot;
    var patch = {};
    patch[slot] = null;
    this.setData(patch);
  },

  submitFiles: function () {
    var self = this;
    if (this.data.submitting) return;
    if (!this.data.ed || !this.data.es) {
      wx.showToast({ title: "请分别选择 ED 与 ES 文件", icon: "none" });
      return;
    }
    if (this.data.ed.size + this.data.es.size > 32 * 1024 * 1024) {
      wx.showToast({ title: "ED 与 ES 总体积需在 32MiB 以内", icon: "none" });
      return;
    }
    this.setData({ submitting: true });
    api.submitTaskFlow({
      type: "acdc",
      files: [
        { path: this.data.ed.path, name: this.data.ed.name },
        { path: this.data.es.path, name: this.data.es.name },
      ],
      params: { ed_name: this.data.ed.name, es_name: this.data.es.name },
      title: "ACDC " + this.data.ed.name,
    }).then(function (id) {
      wx.redirectTo({ url: "/pages/task/task?id=" + id });
    }).catch(function (err) {
      self.setData({ submitting: false });
      wx.showToast({ title: err.message || "提交失败", icon: "none" });
    });
  },

  runDemo: function (e) {
    var self = this;
    var id = e.currentTarget.dataset.id;
    var demo = null;
    this.data.demos.forEach(function (d) { if (d._id === id) demo = d; });
    if (!demo || this.data.submitting || this.data.demoRunning) return;
    this.setData({ demoRunning: id, submitting: true });
    api.createTask({
      type: "acdc",
      inputs: demo.inputs,
      params: demo.params || {},
      title: demo.title || "ACDC 演示",
      key: api.genKey(),
    }).then(function (taskId) {
      wx.redirectTo({ url: "/pages/task/task?id=" + taskId });
    }).catch(function (err) {
      self.setData({ demoRunning: "", submitting: false });
      wx.showToast({ title: err.message || "提交失败", icon: "none" });
    });
  },
});
