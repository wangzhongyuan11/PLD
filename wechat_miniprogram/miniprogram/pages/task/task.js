var api = require("../../utils/api.js");
var album = require("../../utils/album.js");

Page({
  data: {
    id: "",
    task: null,
    views: [],
    previews: [],
    metrics: [],
    files: [],
    percent: 0,
    statusText: "",
    statusCls: "",
    typeName: "",
    boardModel: "",
    busy: "",
  },

  onLoad: function (options) {
    this.setData({ id: options.id || "" });
    this.startWatch();
    this.loadBoard();
  },

  onUnload: function () { this.stopWatch(); },
  onHide: function () { this.stopWatch(); },
  onShow: function () { if (this.data.id) { this.startWatch(); this.loadBoard(); } },

  loadBoard: function () {
    var self = this;
    var db = wx.cloud.database();
    db.collection("board").doc("board_current").get({
      success: function (res) {
        self.setData({ boardModel: (res.data && res.data.model) || "" });
        if (self.data.task) self.applyTask(self.data.task);
      },
      fail: function () {},
    });
  },

  startWatch: function () {
    if (this._watcher || this._timer || !this.data.id) return;
    var self = this;
    var db = wx.cloud.database();
    var pull = function () {
      db.collection("tasks").doc(self.data.id).get({
        success: function (res) { self.applyTask(res.data); },
        fail: function () {},
      });
    };
    pull();
    try {
      this._watcher = db.collection("tasks").doc(this.data.id).watch({
        onChange: function (snap) {
          if (snap.docs && snap.docs.length) self.applyTask(snap.docs[0]);
          else if (snap.doc && snap.doc.data) self.applyTask(snap.doc.data);
        },
        onError: function () {
          self.stopWatch();
          self._timer = setInterval(pull, 3000);
        },
      });
    } catch (e) {
      this._timer = setInterval(pull, 3000);
    }
  },

  stopWatch: function () {
    if (this._watcher) { this._watcher.close(); this._watcher = null; }
    if (this._timer) { clearInterval(this._timer); this._timer = null; }
  },

  applyTask: function (task) {
    if (!task) return;
    var info = api.statusInfo(task, this.data.boardModel);
    var result = task.result || {};
    var progress = task.progress || {};
    this.setData({
      task: task,
      typeName: (api.MODELS[task.type] && api.MODELS[task.type].name) || task.type,
      statusText: info.text,
      statusCls: info.cls,
      time: api.fmtTime(task.created_at),
      views: result.views || [],
      previews: task.status === "done" ? [] : (result.previews || []),
      metrics: result.metrics || [],
      files: result.files || [],
      zipFile: result.zip || "",
      percent: progress.percent || 0,
    });
  },

  previewView: function (e) {
    var fileID = e.currentTarget.dataset.fid;
    wx.cloud.downloadFile({
      fileID: fileID,
      success: function (res) {
        wx.previewImage({ urls: [res.tempFilePath] });
      },
      fail: function () {
        wx.showToast({ title: "图片下载失败", icon: "none" });
      },
    });
  },

  saveView: function (e) {
    if (this.data.busy) return;
    var fileID = e.currentTarget.dataset.fid;
    this.setData({ busy: fileID });
    var self = this;
    album.saveImage(fileID).then(function () {
      wx.showToast({ title: "已保存到相册", icon: "success" });
    }).catch(function (err) {
      wx.showModal({ title: "图片未保存", content: err.message, showCancel: false });
    }).finally(function () { self.setData({ busy: "" }); });
  },

  shareFile: function (e) {
    var fileID = e.currentTarget.dataset.fid;
    var name = e.currentTarget.dataset.name || "result.zip";
    this.setData({ busy: fileID });
    var self = this;
    wx.cloud.downloadFile({
      fileID: fileID,
      success: function (res) {
        wx.shareFileMessage({
          tempFilePath: res.tempFilePath,
          fileName: name,
          success: function () {},
          fail: function (err) {
            wx.showToast({
              title: (err && err.errMsg && err.errMsg.indexOf("cancel") >= 0)
                ? "已取消" : "转发失败，可稍后在电脑端网页下载",
              icon: "none",
            });
          },
          complete: function () { self.setData({ busy: "" }); },
        });
      },
      fail: function () {
        self.setData({ busy: "" });
        wx.showToast({ title: "下载失败", icon: "none" });
      },
    });
  },

  goSwitch: function () {
    wx.switchTab({ url: "/pages/home/home" });
  },

  retry: function (e) {
    var self = this;
    var task = this.data.task;
    if (!task || this._retrying) return;
    this._retrying = true;
    var resume = task.board_job && !(e && e.currentTarget.dataset.fresh);
    api.createTask({
      type: task.type,
      inputs: task.inputs,
      params: task.params || {},
      title: task.title,
      key: api.genKey(),
      resume_job: resume ? task.board_job : null,
      result: resume ? task.result : null,
    }).then(function (id) {
      wx.redirectTo({ url: "/pages/task/task?id=" + id });
    }).catch(function (err) {
      self._retrying = false;
      wx.showToast({ title: err.message || "重提失败", icon: "none" });
    });
  },
});
