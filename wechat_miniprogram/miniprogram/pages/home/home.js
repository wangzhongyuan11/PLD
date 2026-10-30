var api = require("../../utils/api.js");

Page({
  data: {
    board: { online: false, model: "", phase: "", source: "" },
    models: [],
    switchTarget: "",
    switchCommand: "",
    switchError: "",
  },

  onLoad: function () {
    var list = [];
    ["skin", "camus", "acdc"].forEach(function (key) {
      list.push({ key: key, name: api.MODELS[key].name, desc: api.MODELS[key].desc });
    });
    this.setData({ models: list });
  },

  onShow: function () {
    this.startWatch();
    var self = this;
    this._expiryTimer = setInterval(function () { self.applyBoard(self._boardDoc); }, 5000);
    this.startCommandPoll();
  },

  onHide: function () { this.stopWatch(); },
  onUnload: function () { this.stopWatch(); },

  startWatch: function () {
    if (this._watcher) return;
    var self = this;
    var db = wx.cloud.database();
    try {
      this._watcher = db.collection("board").doc("board_current").watch({
      onChange: function (snap) {
        if (snap.docs && snap.docs.length) {
          self.applyBoard(snap.docs[0]);
        } else if (snap.doc && snap.doc.data) {
          self.applyBoard(snap.doc.data);
        } else if (snap.docType === "remove") {
          self.applyBoard(null);
        }
      },
      onError: function () {
        self.stopWatch();
        self.startPoll();
        self.startCommandPoll();
      },
      });
    } catch (e) {
      this.startPoll();
    }
  },

  startPoll: function () {
    if (this._timer) return;
    var self = this;
    var db = wx.cloud.database();
    var pull = function () {
      db.collection("board").doc("board_current").get({
        success: function (res) { self.applyBoard(res.data); },
        fail: function () { self.applyBoard(null); },
      });
    };
    pull();
    this._timer = setInterval(pull, 5000);
  },

  applyBoard: function (doc) {
    this._boardDoc = doc;
    this.setData({ board: api.boardSnapshot(doc) });
  },

  applyCommand: function (command) {
    if (!command || (command.status !== "done" && command.status !== "error")) return;
    this.setData({ switchTarget: "", switchCommand: "", switchError: command.error || "" });
    if (this._commandTimer) { clearInterval(this._commandTimer); this._commandTimer = null; }
    wx.showToast({ title: command.status === "done" ? "切换完成" : "切换失败，请重试", icon: "none" });
  },

  startCommandPoll: function () {
    if (!this.data.switchCommand || this._commandTimer) return;
    var self = this;
    var pull = function () {
      var id = self.data.switchCommand;
      if (!id) return;
      wx.cloud.database().collection("commands").doc(id).get({
        success: function (res) { if (self.data.switchCommand === id) self.applyCommand(res.data); },
        fail: function () {},
      });
    };
    this._commandTimer = setInterval(pull, 2000);
    pull();
  },

  stopWatch: function () {
    if (this._watcher) { this._watcher.close(); this._watcher = null; }
    if (this._timer) { clearInterval(this._timer); this._timer = null; }
    if (this._expiryTimer) { clearInterval(this._expiryTimer); this._expiryTimer = null; }
    if (this._commandTimer) { clearInterval(this._commandTimer); this._commandTimer = null; }
  },

  goModel: function (e) {
    var key = e.currentTarget.dataset.key;
    wx.navigateTo({ url: api.MODELS[key].page });
  },

  switchTo: function (e) {
    var self = this;
    var key = e.currentTarget.dataset.key;
    var board = this.data.board;
    if (!board.online) {
      wx.showToast({ title: "板卡不在线，请检查桥接", icon: "none" });
      return;
    }
    if (board.phase === "switching" || board.phase === "preparing" || this.data.switchTarget) {
      wx.showToast({ title: "正在切换中，请稍候", icon: "none" });
      return;
    }
    if (board.model === key && board.phase === "ready") return;
    if (key === "acdc" && board.source !== "entry") {
      wx.showToast({ title: "板卡未部署三模型入口，暂不可用", icon: "none" });
      return;
    }
    wx.showModal({
      title: "切换模型",
      content: "切换会影响板卡网页和 HDMI 当前使用的模型，需等待已有任务结束。确认切换？",
      success: function (res) {
        if (!res.confirm) return;
        var db = wx.cloud.database();
        db.collection("commands").add({
          data: { type: "switch", model: key, status: "pending", created_at: Date.now() },
          success: function (res) {
            self.setData({ switchTarget: key, switchCommand: res._id, switchError: "" });
            self.startCommandPoll();
            wx.showToast({ title: "切换指令已下发", icon: "success" });
          },
          fail: function () {
            wx.showToast({ title: "指令下发失败", icon: "none" });
          },
        });
      },
    });
  },
});
